"""
Data loaders, optimizer and LR scheduler for STARK training (from the original STARK base_functions.py).

The data transforms, the sampler, the optimizer parameter groups and the scheduler are unchanged. Differences:
  - Dataset roots are passed explicitly (`dataset_roots`) instead of being read from a machine-specific local.py.
  - Only the datasets used by STARK-ST are supported (no LMDB / ImageNet-VID variants).
"""
import torch

import lib.train.data.transforms as tfm
from lib.train.data import LTRLoader, opencv_loader, processing, sampler
from lib.train.dataset import Got10k, Lasot, MSCOCOSeq, TrackingNet

# STARK dataset names -> (root key, constructor)
DATASET_BUILDERS = {
    "LASOT": ("lasot", lambda root, loader: Lasot(root, split="train", image_loader=loader)),
    "GOT10K_vottrain": ("got10k", lambda root, loader: Got10k(root, split="vottrain", image_loader=loader)),
    "GOT10K_votval": ("got10k", lambda root, loader: Got10k(root, split="votval", image_loader=loader)),
    "GOT10K_train_full": ("got10k", lambda root, loader: Got10k(root, split="train_full", image_loader=loader)),
    "COCO17": ("coco", lambda root, loader: MSCOCOSeq(root, version="2017", image_loader=loader)),
    "TRACKINGNET": ("trackingnet", lambda root, loader: TrackingNet(root, image_loader=loader)),
}


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR, 'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE, 'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list, dataset_roots, image_loader):
    datasets = []
    for name in name_list:
        if name not in DATASET_BUILDERS:
            raise ValueError(f"Unknown dataset {name}; supported: {list(DATASET_BUILDERS)}")
        key, build = DATASET_BUILDERS[name]
        datasets.append(build(str(dataset_roots[key]), image_loader))
    return datasets


def build_dataloaders(cfg, settings, dataset_roots):
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))
    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))
    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    common = dict(search_area_factor=settings.search_area_factor, output_sz=settings.output_sz,
                  center_jitter_factor=settings.center_jitter_factor,
                  scale_jitter_factor=settings.scale_jitter_factor, mode='sequence', settings=settings)
    data_processing_train = processing.STARKProcessing(transform=transform_train, joint_transform=transform_joint,
                                                       **common)
    data_processing_val = processing.STARKProcessing(transform=transform_val, joint_transform=transform_joint,
                                                     **common)

    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    dataset_train = sampler.TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, dataset_roots, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO, samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
        num_template_frames=settings.num_template, processing=data_processing_train,
        frame_sample_mode=sampler_mode, train_cls=train_cls)
    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1)
    loaders = [loader_train]

    if cfg.DATA.VAL.DATASETS_NAME:
        dataset_val = sampler.TrackingSampler(
            datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, dataset_roots, opencv_loader),
            p_datasets=cfg.DATA.VAL.DATASETS_RATIO, samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
            num_template_frames=settings.num_template, processing=data_processing_val,
            frame_sample_mode=sampler_mode, train_cls=train_cls)
        loaders.append(LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1,
                                 epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL))
    return loaders


def get_optimizer_scheduler(net, cfg):
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        # Stage 2: only the classification head is trained
        param_dicts = [{"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}]
        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER},
        ]
    if cfg.TRAIN.OPTIMIZER != "ADAMW":
        raise ValueError("Unsupported optimizer")
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
