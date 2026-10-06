"""
Training STARK-ST (stage 1 or 2) on any combination of GOT-10k, COCO, LaSOT and TrackingNet.

The training procedure is the original STARK one (lib/train/*): same data sampling and augmentation, losses,
optimizer, learning rates, LR schedule and number of samples per epoch. The original was trained on 8 GPUs with 16
samples each (128 per optimizer step); here the same effective batch is obtained on one GPU with gradient
accumulation (see lib/train/trainers/ltr_trainer.py).

Run folder (<train_outputs>/<run name>/):
    train_config.json    all parameters, dataset roots, initial weights, code hash
    checkpoints/         latest.pth.tar (full state, every epoch, for resuming)
                         + STARKST_epXXXX.pth.tar (weights only, every `keep_every` epochs and the last one)
    history.csv          per-epoch losses, learning rate and duration
    history.png          plot of history.csv (stark_ft/train/report.py)
    logs/train.log       progress output
    tensorboard/         TensorBoard logs
    final.pth.tar        final weights (written when training finishes)
A trained stage-2 model is tested with  checkpoint="train:<run name>", and a stage-1 run initialises stage 2 with
init="<stage-1 run name>". Stage 2 always needs an explicit `init`; init="official" (STARK's weights, trained on all
four datasets) has to be asked for by name. Nothing is written to the `checkpoints` folder.
"""
import datetime
import importlib
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from stark_ft.common import code_hash
from stark_ft.paths import REPO_ROOT, Paths, get_paths
from stark_ft.train.config import ROOT_KEY, TRAIN_DATASETS, VAL_DATASETS, TrainConfig

TRAINING_CODE = ("lib/train", "lib/models", "lib/config", "lib/utils", "model_configs/stark_st1",
                 "model_configs/stark_st2", "stark_ft/train")  # code that determines the training result
OFFICIAL_STAGE2 = "STARKST_ep0050.pth.tar"


# ---------------------------------------------------------------------- helpers
def training_code_hash() -> str:
    return code_hash(TRAINING_CODE)


def dataset_roots(paths: Paths) -> dict:
    return {"got10k": paths.train_data / "got10k" / "train", "coco": paths.train_data / "coco",
            "lasot": paths.train_data / "lasot", "trackingnet": paths.train_data / "trackingnet"}


def check_dataset(name: str, root: Path) -> Optional[str]:
    """Returns an error message if the dataset folder does not have the expected layout."""
    expected = {
        "got10k": [root / "list.txt"],
        "coco": [root / "annotations" / "instances_train2017.json", root / "images" / "train2017"],
        "lasot": [root],
        "trackingnet": [root / "TRAIN_0"],
    }[name]
    missing = [str(p) for p in expected if not p.exists()]
    return f"{name}: missing {missing}" if missing else None


def resolve_init(tc: TrainConfig, paths: Paths):
    """Returns (checkpoint path, keys to skip) for the initial weights of a stage-2 run; (None, None) for stage 1."""
    if tc.stage == 1:
        return None, None
    init = tc.init
    if not init:
        raise ValueError("stage 2 needs init: a stage-1 run name, a checkpoint path or 'official'")
    if init == "official":
        # The official STARK-ST checkpoint is the result of stage 2, whose backbone, transformer and box head are
        # the (frozen) stage-1 weights. Loading it without the classification head is identical to starting stage
        # 2 from the official stage-1 weights.
        path = paths.checkpoints / "stark_st2" / tc.model_config / OFFICIAL_STAGE2
        if not path.is_file():
            raise FileNotFoundError(f"Official checkpoint not found: {path}\n"
                                    "Download it first: `python -m stark_ft download-checkpoints` "
                                    "(notebook: nb.download_checkpoints()).")
        return path, ("cls_head.",)
    candidates = [paths.train_outputs / init / "final.pth.tar",
                  Path(init) if Path(init).is_absolute() else REPO_ROOT / init,
                  paths.checkpoints / "stark_st1" / tc.model_config / init]
    for c in candidates:
        if c.is_file():
            return c, ()
    raise FileNotFoundError(f"Stage-1 weights {init!r} not found. Looked at:\n  " +
                            "\n  ".join(str(c) for c in candidates))


def _load_model_cfg(tc: TrainConfig):
    module = importlib.reload(importlib.import_module(f"lib.config.{tc.stage_dir}.config"))
    module.update_config_from_file(str(tc.model_yaml()))
    cfg = module.cfg
    cfg.DATA.TRAIN.DATASETS_NAME = [TRAIN_DATASETS[d] for d in tc.datasets]
    cfg.DATA.TRAIN.DATASETS_RATIO = list(tc.dataset_ratios) if tc.dataset_ratios else [1] * len(tc.datasets)
    cfg.DATA.VAL.DATASETS_NAME = [VAL_DATASETS[d] for d in tc.val_datasets]
    cfg.DATA.VAL.DATASETS_RATIO = [1] * len(tc.val_datasets)
    if tc.samples_per_epoch is not None:
        cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = tc.samples_per_epoch
    if tc.val_samples_per_epoch is not None:
        cfg.DATA.VAL.SAMPLE_PER_EPOCH = tc.val_samples_per_epoch
    if tc.epochs is not None:
        cfg.TRAIN.EPOCH = tc.epochs
    if tc.lr_drop_epoch is not None:
        cfg.TRAIN.LR_DROP_EPOCH = tc.lr_drop_epoch
    if tc.val_interval is not None:
        cfg.TRAIN.VAL_EPOCH_INTERVAL = tc.val_interval
    cfg.TRAIN.BATCH_SIZE = tc.micro_batch
    cfg.TRAIN.NUM_WORKER = tc.num_workers
    # Stage 1 starts from ImageNet weights (as in STARK); stage 2 loads full weights anyway
    cfg.MODEL.BACKBONE.PRETRAINED = tc.stage == 1
    return cfg


def _init_seeds(seed):  # as in the original run_training.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe(tc: TrainConfig, paths: Paths = None) -> dict:
    """Validates the configuration and reports what a run would do (without starting it)."""
    tc.validate()
    paths = paths or get_paths()
    roots = dataset_roots(paths)
    used = sorted({ROOT_KEY[d] for d in tc.datasets} | {ROOT_KEY[d] for d in tc.val_datasets})
    problems = [p for p in (check_dataset(d, roots[d]) for d in used) if p]
    init_path, init_error = None, None
    try:
        init_path = resolve_init(tc, paths)[0]
    except FileNotFoundError as e:
        init_error = str(e)
    samples = tc.samples_per_epoch or int(tc.yaml_value("DATA", "TRAIN", "SAMPLE_PER_EPOCH"))
    return {
        "run_name": tc.run_name,
        "run_dir": str(paths.train_outputs / tc.run_name),
        "run_exists": (paths.train_outputs / tc.run_name).exists(),
        "export": str(paths.train_outputs / tc.run_name / "final.pth.tar"),
        "epochs": tc.total_epochs,
        "steps_per_epoch": samples // tc.effective_batch,
        "accumulation": tc.effective_batch // tc.micro_batch,
        "dataset_roots": {d: str(roots[d]) for d in used},
        "dataset_problems": problems,
        "init": str(init_path) if init_path else ("ImageNet" if tc.stage == 1 else None),
        "init_error": init_error,
        "config": tc.to_dict(),
    }


# ---------------------------------------------------------------------- main entry
def run_training(tc: TrainConfig, paths: Paths = None, overwrite: bool = False) -> Path:
    tc.validate()
    paths = paths or get_paths()
    info = describe(tc, paths)
    if info["dataset_problems"]:
        raise FileNotFoundError("Training data not found:\n  - " + "\n  - ".join(info["dataset_problems"]) +
                                "\nPrepare them first (`python -m stark_ft prepare-train-data`, notebook: "
                                "nb.prepare_training_data()) or set `train_data` in configs/paths.local.yaml.")
    if info["init_error"]:
        raise FileNotFoundError(info["init_error"])

    run_dir = paths.train_outputs / tc.run_name
    meta_path = run_dir / "train_config.json"
    code_hash = training_code_hash()
    if run_dir.exists():
        same = False
        if meta_path.is_file():
            old = json.loads(meta_path.read_text())
            try:
                same = TrainConfig.from_dict(old["config"]).to_dict() == tc.to_dict()
            except ValueError:
                same = False
            if same and old.get("code_hash") not in (None, code_hash) and not overwrite:
                raise RuntimeError(f"'{run_dir}' was started with a different version of the training code "
                                   f"({old.get('code_hash')} -> {code_hash}). Use overwrite=True for a fresh run.")
        if overwrite:
            if paths.train_outputs.resolve() not in run_dir.resolve().parents:
                raise RuntimeError(f"Safety check: {run_dir} is not inside train_outputs; not deleted")
            shutil.rmtree(run_dir)
        elif not same:
            raise RuntimeError(f"'{run_dir}' already exists with different parameters. "
                               "Use a different `name`, or overwrite=True to delete it.")

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    meta = {"config": tc.to_dict(), "run_name": tc.run_name, "code_hash": code_hash,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "dataset_roots": info["dataset_roots"], "init": info["init"],
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tf32_conv": torch.backends.cudnn.allow_tf32}
    if not meta_path.is_file():
        meta_path.write_text(json.dumps(meta, indent=2))

    _init_seeds(tc.seed)
    import cv2
    cv2.setNumThreads(0)  # as in the original run_training.py (avoids OpenCV crashes in data workers)

    from lib.models.stark import build_starkst
    from lib.train.actors import STARKSActor, STARKSTActor
    from lib.train.base_functions import build_dataloaders, get_optimizer_scheduler, update_settings
    from lib.train.trainers import LTRTrainer
    from lib.utils.box_ops import giou_loss
    from lib.utils.checkpoint import load_network_weights

    cfg = _load_model_cfg(tc)
    keep_every = tc.keep_every if tc.keep_every is not None else (50 if tc.stage == 1 else 10)
    settings = SimpleNamespace(
        script_name=tc.stage_dir, description=f"STARK-ST stage {tc.stage}", local_rank=-1,
        device=torch.device("cuda:0"), checkpoint_dir=str(run_dir / "checkpoints"),
        log_file=str(run_dir / "logs" / "train.log"), history_file=str(run_dir / "history.csv"),
        tensorboard_dir=str(run_dir / "tensorboard"), keep_every=keep_every, max_epochs=cfg.TRAIN.EPOCH,
        accum_steps=tc.effective_batch // tc.micro_batch, train_config=tc.to_dict(),
        deep_sup=False, distill=False, move_data_to_gpu=True)
    update_settings(settings, cfg)

    print(f"Run         : {tc.run_name}\nFolder      : {run_dir}")
    print(f"Stage {tc.stage}, {tc.model_config}, datasets {cfg.DATA.TRAIN.DATASETS_NAME} "
          f"(ratios {cfg.DATA.TRAIN.DATASETS_RATIO}), validation {cfg.DATA.VAL.DATASETS_NAME}")
    print(f"Epochs {cfg.TRAIN.EPOCH} (LR drop at {cfg.TRAIN.LR_DROP_EPOCH}), "
          f"{cfg.DATA.TRAIN.SAMPLE_PER_EPOCH} samples/epoch, effective batch {tc.effective_batch} "
          f"= {settings.accum_steps} x {tc.micro_batch}, {info['steps_per_epoch']} steps/epoch")
    print(f"Initial weights: {info['init']}", flush=True)

    loaders = build_dataloaders(cfg, settings, dataset_roots(paths))
    net = build_starkst(cfg).cuda()
    if tc.stage == 1:
        objective = {'giou': giou_loss, 'l1': torch.nn.functional.l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        actor = STARKSTActor(net=net, objective={'cls': torch.nn.BCEWithLogitsLoss()}, loss_weight={'cls': 1.0},
                             settings=settings)
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    init_path, skip = resolve_init(tc, paths)

    def init_fn(model):
        if init_path is None:
            return
        state = load_network_weights(str(init_path))
        state = {k: v for k, v in state.items() if not k.startswith(skip)} if skip else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        fresh = [k for k in missing if skip and k.startswith(skip)]
        print(f"Loaded initial weights from {init_path}" +
              (" (all except the classification head, which starts from random weights as in STARK stage 2)"
               if fresh else ""))
        other = [k for k in missing if k not in fresh]
        if other or unexpected:  # should not happen; shown so that a wrong checkpoint is noticed
            print(f"  WARNING: weights not found in the checkpoint: {other}\n  unused checkpoint weights: {unexpected}")

    trainer = LTRTrainer(actor, loaders, optimizer, settings, lr_scheduler)
    trainer.train(cfg.TRAIN.EPOCH, init_fn=init_fn)

    export = Path(info["export"])
    export.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"net": trainer.net.state_dict(), "net_type": type(trainer.net).__name__, "epoch": trainer.epoch,
                "train_config": tc.to_dict(), "code_hash": code_hash}, export)
    print(f"\nFinal weights exported to {export}")
    if tc.stage == 2:
        print(f"Evaluate with: ExperimentConfig(model_config={tc.model_config!r}, "
              f"checkpoint='train:{tc.run_name}')")
    else:
        print(f"Use as stage-2 initialisation with: TrainConfig(stage=2, init={tc.run_name!r}, ...)")
    return run_dir
