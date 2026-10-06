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
Our runs are kept apart from STARK's published training: stage 1 starts from ImageNet, stage 2 by default from our
own stage-1 run of the same family (model + datasets + seed), so the result is trained only on the run's datasets
(origin "imagenet"). Stage 2 on STARK's weights must be asked for (init="official"); such runs are named
"..._on-official_..." and listed separately (origin "official"). A finished stage-2 run is tested with
weights="<run name>" (stark_ft/weights.py). Nothing is written to the `checkpoints` folder.
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
from stark_ft.setup_utils import download_checkpoints
from stark_ft.train.config import ROOT_KEY, TRAIN_DATASETS, VAL_DATASETS, TrainConfig
from stark_ft.weights import read_run

# Code that determines the training result (the parameters themselves are compared via train_config.json)
TRAINING_CODE = ("lib/train", "lib/models", "lib/config", "lib/utils", "model_configs/stark_st1",
                 "model_configs/stark_st2", "stark_ft/train/run.py")
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


def resolve_init(tc: TrainConfig, paths: Paths, download: bool = False):
    """(checkpoint path, keys to skip, origin) of the initial weights; (None, None, "imagenet") for stage 1.
    origin: "imagenet" (our own chain), "official" (STARK's weights) or "external" (another checkpoint file).
    download=True downloads the official weights if they are missing."""
    if tc.stage == 1:
        return None, None, "imagenet"
    init = tc.init
    if init == "official":
        # The official STARK-ST checkpoint is the result of stage 2, whose backbone, transformer and box head are
        # the (frozen) stage-1 weights. Loading it without the classification head is identical to starting stage
        # 2 from the official stage-1 weights.
        path = paths.checkpoints / "stark_st2" / tc.model_config / OFFICIAL_STAGE2
        if not path.is_file() and download:
            download_checkpoints([("stark_st", tc.model_config)])
        return path, ("cls_head.",), "official"
    run_dir = paths.train_outputs / init
    if (run_dir / "train_config.json").is_file():
        meta = read_run(run_dir)
        if meta["config"].get("stage") != 1:
            raise ValueError(f"init={init!r} is not a stage-1 run; stage 2 starts from a stage-1 run")
        if not (run_dir / "final.pth.tar").is_file():
            raise FileNotFoundError(f"The stage-1 run {init!r} has not finished yet (no final.pth.tar).")
        return run_dir / "final.pth.tar", (), meta["origin"]
    for c in (Path(init) if Path(init).is_absolute() else REPO_ROOT / init,
              paths.checkpoints / "stark_st1" / tc.model_config / init):
        if c.is_file():
            return c, (), "external"
    finished = sorted(p.parent.name for p in paths.train_outputs.glob("*/final.pth.tar")
                      if read_run(p.parent)["config"].get("stage") == 1) if paths.train_outputs.is_dir() else []
    default = init == tc.own_stage1
    raise FileNotFoundError(
        f"Stage 2 starts from {'the stage-1 run of the same family' if default else 'the stage-1 run'} {init!r}, "
        f"which is not in {paths.train_outputs}.\n"
        + ("Train it first (the same parameters with stage=1), " if default else "")
        + "or set init to a finished stage-1 run, a checkpoint path or 'official' (STARK's weights; the run is then "
        f"marked on-official).\nFinished stage-1 runs: {finished or 'none'}")


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
    init_path, init_error, origin = None, None, None
    try:
        init_path, _, origin = resolve_init(tc, paths)
    except (FileNotFoundError, ValueError) as e:
        init_error = str(e)
    if tc.init == "official" and init_path and not init_path.is_file():
        init_text = f"{init_path} (official STARK weights; downloaded when the run starts)"
    else:
        init_text = str(init_path) if init_path else ("ImageNet" if tc.stage == 1 else None)
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
        "init": init_text if not init_error else None,
        "init_error": init_error,
        "origin": origin,
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
            "dataset_roots": info["dataset_roots"], "init": info["init"], "origin": info["origin"],
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

    init_path, skip, _ = resolve_init(tc, paths, download=True)

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
                "train_config": tc.to_dict(), "origin": info["origin"], "code_hash": code_hash}, export)
    print(f"\nFinal weights exported to {export}")
    if tc.stage == 2:
        print(f"Test it with: weights={tc.run_name!r} (and model_config={tc.model_config!r})")
    else:
        print(f"Stage 2 of this family: the same parameters with stage=2 (init={tc.run_name!r})")
    return run_dir
