"""
Training STARK-ST (stage 1 and/or stage 2) on any combination of GOT-10k, COCO, LaSOT and TrackingNet.

The training procedure is the original STARK one (lib/train/*): same data sampling and augmentation, losses,
optimizer, learning rates, LR schedule and number of samples per epoch. The original was trained on 8 GPUs with 16
samples each (128 per optimizer step); here the same effective batch is obtained on one GPU with gradient
accumulation (see lib/train/trainers/ltr_trainer.py).

Run folder (<train_outputs>/<run name>/):
    train_config.json    all parameters, dataset roots, initial weights, code hash
    checkpoints/         latest.pth.tar (full state, every epoch, for resuming)
                         + STARKST_epXXXX.pth.tar (weights only, every `keep_every` epochs and the last one)
    history.csv          per-epoch losses, learning rate and duration
    history.png          plot of history.csv (written by `python -m stark_ft train-report <run>`)
    logs/train.log       progress output
    tensorboard/         TensorBoard logs
    final.pth.tar        final weights (written when training finishes)
A trained stage-2 model is evaluated with  ExperimentConfig(model_config=..., checkpoint="train:<run name>"), and a
stage-1 run is used to initialise stage 2 with  TrainConfig(stage=2, init="<stage-1 run name>").
Nothing is written to the `checkpoints` folder (it may be a shared, read-only location).
"""
import datetime
import hashlib
import importlib
import json
import random
import shutil
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
import torch
import yaml

from stark_ft.config import _coerce
from stark_ft.paths import MODEL_CONFIG_DIR, REPO_ROOT, Paths, get_paths

# short name -> STARK dataset name used for training
TRAIN_DATASETS = {
    "got10k": "GOT10K_vottrain",       # GOT-10k train videos without the 1000 that overlap with VOT (7086 videos)
    "got10k_full": "GOT10K_train_full",  # all 9335 GOT-10k train videos (GOT-10k protocol, *_got10k_only configs)
    "coco": "COCO17",
    "lasot": "LASOT",
    "trackingnet": "TRACKINGNET",
}
VAL_DATASETS = {"got10k": "GOT10K_votval"}  # 1249 GOT-10k train videos, disjoint from GOT10K_vottrain
ROOT_KEY = {"got10k": "got10k", "got10k_full": "got10k", "coco": "coco", "lasot": "lasot",
            "trackingnet": "trackingnet"}  # dataset -> folder in <train_data>
# Order of DATA.TRAIN.DATASETS_NAME in the STARK configs. The order changes the random sampling stream, so it is
# normalised: ["coco", "got10k"] and ["got10k", "coco"] are the same run.
STARK_ORDER = ["lasot", "got10k", "got10k_full", "coco", "trackingnet"]
STAGE_DIR = {1: "stark_st1", 2: "stark_st2"}
TRAINING_CODE = ("lib/train", "lib/models", "lib/config", "lib/utils", "model_configs/stark_st1",
                 "model_configs/stark_st2", "stark_ft/training.py")  # code that determines the training result
OFFICIAL_STAGE2 = "STARKST_ep0050.pth.tar"


@dataclass
class TrainConfig:
    name: Optional[str] = None               # Run folder name; None = generated from the parameters
    model_config: str = "baseline_R101"      # baseline_R101 (STARK-ST101) | baseline (STARK-ST50)
    stage: int = 2                           # 1 = backbone + transformer + box head, 2 = classification head
    datasets: List[str] = field(default_factory=lambda: ["got10k"])  # got10k(_full), coco, lasot, trackingnet
    dataset_ratios: Optional[List[float]] = None   # sampling weights; None = equal (as in STARK)
    val_datasets: List[str] = field(default_factory=lambda: ["got10k"])  # [] = no validation
    init: Optional[str] = None               # stage 2: "official" (default) | stage-1 run name | checkpoint path
    epochs: Optional[int] = None             # None = model YAML (stage 1: 500, stage 2: 50)
    lr_drop_epoch: Optional[int] = None      # None = model YAML (stage 1: 400, stage 2: 40)
    samples_per_epoch: Optional[int] = None  # None = model YAML (60000)
    val_samples_per_epoch: Optional[int] = None  # None = model YAML (10000)
    effective_batch: int = 128               # samples per optimizer step (original: 8 GPUs x 16)
    micro_batch: int = 16                    # samples per forward/backward pass; effective_batch / micro_batch
                                             # passes are accumulated per optimizer step
    num_workers: int = 8                     # data loading processes
    seed: int = 42                           # STARK default
    val_interval: Optional[int] = None       # None = model YAML (stage 1: 20, stage 2: 10)
    keep_every: Optional[int] = None         # keep a numbered checkpoint every N epochs (None: 50 / 10)

    def __post_init__(self):
        ds = list(self.datasets) if isinstance(self.datasets, (list, tuple)) else self.datasets
        if isinstance(ds, list) and len(set(ds)) == len(ds) and all(d in STARK_ORDER for d in ds):
            order = sorted(range(len(ds)), key=lambda i: STARK_ORDER.index(ds[i]))
            self.datasets = [ds[i] for i in order]
            if isinstance(self.dataset_ratios, (list, tuple)) and len(self.dataset_ratios) == len(ds):
                self.dataset_ratios = [self.dataset_ratios[i] for i in order]

    # ------------------------------------------------------------------
    @property
    def stage_dir(self) -> str:
        return STAGE_DIR.get(self.stage, "?")

    def model_yaml(self) -> Path:
        return MODEL_CONFIG_DIR / self.stage_dir / f"{self.model_config}.yaml"

    def validate(self):
        errors = []
        if self.stage not in STAGE_DIR:
            errors.append("stage must be 1 or 2")
        elif not self.model_yaml().is_file():
            options = sorted(p.stem for p in (MODEL_CONFIG_DIR / self.stage_dir).glob("*.yaml"))
            errors.append(f"model_config={self.model_config!r} not found for stage {self.stage}; options: {options}")
        unknown = [d for d in self.datasets if d not in TRAIN_DATASETS]
        if not self.datasets or unknown:
            errors.append(f"datasets must be a non-empty subset of {list(TRAIN_DATASETS)} (got {self.datasets})")
        if len(set(self.datasets)) != len(self.datasets):
            errors.append("datasets contains duplicates")
        if {"got10k", "got10k_full"} <= set(self.datasets):
            errors.append("use either got10k or got10k_full (got10k_full contains got10k)")
        if self.dataset_ratios is not None and len(self.dataset_ratios) != len(self.datasets):
            errors.append("dataset_ratios must have one value per dataset")
        bad_val = [d for d in self.val_datasets if d not in VAL_DATASETS]
        if bad_val:
            errors.append(f"val_datasets must be a subset of {list(VAL_DATASETS)}")
        if self.effective_batch % self.micro_batch != 0:
            errors.append("effective_batch must be a multiple of micro_batch")
        if self.stage == 1 and self.init not in (None, "imagenet"):
            errors.append("stage 1 starts from ImageNet weights; init must be None or 'imagenet'")
        if self.name is not None and (not self.name or any(c in self.name for c in '/\\:*?"<>| ')):
            errors.append(f"name={self.name!r} cannot be used as a folder name")
        if errors:
            raise ValueError("Invalid training configuration:\n  - " + "\n  - ".join(errors))
        return self

    def yaml_value(self, *keys):
        with open(self.model_yaml()) as f:
            value = yaml.safe_load(f)
        for k in keys:
            value = value[k]
        return value

    @property
    def total_epochs(self) -> int:
        return self.epochs if self.epochs is not None else int(self.yaml_value("TRAIN", "EPOCH"))

    @property
    def run_name(self) -> str:
        if self.name:
            return self.name
        model = "st101" if "R101" in self.model_config else "st50"
        if "got10k_only" in self.model_config:
            model += "got"
        parts = [model, f"stage{self.stage}", "+".join(self.datasets)]
        if self.stage == 2 and self.init not in (None, "official"):
            parts.append("from-" + Path(self.init).name.split(".")[0])
        if self.epochs is not None:
            parts.append(f"e{self.epochs}")
        if self.dataset_ratios is not None:
            parts.append("r" + "-".join(f"{r:g}" for r in self.dataset_ratios))
        parts.append(f"s{self.seed}")
        return "_".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        types = {f.name: f.type for f in fields(cls)}
        unknown = set(data) - set(types)
        if unknown:
            raise ValueError(f"Unknown training parameter(s): {sorted(unknown)}")
        return cls(**{k: _coerce_field(k, v, types[k]) for k, v in data.items()})

    @classmethod
    def from_file(cls, path) -> "TrainConfig":
        path = Path(path)
        with open(path) as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        if "config" in data and isinstance(data["config"], dict):
            data = data["config"]
        return cls.from_dict(data)


def _coerce_field(name, value, type_):
    """_coerce for Optional[...] and List[...] fields ("3" -> 3, ["1", "2.5"] -> [1.0, 2.5], "none" -> None)."""
    if getattr(type_, "__origin__", None) is Union:  # Optional[X]
        if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null", "")):
            return None
        type_ = next(a for a in type_.__args__ if a is not type(None))
    if getattr(type_, "__origin__", None) in (list, List) and isinstance(value, (list, tuple)):
        return [_coerce(name, v, type_.__args__[0]) for v in value]
    return _coerce(name, value, type_)


# ---------------------------------------------------------------------- helpers
def training_code_hash() -> str:
    h = hashlib.sha1()
    for entry in TRAINING_CODE:
        root = REPO_ROOT / entry
        for f in (sorted(root.rglob("*")) if root.is_dir() else [root]):
            if f.is_file() and f.suffix in (".py", ".yaml"):
                h.update(str(f.relative_to(REPO_ROOT)).encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


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
    init = tc.init or "official"
    if init == "official":
        # The official STARK-ST checkpoint is the result of stage 2, whose backbone, transformer and box head are
        # the (frozen) stage-1 weights. Loading it without the classification head is identical to starting stage
        # 2 from the official stage-1 weights.
        path = paths.checkpoints / "stark_st2" / tc.model_config / OFFICIAL_STAGE2
        if not path.is_file():
            raise FileNotFoundError(f"Official checkpoint not found: {path}\n"
                                    "Download it first (00_setup / `python -m stark_ft download-checkpoints`).")
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
                                "\nPrepare the datasets first (03_train.ipynb) or set `train_data` in "
                                "configs/paths.local.yaml.")
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


# ---------------------------------------------------------------------- reporting
def find_run(name_or_path, paths: Paths = None) -> Path:
    """A run folder given by its name (in train_outputs) or its path."""
    p = Path(str(name_or_path))
    if (p / "train_config.json").is_file():
        return p
    run_dir = (paths or get_paths()).train_outputs / str(name_or_path)
    if not (run_dir / "train_config.json").is_file():
        raise FileNotFoundError(f"No training run {str(name_or_path)!r} (looked for {run_dir}/train_config.json)")
    return run_dir


def _history(run_dir: Path):
    import pandas as pd
    path = run_dir / "history.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    h = pd.read_csv(path)
    return h.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def run_status(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "train_config.json").read_text())
    tc = TrainConfig.from_dict(meta["config"])
    status = {"run_name": meta.get("run_name", run_dir.name), "run_dir": str(run_dir), "stage": tc.stage,
              "model_config": tc.model_config, "datasets": tc.datasets, "init": meta.get("init"),
              "gpu": meta.get("gpu"), "total_epochs": tc.total_epochs, "epochs_done": 0,
              "finished": (run_dir / "final.pth.tar").is_file(), "last": {}}
    h = _history(run_dir)
    if h is not None and len(h):
        done = int(h["epoch"].iloc[-1])
        epoch_s = float(h["seconds"].tail(10).mean())
        status.update(epochs_done=done, epoch_seconds=epoch_s,
                      remaining_hours=max(0, status["total_epochs"] - done) * epoch_s / 3600)
        for col in h.columns:
            if "/" in col:
                values = h[col].dropna()
                if len(values):
                    status["last"][col] = {"epoch": int(h.loc[values.index[-1], "epoch"]),
                                           "value": float(values.iloc[-1])}
    return status


def report(name_or_path, plot=True, paths: Paths = None) -> dict:
    """Status of a training run; writes history.png into the run folder."""
    run_dir = find_run(name_or_path, paths)
    status = run_status(run_dir)
    h = _history(run_dir)
    if plot and h is not None and len(h):
        from stark_ft.plots import plot_training_history
        plot_training_history(h, run_dir / "history.png", status["run_name"])
        status["plot"] = str(run_dir / "history.png")
    return status


def list_runs(paths: Paths = None):
    root = (paths or get_paths()).train_outputs
    if not root.is_dir():
        return []
    return [run_status(p) for p in sorted(root.iterdir()) if (p / "train_config.json").is_file()]
