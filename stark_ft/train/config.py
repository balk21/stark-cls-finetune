"""
Training parameters (TrainConfig). Their meaning is explained in docs/train.md.
"""
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml

from stark_ft.common import coerce
from stark_ft.paths import MODEL_CONFIG_DIR

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
SEED = 42  # STARK's training seed; every run uses it


@dataclass
class TrainConfig:
    name: Optional[str] = None               # Run folder name; None = generated from the parameters
    model_config: str = "baseline_R101"      # baseline_R101 (STARK-ST101) | baseline (STARK-ST50)
    stage: int = 1                           # 1 = backbone + transformer + box head, 2 = classification head
    datasets: List[str] = field(default_factory=lambda: ["got10k"])  # got10k(_full), coco, lasot, trackingnet
    dataset_ratios: Optional[List[float]] = None   # sampling weights; None = equal (as in STARK)
    val_datasets: List[str] = field(default_factory=lambda: ["got10k"])  # [] = no validation
    init: Optional[str] = None               # stage 2: None = our stage-1 run with the same model and datasets
                                             #   | another stage-1 run name | "official" (STARK's weights) | path
    epochs: Optional[int] = None             # None = model YAML (stage 1: 500, stage 2: 50)
    lr_drop_epoch: Optional[int] = None      # None = model YAML (stage 1: 400, stage 2: 40)
    samples_per_epoch: Optional[int] = None  # None = model YAML (60000)
    val_samples_per_epoch: Optional[int] = None  # None = model YAML (10000)
    effective_batch: int = 128               # samples per optimizer step (original: 8 GPUs x 16)
    micro_batch: int = 16                    # samples per forward/backward pass; effective_batch / micro_batch
                                             # passes are accumulated per optimizer step
    num_workers: int = 8                     # data loading processes
    val_interval: Optional[int] = None       # None = model YAML (stage 1: 20, stage 2: 10)
    keep_every: Optional[int] = None         # keep a numbered checkpoint every N epochs (None: 50 / 10)

    def __post_init__(self):
        ds = list(self.datasets) if isinstance(self.datasets, (list, tuple)) else self.datasets
        if isinstance(ds, list) and len(set(ds)) == len(ds) and all(d in STARK_ORDER for d in ds):
            order = sorted(range(len(ds)), key=lambda i: STARK_ORDER.index(ds[i]))
            self.datasets = [ds[i] for i in order]
            if isinstance(self.dataset_ratios, (list, tuple)) and len(self.dataset_ratios) == len(ds):
                self.dataset_ratios = [self.dataset_ratios[i] for i in order]
        if self.stage == 2 and not self.init and isinstance(self.datasets, list):
            self.init = self.own_stage1  # our own stage 1 of the same family, never the official weights silently

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
    def family(self) -> str:
        """Model + datasets (+ sampling ratios): the runs of one family belong together, e.g. st101_coco."""
        model = "st101" if "R101" in self.model_config else "st50"
        family = f"{model}_{'+'.join(self.datasets)}"
        if self.dataset_ratios is not None:
            family += "_r" + "-".join(f"{r:g}" for r in self.dataset_ratios)
        return family

    @property
    def own_stage1(self) -> str:
        """Default name of the stage-1 run of this family, i.e. of the same parameters with stage=1 (the default init
        of stage 2)."""
        return f"{self.family}_stage1" + (f"_e{self.epochs}" if self.epochs is not None else "")

    @property
    def run_name(self) -> str:
        """e.g. st101_coco_stage1, st101_coco_stage2 (on our own stage 1), st101_coco_stage2_on-official (on STARK's
        weights), st101_coco_stage2_from-<run> (on another stage-1 run)."""
        if self.name:
            return self.name
        parts = [self.family, f"stage{self.stage}"]
        if self.epochs is not None:
            parts.append(f"e{self.epochs}")
        if self.stage == 2 and self.init != self.own_stage1:
            if self.init == "official":
                parts.append("on-official" + ("-got10k" if "got10k_only" in self.model_config else ""))
            else:
                parts.append("from-" + Path(self.init).name.split(".")[0])
        return "_".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        data = dict(data)
        if "seed" in data:  # earlier parameter; every run uses SEED
            if coerce("seed", data.pop("seed"), int) != SEED:
                raise ValueError(f"The training seed is fixed ({SEED}, the STARK default)")
        types = {f.name: f.type for f in fields(cls)}
        unknown = set(data) - set(types)
        if unknown:
            raise ValueError(f"Unknown training parameter(s): {sorted(unknown)}")
        return cls(**{k: coerce(k, v, types[k]) for k, v in data.items()})

    @classmethod
    def from_file(cls, path) -> "TrainConfig":
        path = Path(path)
        with open(path) as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        if "config" in data and isinstance(data["config"], dict):
            data = data["config"]
        return cls.from_dict(data)
