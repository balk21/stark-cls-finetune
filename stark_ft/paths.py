"""
Path resolution. There are no hard-coded (absolute) paths anywhere in the code.

Priority (later entries override earlier ones):
  1. Defaults (relative to the repository root):  checkpoints/, data/votlt2020/sequences/, outputs/,
     data/train/, outputs/training/
  2. configs/paths.yaml          (in the repository; relative paths are resolved against the repository root)
  3. configs/paths.local.yaml    (ignored by git; for machine-specific settings)
  4. Environment variables: STARK_CLEAN_CHECKPOINTS, STARK_CLEAN_DATASET, STARK_CLEAN_DATASET_CACHE,
     STARK_CLEAN_OUTPUTS, STARK_CLEAN_TRAIN_DATA, STARK_CLEAN_TRAIN_OUTPUTS, STARK_CLEAN_ARCHIVES
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_DIR = REPO_ROOT / "model_configs"

_KEYS = {
    "checkpoints": "STARK_CLEAN_CHECKPOINTS",
    "dataset": "STARK_CLEAN_DATASET",
    "dataset_cache": "STARK_CLEAN_DATASET_CACHE",
    "outputs": "STARK_CLEAN_OUTPUTS",
    "train_data": "STARK_CLEAN_TRAIN_DATA",
    "train_outputs": "STARK_CLEAN_TRAIN_OUTPUTS",
    "archives": "STARK_CLEAN_ARCHIVES",
}
_DEFAULTS = {
    "checkpoints": "checkpoints",
    "dataset": "data/votlt2020/sequences",
    "dataset_cache": None,
    "outputs": "outputs",
    "train_data": "data/train",
    "train_outputs": "outputs/training",
    "archives": None,
}


@dataclass
class Paths:
    checkpoints: Path  # <checkpoints>/<stark_st2|stark_s>/<model_config>/<checkpoint file>
    dataset: Path      # VOT-format sequence folder: <dataset>/<sequence>/{color/, groundtruth.txt, sequence}
    outputs: Path      # Each experiment is written to <outputs>/<experiment name>/
    train_data: Path   # Training datasets: <train_data>/{got10k/train, coco, lasot, trackingnet}
    train_outputs: Path  # Each training run is written to <train_outputs>/<run name>/
    dataset_cache: Optional[Path] = None  # Optional: a copy of every downloaded sequence (<sequence>.tar)
    archives: Optional[Path] = None   # Dataset archives (COCO, GOT-10k); default <train_data>/_archives

    def as_dict(self):
        return {k: (str(v) if v is not None else None) for k, v in self.__dict__.items()}


def _resolve(value: str) -> Path:
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    unknown = set(data) - set(_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}; valid keys: {sorted(_KEYS)}")
    return data


def get_paths() -> Paths:
    values = dict(_DEFAULTS)
    values.update(_read_yaml(REPO_ROOT / "configs" / "paths.yaml"))
    values.update(_read_yaml(REPO_ROOT / "configs" / "paths.local.yaml"))
    for key, env in _KEYS.items():
        if os.environ.get(env):
            values[key] = os.environ[env]
    paths = Paths(**{k: (_resolve(v) if v else None) for k, v in values.items()})
    if paths.archives is None:
        paths.archives = paths.train_data / "_archives"
    return paths


def list_sequences(dataset_dir: Path):
    """Lists the valid VOT sequences in the dataset folder (folders containing a `sequence` file).
    list.txt is not used, so nothing ever has to be written into the dataset folder."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        return []
    return sorted(p.name for p in dataset_dir.iterdir()
                  if p.is_dir() and (p / "sequence").is_file() and (p / "groundtruth.txt").is_file())
