"""
Experiment configuration: ALL parameters that define a test run live here.

The meaning of every parameter is explained in docs/test.md.
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional, Union

import yaml

from stark_ft.common import coerce
from stark_ft.paths import MODEL_CONFIG_DIR, REPO_ROOT, Paths

# model name -> (checkpoint / model_config folder, checkpoint file prefix)
MODELS = {
    "stark_st": {"dir": "stark_st2", "prefix": "STARKST"},
    "stark_s": {"dir": "stark_s", "prefix": "STARKS"},
}
FT_MODES = ("none", "init", "online")
FT_SAMPLES = ("pos", "posneg")
NO_UPDATE_INTERVAL = 99999  # conventional value meaning "no template update"
# Parameters that only affect the analysis: changing them does not invalidate tracking results
EVAL_ONLY_FIELDS = ("eval_score_thr", "eval_iou_thr", "eval_thr_resolution")


def available_model_configs(model: str) -> List[str]:
    d = MODEL_CONFIG_DIR / MODELS[model]["dir"]
    return sorted(p.stem for p in d.glob("*.yaml"))


@dataclass
class ExperimentConfig:
    # ---- Identity ----
    name: Optional[str] = None            # Output folder name; None = generated from the parameters

    # ---- Model ----
    model: str = "stark_st"               # "stark_st" (STARK-ST, with confidence) | "stark_s" (STARK-S, no confidence)
    model_config: str = "baseline_R101"   # model_configs/<stark_st2|stark_s>/<name>.yaml
    checkpoint: Optional[str] = None      # None: <PREFIX>_ep<EPOCH>.pth.tar; a file name, a path or "train:<run name>"

    # ---- Data ----
    sequences: Union[str, List[str]] = "all"  # "all" or ["bull", "ballet", ...]

    # ---- Template update (STARK-ST) ----
    update_interval: int = 100            # An update is attempted every N frames; 99999 = no updates
    update_conf_thr: float = 0.5          # The confidence must be GREATER than this for an update
    max_template_updates: int = -1        # Max number of updates per sequence; -1 = unlimited

    # ---- Fine-tuning (STARK-ST only) ----
    ft_mode: str = "online"               # "none" | "init" | "online"
    ft_samples: str = "pos"               # "pos" | "posneg"
    ft_lr: float = 1e-4
    ft_epochs_init: int = 15              # Number of fine-tuning steps on the first frame
    ft_epochs_online: int = 1             # Number of steps in every online fine-tuning session
    ft_weight_decay: float = 1e-4
    ft_grad_clip_norm: float = 0.1
    ft_pos_jitter: bool = True            # Apply the ST2 training jitter to the positive box
    ft_center_jitter: float = 4.5
    ft_scale_jitter: float = 0.5
    max_ft_updates: int = -1              # Max number of online fine-tuning sessions per sequence; -1 = unlimited
    seed: int = 0

    # ---- Evaluation ----
    eval_score_thr: float = 0.35          # Fixed confidence threshold for the "found" decision (P/R/F1)
    eval_iou_thr: float = 0.5             # Minimum IoU for a correct detection (P/R/F1)
    eval_thr_resolution: int = 100        # Number of candidate thresholds in the F-max search (vot-toolkit: 100)

    # ---- VOT-toolkit ----
    run_redetection: bool = False         # Also run the "redetection" experiment of the VOT-LT2020 stack
    tracker_timeout: int = 300            # Timeout in seconds for a single tracker response

    # ------------------------------------------------------------------
    def __post_init__(self):
        if isinstance(self.sequences, (list, tuple)):
            self.sequences = list(dict.fromkeys(self.sequences))  # drop duplicates, keep order

    @property
    def model_dir(self) -> str:
        return MODELS[self.model]["dir"]

    def validate(self):
        errors = []
        if self.model not in MODELS:
            errors.append(f"model={self.model!r}; valid: {list(MODELS)}")
        elif self.model_config not in available_model_configs(self.model):
            errors.append(f"model_config={self.model_config!r}; valid: {available_model_configs(self.model)}")
        if self.ft_mode not in FT_MODES:
            errors.append(f"ft_mode={self.ft_mode!r}; valid: {FT_MODES}")
        if self.ft_samples not in FT_SAMPLES:
            errors.append(f"ft_samples={self.ft_samples!r}; valid: {FT_SAMPLES}")
        if self.model == "stark_s" and self.ft_mode != "none":
            errors.append("stark_s has no classification head; ft_mode must be 'none'")
        if self.update_interval < 1:
            errors.append("update_interval must be >= 1 (use 99999 for no updates)")
        if self.ft_epochs_init < 0 or self.ft_epochs_online < 0:
            errors.append("ft_epochs_* cannot be negative")
        if self.ft_lr <= 0:
            errors.append("ft_lr must be > 0")
        if not (0 <= self.eval_score_thr <= 1) or not (0 < self.eval_iou_thr <= 1):
            errors.append("eval_score_thr must be in [0, 1] and eval_iou_thr in (0, 1]")
        if self.eval_thr_resolution < 3:
            errors.append("eval_thr_resolution must be >= 3")
        if not (self.sequences == "all" or (isinstance(self.sequences, list) and self.sequences)):
            errors.append("sequences must be 'all' or a non-empty list")
        if self.name is not None and (not self.name or any(c in self.name for c in '/\\:*?"<>| ')):
            errors.append(f"name={self.name!r} cannot be used as a folder name (no spaces or /\\:*?\"<>|)")
        if errors:
            raise ValueError("Invalid experiment configuration:\n  - " + "\n  - ".join(errors))
        return self

    # ------------------------------------------------------------------ derived values
    def model_yaml(self) -> Path:
        return MODEL_CONFIG_DIR / self.model_dir / f"{self.model_config}.yaml"

    def default_checkpoint_name(self) -> str:
        with open(self.model_yaml()) as f:
            epoch = int(yaml.safe_load(f)["TEST"]["EPOCH"])
        return f"{MODELS[self.model]['prefix']}_ep{epoch:04d}.pth.tar"

    def checkpoint_path(self, paths: Paths) -> Path:
        ckpt = self.checkpoint or self.default_checkpoint_name()
        if ckpt.startswith("train:"):
            # Weights of a training run of this repository (stark_ft/train)
            return paths.train_outputs / ckpt[len("train:"):] / "final.pth.tar"
        if "/" in ckpt or "\\" in ckpt:
            # A path was given: absolute as is, relative to the repository root otherwise
            p = Path(ckpt).expanduser()
            return p if p.is_absolute() else (REPO_ROOT / p).resolve()
        # Only a file name was given: look it up in the standard checkpoint folder
        return paths.checkpoints / self.model_dir / self.model_config / ckpt

    def short_model_name(self) -> str:
        if self.model == "stark_s":
            base = "s50"
        else:
            base = "st101" if "R101" in self.model_config else "st50"
        if "got10k_only" in self.model_config:
            base += "got"
        if self.checkpoint:
            base += "-" + Path(self.checkpoint.replace("train:", "")).name.split(".")[0]
        return base

    def auto_name(self) -> str:
        parts = [self.short_model_name()]
        if self.model == "stark_st":
            interval = "noupd" if self.update_interval >= NO_UPDATE_INTERVAL else f"int{self.update_interval}"
            if self.ft_mode == "none":
                parts += ["base", interval]
            else:
                parts += [self.ft_mode, self.ft_samples, f"lr{self.ft_lr:g}", f"i{self.ft_epochs_init}"]
                if self.ft_mode == "online":
                    parts.append(f"o{self.ft_epochs_online}")
                parts += [interval, f"s{self.seed}"]
            if self.update_conf_thr != 0.5:
                parts.append(f"uthr{self.update_conf_thr:g}")
            if self.max_template_updates >= 0:
                parts.append(f"maxupd{self.max_template_updates}")
            if self.max_ft_updates >= 0 and self.ft_mode == "online":
                parts.append(f"maxft{self.max_ft_updates}")
        else:
            parts.append("base")
        if self.sequences != "all":
            parts.append(self.sequences[0] if len(self.sequences) == 1 else f"{len(self.sequences)}seq")
        return "_".join(parts)

    @property
    def experiment_name(self) -> str:
        return self.name or self.auto_name()

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    def tracking_dict(self) -> dict:
        """Parameters that affect the tracking result (used for the resume / overwrite decision)."""
        return {k: v for k, v in asdict(self).items() if k not in EVAL_ONLY_FIELDS}

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        types = {f.name: f.type for f in fields(cls)}
        unknown = set(data) - set(types)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        return cls(**{k: coerce(k, v, types[k]) for k, v in data.items()})

    @classmethod
    def from_file(cls, path) -> "ExperimentConfig":
        path = Path(path)
        with open(path) as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        # experiment.json has the structure {"config": {...}, ...}
        if "config" in data and isinstance(data["config"], dict):
            data = data["config"]
        return cls.from_dict(data)
