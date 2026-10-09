"""
Experiment configuration: ALL parameters that define a test run live here.

The meaning of every parameter is explained in docs/test.md.
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional, Union

import yaml

from stark_ft import weights as weights_mod
from stark_ft.test import datasets
from stark_ft.common import coerce
from stark_ft.paths import MODEL_CONFIG_DIR, Paths

# model name -> (checkpoint / model_config folder, checkpoint file prefix)
MODELS = {
    "stark_st": {"dir": "stark_st2", "prefix": "STARKST"},
    "stark_s": {"dir": "stark_s", "prefix": "STARKS"},
}
FT_MODES = ("none", "init", "online")
FT_SAMPLES = ("pos", "posneg")
UPDATE_MODES = ("stark", "max")
NO_UPDATE_INTERVAL = 99999  # conventional value meaning "no template update"
# Parameters that only affect the analysis: changing them does not invalidate tracking results
EVAL_ONLY_FIELDS = ("eval_score_thr", "eval_iou_thr", "eval_thr_resolution")
RUN_ONLY_FIELDS = ("tracker_timeout",)  # how the run is done, not its result
UPDATE_FIELDS = ("update_mode", "update_interval", "update_conf_thr", "update_iou_thr", "max_template_updates")
# Not at their default value -> in the experiment name as <tag><value>
NAME_TAGS = (("update_conf_thr", "uthr"), ("update_iou_thr", "uiou"), ("max_template_updates", "maxupd"),
             ("max_ft_updates", "maxft"), ("ft_weight_decay", "wd"), ("ft_grad_clip_norm", "clip"),
             ("ft_center_jitter", "cj"), ("ft_scale_jitter", "sj"), ("seed", "s"))


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
    weights: str = "official"             # "official" (STARK's) | "<training run name>" | checkpoint file / path

    # ---- Data ----
    dataset: str = "votlt2020"            # "votlt2020" | "got10k_val" | "got10k_test" | "got10k_train"
    sequences: Union[str, List[str]] = "all"  # "all" or ["bull", "ballet", ...] / ["GOT-10k_Val_000001", ...]

    # ---- Template update (STARK-ST) ----
    update_mode: str = "stark"            # "stark": every N-th frame | "max": the best frame of every N frames
    update_interval: int = 100            # N; 99999 = no updates
    update_conf_thr: float = 0.5          # The confidence must be GREATER than this for an update
    update_iou_thr: float = 0.5           # "max" only: min IoU with the box of the previous candidate frame
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
    eval_score_thr: float = 0.35          # Score threshold: mAP / AP50 / AP75 and P/R/F1 count predictions above it
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
        if self.update_mode not in UPDATE_MODES:
            errors.append(f"update_mode={self.update_mode!r}; valid: {UPDATE_MODES}")
        elif self.model == "stark_s" and self.update_mode != "stark":
            errors.append("stark_s has no template update; update_mode must be 'stark'")
        if not 0 <= self.update_iou_thr <= 1:
            errors.append("update_iou_thr must be in [0, 1]")
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
        if self.dataset not in datasets.DATASETS:
            errors.append(f"dataset={self.dataset!r}; valid: {list(datasets.DATASETS)}")
        elif self.run_redetection and datasets.is_got10k(self.dataset):
            errors.append("run_redetection is a VOT-LT2020 experiment; it is not available on GOT-10k")
        if not isinstance(self.weights, str) or not self.weights:
            errors.append("weights must be 'official', a training run name or a checkpoint file / path")
        if self.name is not None and (not self.name or any(c in self.name for c in '/\\:*?"<>| ')):
            errors.append(f"name={self.name!r} cannot be used as a folder name (no spaces or /\\:*?\"<>|)")
        if errors:
            raise ValueError("Invalid experiment configuration:\n  - " + "\n  - ".join(errors))
        return self

    # ------------------------------------------------------------------ derived values
    def model_yaml(self) -> Path:
        return MODEL_CONFIG_DIR / self.model_dir / f"{self.model_config}.yaml"

    def checkpoint_path(self, paths: Paths) -> Path:
        """The checkpoint file of `weights` (stark_ft/weights.py; nothing is downloaded here)."""
        return weights_mod.path_of(self.weights, self.model, self.model_config, paths)

    def short_model_name(self) -> str:
        if self.model == "stark_s":
            base = "s50"
        else:
            base = "st101" if "R101" in self.model_config else "st50"
        if "got10k_only" in self.model_config:
            base += "got"
        run = weights_mod.run_name(self.weights)
        if run:  # a training run of this repository: its name already says model, datasets and stage
            return run
        if self.weights != weights_mod.OFFICIAL:
            base += "-" + Path(self.weights).name.split(".")[0]
        return base

    def unused_fields(self) -> list:
        """Parameters that have no effect with the other settings (e.g. ft_epochs_online with ft_mode="init"):
        they are left out of the experiment name and of the resume check."""
        ft = [f.name for f in fields(self) if f.name.startswith("ft_") and f.name != "ft_mode"]
        ft += ["max_ft_updates", "seed"]
        if self.model == "stark_s":  # no template update, no classification head
            return list(UPDATE_FIELDS) + ft
        no_updates = self.update_interval >= NO_UPDATE_INTERVAL
        unused = []
        if no_updates:
            unused += [f for f in UPDATE_FIELDS if f != "update_interval"]
        elif self.update_mode != "max":
            unused.append("update_iou_thr")
        if self.ft_mode == "none":
            return unused + ft
        if self.ft_mode == "init":  # no fine-tuning at the template updates
            unused += ["ft_epochs_online", "max_ft_updates"]
        if not self.ft_pos_jitter:
            unused += ["ft_center_jitter", "ft_scale_jitter"]
        return unused

    def auto_name(self) -> str:
        """Model, mode and every parameter that changes the result and is not at its default value."""
        unused = set(self.unused_fields())
        defaults = {f.name: f.default for f in fields(self)}
        changed = [k for k in defaults if k not in unused and getattr(self, k) != defaults[k]]
        parts = [self.short_model_name()]
        if self.model == "stark_st":
            interval = "noupd" if self.update_interval >= NO_UPDATE_INTERVAL else \
                f"{'max' if self.update_mode == 'max' else 'int'}{self.update_interval}"
            if self.ft_mode == "none":
                parts += ["base", interval]
            else:
                # fine-tuning steps: ep<first frame> (init) | ep<first frame>+<every template update> (online)
                epochs = f"ep{self.ft_epochs_init}" + (f"+{self.ft_epochs_online}" if self.ft_mode == "online" else "")
                parts += [self.ft_mode, self.ft_samples, f"lr{self.ft_lr:g}", epochs, interval]
            parts += [f"{tag}{getattr(self, key):g}" for key, tag in NAME_TAGS if key in changed]
            if "ft_pos_jitter" in changed:
                parts.append("nojit")
        else:
            parts.append("base")
        if self.run_redetection:
            parts.append("redet")
        if self.dataset != datasets.VOT:
            parts.append(self.dataset.replace("_", "-"))  # e.g. got10k-val
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
        skip = {"name", *EVAL_ONLY_FIELDS, *RUN_ONLY_FIELDS, *self.unused_fields()}
        return {k: v for k, v in asdict(self).items() if k not in skip}

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        data = dict(data)
        if "checkpoint" in data:  # earlier name of `weights`: None = official, "train:<run>" = a training run
            old = data.pop("checkpoint")
            data.setdefault("weights", weights_mod.OFFICIAL if not old else str(old).replace("train:", "", 1))
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
