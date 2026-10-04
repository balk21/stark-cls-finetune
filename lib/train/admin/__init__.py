from .stats import AverageMeter, StatValue
from .tensorboard import TensorboardWriter


def env_settings():
    """The original STARK read dataset roots from a machine-specific local.py. In this repository the roots are
    always passed explicitly (see stark_ft/training.py and configs/paths.yaml)."""
    raise RuntimeError("Dataset root not given: pass `root=` explicitly (see stark_ft/training.py).")
