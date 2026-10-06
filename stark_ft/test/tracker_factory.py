"""
Builds a tracker object from an ExperimentConfig (independent of VOT; shared by the smoke test and the VOT entry).
"""
import importlib
from pathlib import Path
from types import SimpleNamespace

from stark_ft.test.config import ExperimentConfig


def load_model_cfg(cfg: ExperimentConfig):
    """Applies the model YAML on top of STARK's default config.
    The config module holds a global object, so it is reloaded on every call (clean defaults)."""
    module_name = "lib.config.stark_st2.config" if cfg.model == "stark_st" else "lib.config.stark_s.config"
    module = importlib.reload(importlib.import_module(module_name))
    module.update_config_from_file(str(cfg.model_yaml()))
    model_cfg = module.cfg
    # The full tracker checkpoint is loaded afterwards, so the ImageNet weights do not need to be downloaded
    model_cfg.MODEL.BACKBONE.PRETRAINED = False
    return model_cfg


def build_tracker(cfg: ExperimentConfig, checkpoint: Path, log_dir=None):
    cfg.validate()
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_cfg = load_model_cfg(cfg)
    params = SimpleNamespace(
        cfg=model_cfg,
        checkpoint=str(checkpoint),
        template_factor=model_cfg.TEST.TEMPLATE_FACTOR,
        template_size=model_cfg.TEST.TEMPLATE_SIZE,
        search_factor=model_cfg.TEST.SEARCH_FACTOR,
        search_size=model_cfg.TEST.SEARCH_SIZE,
        save_all_boxes=False,
        debug=False,
    )

    if cfg.model == "stark_s":
        from lib.test.tracker.stark_s import STARK_S
        return STARK_S(params)

    params.update_intervals = [cfg.update_interval]
    params.update_conf_thr = cfg.update_conf_thr
    params.max_template_updates = cfg.max_template_updates

    if cfg.ft_mode == "none":
        from lib.test.tracker.stark_st import STARK_ST
        return STARK_ST(params)

    from lib.test.tracker.stark_st_ft import STARK_ST_FT
    params.ft_mode = cfg.ft_mode
    params.ft_samples = cfg.ft_samples
    params.ft_lr = cfg.ft_lr
    params.ft_weight_decay = cfg.ft_weight_decay
    params.ft_grad_clip_norm = cfg.ft_grad_clip_norm
    params.ft_epochs_init = cfg.ft_epochs_init
    params.ft_epochs_online = cfg.ft_epochs_online
    params.ft_pos_jitter = cfg.ft_pos_jitter
    params.ft_center_jitter = cfg.ft_center_jitter
    params.ft_scale_jitter = cfg.ft_scale_jitter
    params.max_ft_updates = cfg.max_ft_updates
    params.seed = cfg.seed
    params.log_dir = None if log_dir is None else str(log_dir)
    return STARK_ST_FT(params)
