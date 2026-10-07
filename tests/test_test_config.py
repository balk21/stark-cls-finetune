"""ExperimentConfig: reading from files (the path used by the notebooks), parameter type coercion, experiment
names and the resume check."""
import json
import tempfile
from dataclasses import fields, replace
from pathlib import Path

from stark_ft.test.config import EVAL_ONLY_FIELDS, RUN_ONLY_FIELDS, ExperimentConfig


def test_from_file_json_and_yaml():
    params = {"sequences": ["bull"], "ft_lr": 1e-5, "ft_epochs_online": 15, "seed": 3}
    with tempfile.TemporaryDirectory() as tmp:
        js = Path(tmp) / "params.json"
        js.write_text(json.dumps(params))
        ym = Path(tmp) / "params.yaml"
        ym.write_text("sequences: [bull]\nft_lr: 1e-5\nft_epochs_online: 15\nseed: 3\n")  # 1e-5 is a str in YAML 1.1
        for path in (js, ym):
            cfg = ExperimentConfig.from_file(path).validate()
            assert cfg.ft_lr == 1e-5 and isinstance(cfg.ft_lr, float)
            assert cfg.seed == 3 and cfg.sequences == ["bull"]


def test_from_file_experiment_json():
    # experiment.json written by the runner has the structure {"config": {...}, ...}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "experiment.json"
        path.write_text(json.dumps({"config": ExperimentConfig(seed=5).to_dict(), "checkpoint": "x"}))
        assert ExperimentConfig.from_file(path).seed == 5


def test_coercion_and_errors():
    cfg = ExperimentConfig.from_dict({"ft_lr": "1e-4", "seed": "2", "ft_pos_jitter": "false", "update_conf_thr": 1})
    assert cfg.ft_lr == 1e-4 and cfg.seed == 2 and cfg.ft_pos_jitter is False and cfg.update_conf_thr == 1.0
    for bad in ({"seed": "abc"}, {"unknown_param": 1}):
        try:
            ExperimentConfig.from_dict(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"no error for {bad}")


# A different valid value for every parameter (a new parameter has to be added here and to the experiment name)
OTHER_VALUE = {
    "model": "stark_s", "model_config": "baseline", "weights": "st101_coco_stage2", "dataset": "got10k_val",
    "sequences": ["ballet"], "update_mode": "stark", "update_interval": 200, "update_conf_thr": 0.8,
    "update_iou_thr": 0.7, "max_template_updates": 3, "ft_mode": "init", "ft_samples": "posneg", "ft_lr": 1e-5,
    "ft_epochs_init": 5, "ft_epochs_online": 5, "ft_weight_decay": 0.0, "ft_grad_clip_norm": 1.0,
    "ft_pos_jitter": False, "ft_center_jitter": 3.0, "ft_scale_jitter": 0.25, "max_ft_updates": 2, "seed": 1,
    "run_redetection": True,
}


def test_every_parameter_that_changes_the_result_is_in_the_name():
    base = ExperimentConfig(sequences=["bull"], update_mode="max")  # every parameter is used
    assert base.unused_fields() == []
    for f in fields(ExperimentConfig):
        if f.name in ("name", *EVAL_ONLY_FIELDS, *RUN_ONLY_FIELDS):
            continue
        other = replace(base, **{f.name: OTHER_VALUE[f.name]})
        assert other.experiment_name != base.experiment_name, f.name
        assert other.tracking_dict() != base.tracking_dict(), f.name
    assert base.experiment_name == "st101_online_pos_lr0.0001_ep15+1_max100_bull"  # default seed: not in the name
    # online: steps on the first frame + at every update, also without updates; init: first frame only
    assert replace(base, ft_epochs_online=15, update_interval=99999).experiment_name == \
        "st101_online_pos_lr0.0001_ep15+15_noupd_bull"
    assert replace(base, ft_mode="init").experiment_name == "st101_init_pos_lr0.0001_ep15_max100_bull"


def test_unused_parameters_do_not_block_resuming():
    init = ExperimentConfig(ft_mode="init", sequences=["bull"])
    none = ExperimentConfig(ft_mode="none", sequences=["bull"])
    for cfg, unused in ((init, dict(ft_epochs_online=5, max_ft_updates=2, update_iou_thr=0.7)),
                        (none, dict(seed=3, ft_lr=1e-5, ft_epochs_online=5)),
                        (replace(init, ft_pos_jitter=False), dict(ft_center_jitter=3.0)),
                        (init, dict(tracker_timeout=600, eval_score_thr=0.5))):
        other = replace(cfg, **unused)
        assert other.experiment_name == cfg.experiment_name and other.tracking_dict() == cfg.tracking_dict(), unused


if __name__ == "__main__":  # without pytest: python -m tests.test_test_config
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
