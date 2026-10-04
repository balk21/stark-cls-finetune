"""ExperimentConfig: reading from files (the path used by the notebooks) and parameter type coercion."""
import json
import tempfile
from pathlib import Path

from stark_ft.config import ExperimentConfig


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


if __name__ == "__main__":  # without pytest: python -m tests.test_config
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
