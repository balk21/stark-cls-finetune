"""Choosing the weights of a test: official STARK weights vs. training runs of this repository (no GPU needed)."""
import json
import tempfile
from pathlib import Path

from stark_ft import weights
from stark_ft.paths import Paths
from stark_ft.test.config import ExperimentConfig
from stark_ft.train.config import TrainConfig


def _paths(tmp):
    tmp = Path(tmp)
    return Paths(checkpoints=tmp / "ckpt", dataset=tmp / "seq", outputs=tmp / "out", train_data=tmp / "train_data",
                 train_outputs=tmp / "train_out")


def _run(paths, stage, finished=True, origin=None, **kw):
    tc = TrainConfig(stage=stage, **kw)
    run_dir = paths.train_outputs / tc.run_name
    run_dir.mkdir(parents=True)
    meta = {"config": tc.to_dict(), "run_name": tc.run_name}
    if origin:
        meta["origin"] = origin
    (run_dir / "train_config.json").write_text(json.dumps(meta))
    if finished:
        (run_dir / "final.pth.tar").write_bytes(b"x")
    return tc.run_name


def test_paths_of_weights():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        official = paths.checkpoints / "stark_st2" / "baseline_R101" / "STARKST_ep0050.pth.tar"
        assert weights.path_of("official", "stark_st", "baseline_R101", paths) == official
        assert weights.path_of("official", "stark_s", "baseline", paths) == \
            paths.checkpoints / "stark_s" / "baseline" / "STARKS_ep0500.pth.tar"
        run = paths.train_outputs / "st101_coco_stage2_s42" / "final.pth.tar"
        assert weights.path_of("st101_coco_stage2_s42", "stark_st", "baseline_R101", paths) == run
        assert weights.path_of("train:st101_coco_stage2_s42", "stark_st", "baseline_R101", paths) == run
        assert weights.path_of("STARKSTcoco_ep0050.pth.tar", "stark_st", "baseline_R101", paths) == \
            official.parent / "STARKSTcoco_ep0050.pth.tar"
        assert weights.path_of("/x/y.pth.tar", "stark_st", "baseline_R101", paths) == Path("/x/y.pth.tar")


def test_runs_are_checked():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        s1 = _run(paths, 1, datasets=["coco"])
        s2 = _run(paths, 2, datasets=["coco"])
        s2_50 = _run(paths, 2, datasets=["coco"], model_config="baseline")
        running = _run(paths, 2, finished=False, datasets=["got10k"])
        ok = weights.describe(s2, "stark_st", "baseline_R101", paths)
        assert ok["problem"] is None and "from ImageNet" in ok["origin"] and ok["datasets"] == ["coco"]
        for w, model_config, text in ((s1, "baseline_R101", "stage-1 run"), (s2_50, "baseline_R101", "model_config"),
                                      (running, "baseline_R101", "not finished"), ("nope", "baseline_R101", "No training run")):
            problem = weights.describe(w, "stark_st", model_config, paths)["problem"]
            assert problem and text in problem, (w, problem)
        assert weights.resolve(s2, "stark_st", "baseline_R101", paths) == paths.train_outputs / s2 / "final.pth.tar"
        try:
            weights.resolve(s1, "stark_st", "baseline_R101", paths)
            raise AssertionError("a stage-1 run was accepted for testing")
        except FileNotFoundError:
            pass


def test_official_is_downloaded_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        info = weights.describe("official", "stark_st", "baseline_R101_got10k_only", paths)
        assert info["problem"] is None and "GOT-10k only" in info["origin"] and "automatically" in info["note"]
        calls = []
        original = weights.download_checkpoints
        weights.download_checkpoints = lambda models: calls.append(list(models))
        try:
            weights.resolve("official", "stark_st", "baseline_R101", paths)
        finally:
            weights.download_checkpoints = original
        assert calls == [[("stark_st", "baseline_R101")]]


def test_weights_are_listed_by_origin():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        own = _run(paths, 2, datasets=["coco"])
        on_official = _run(paths, 2, datasets=["coco"], init="official", origin="official")
        listed = weights.list_weights(paths)
        assert [r["run_name"] for r in listed["runs"]["imagenet"]] == [own]
        assert [r["run_name"] for r in listed["runs"]["official"]] == [on_official]
        assert len(listed["official"]) == 6 and not any(o["downloaded"] for o in listed["official"])


def test_experiment_config_weights():
    assert ExperimentConfig().weights == "official"
    # earlier parameter name `checkpoint` (parameter files and experiment.json of earlier experiments)
    assert ExperimentConfig.from_dict({"checkpoint": None}).weights == "official"
    assert ExperimentConfig.from_dict({"checkpoint": "train:st101_coco_stage2_s42"}).weights == "st101_coco_stage2_s42"
    assert ExperimentConfig.from_dict({"checkpoint": "STARKSTcoco_ep0050.pth.tar"}).weights == "STARKSTcoco_ep0050.pth.tar"
    # experiment names say which weights were used
    base = ExperimentConfig(ft_mode="none", sequences=["bull"])
    assert base.experiment_name == "st101_base_int100_bull"
    assert ExperimentConfig(ft_mode="none", sequences=["bull"], weights="st101_coco_stage2_s42").experiment_name == \
        "st101_coco_stage2_s42_base_int100_bull"
    assert ExperimentConfig(ft_mode="none", weights="STARKSTcoco_ep0050.pth.tar").experiment_name == \
        "st101-STARKSTcoco_ep0050_base_int100"


if __name__ == "__main__":  # without pytest: python -m tests.test_weights
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
