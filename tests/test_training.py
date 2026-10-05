"""Training configuration (no GPU needed): validation, run names, dataset order, overrides of the STARK config,
initial weights and the dataset checks."""
import json
import tempfile
from pathlib import Path

from stark_ft.paths import Paths
from stark_ft.training import TrainConfig, _load_model_cfg, describe, resolve_init


def _paths(tmp):
    tmp = Path(tmp)
    return Paths(checkpoints=tmp / "ckpt", dataset=tmp / "seq", outputs=tmp / "out", train_data=tmp / "train_data",
                 train_outputs=tmp / "train_out")


def _expect_error(**kwargs):
    try:
        TrainConfig(**kwargs).validate()
    except ValueError:
        return
    raise AssertionError(f"no error for {kwargs}")


def test_defaults_and_run_name():
    tc = TrainConfig().validate()
    assert (tc.stage, tc.model_config, tc.effective_batch, tc.micro_batch, tc.seed) == (1, "baseline_R101", 128, 16, 42)
    assert tc.total_epochs == 500 and TrainConfig(stage=2, init="official").total_epochs == 50
    assert tc.run_name == "st101_stage1_got10k_s42"
    assert TrainConfig(model_config="baseline", stage=1, datasets=["coco"], epochs=3).run_name == "st50_stage1_coco_e3_s42"
    assert (TrainConfig(stage=2, datasets=["coco"], init="st101_stage1_coco_s42").run_name
            == "st101_stage2_coco_from-st101_stage1_coco_s42_s42")
    # the official weights (trained on all four datasets) are always visible in the run name
    assert TrainConfig(stage=2, datasets=["coco"], init="official").run_name == "st101_stage2_coco_from-official_s42"
    assert TrainConfig(name="my_run").run_name == "my_run"


def test_dataset_order_is_normalised():
    a = TrainConfig(datasets=["coco", "got10k"], dataset_ratios=[3, 1])
    b = TrainConfig(datasets=["got10k", "coco"], dataset_ratios=[1, 3])
    assert a.datasets == b.datasets == ["got10k", "coco"]
    assert a.dataset_ratios == b.dataset_ratios == [1, 3]
    assert a.run_name == b.run_name
    full = TrainConfig(datasets=["trackingnet", "coco", "got10k", "lasot"])
    assert full.datasets == ["lasot", "got10k", "coco", "trackingnet"]  # the order of the STARK configs


def test_invalid_configs():
    _expect_error(stage=3)
    _expect_error(model_config="nope")
    _expect_error(datasets=[])
    _expect_error(datasets=["imagenet"])
    _expect_error(datasets=["coco", "coco"])
    _expect_error(datasets=["got10k", "got10k_full"])
    _expect_error(datasets=["coco"], dataset_ratios=[1, 2])
    _expect_error(val_datasets=["coco"])
    _expect_error(effective_batch=100, micro_batch=16)
    _expect_error(stage=1, init="official")
    _expect_error(stage=2)  # stage 2 never falls back to the official weights silently
    _expect_error(stage=2, init="")
    _expect_error(name="a/b")


def test_from_dict_and_file():
    tc = TrainConfig.from_dict({"stage": "1", "epochs": "3", "datasets": ["coco"], "dataset_ratios": ["2"],
                                "lr_drop_epoch": "none"})
    assert tc.stage == 1 and tc.epochs == 3 and tc.dataset_ratios == [2.0] and tc.lr_drop_epoch is None
    try:
        TrainConfig.from_dict({"lr": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown parameter accepted")
    with tempfile.TemporaryDirectory() as tmp:  # train_config.json written by a run: {"config": {...}, ...}
        path = Path(tmp) / "train_config.json"
        path.write_text(json.dumps({"config": TrainConfig(seed=7).to_dict(), "code_hash": "x"}))
        assert TrainConfig.from_file(path).seed == 7


def test_model_cfg_overrides():
    tc = TrainConfig(stage=1, datasets=["coco", "got10k"], epochs=4, lr_drop_epoch=3, samples_per_epoch=256,
                     micro_batch=8, effective_batch=32, num_workers=2, val_datasets=[])
    cfg = _load_model_cfg(tc)
    assert cfg.DATA.TRAIN.DATASETS_NAME == ["GOT10K_vottrain", "COCO17"]
    assert cfg.DATA.TRAIN.DATASETS_RATIO == [1, 1] and cfg.DATA.VAL.DATASETS_NAME == []
    assert (cfg.TRAIN.EPOCH, cfg.TRAIN.LR_DROP_EPOCH, cfg.DATA.TRAIN.SAMPLE_PER_EPOCH) == (4, 3, 256)
    assert cfg.TRAIN.BATCH_SIZE == 8 and cfg.TRAIN.NUM_WORKER == 2
    assert cfg.MODEL.BACKBONE.PRETRAINED is True and not getattr(cfg.TRAIN, "TRAIN_CLS", False)
    cfg2 = _load_model_cfg(TrainConfig(stage=2, init="official"))
    assert cfg2.TRAIN.TRAIN_CLS and cfg2.MODEL.BACKBONE.PRETRAINED is False
    assert (cfg2.TRAIN.EPOCH, cfg2.TRAIN.LR, cfg2.DATA.TRAIN.SAMPLE_PER_EPOCH) == (50, 1e-4, 60000)
    assert cfg2.DATA.VAL.DATASETS_NAME == ["GOT10K_votval"]


def test_init_and_dataset_checks():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        info = describe(TrainConfig(stage=2, init="official", datasets=["got10k", "coco"]), paths)
        assert len(info["dataset_problems"]) == 2 and "Official checkpoint not found" in info["init_error"]
        assert info["steps_per_epoch"] == 468 and info["accumulation"] == 8
        # official stage-2 checkpoint: everything except the classification head is loaded
        official = paths.checkpoints / "stark_st2" / "baseline_R101" / "STARKST_ep0050.pth.tar"
        official.parent.mkdir(parents=True)
        official.write_bytes(b"x")
        assert resolve_init(TrainConfig(stage=2, init="official"), paths) == (official, ("cls_head.",))
        # a finished stage-1 run
        run = paths.train_outputs / "s1run" / "final.pth.tar"
        run.parent.mkdir(parents=True)
        run.write_bytes(b"x")
        assert resolve_init(TrainConfig(stage=2, init="s1run"), paths) == (run, ())
        assert resolve_init(TrainConfig(stage=1), paths) == (None, None)
        # dataset layouts
        (paths.train_data / "got10k" / "train").mkdir(parents=True)
        (paths.train_data / "got10k" / "train" / "list.txt").write_text("GOT-10k_Train_000001")
        (paths.train_data / "coco" / "annotations").mkdir(parents=True)
        (paths.train_data / "coco" / "annotations" / "instances_train2017.json").write_text("{}")
        (paths.train_data / "coco" / "images" / "train2017").mkdir(parents=True)
        info = describe(TrainConfig(stage=2, init="official", datasets=["got10k", "coco"]), paths)
        assert info["dataset_problems"] == [] and info["init_error"] is None
        info = describe(TrainConfig(datasets=["got10k_full"], model_config="baseline_R101_got10k_only"), paths)
        assert info["dataset_problems"] == [] and list(info["dataset_roots"]) == ["got10k"]


if __name__ == "__main__":  # without pytest: python -m tests.test_training
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
