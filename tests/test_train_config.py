"""Training configuration (no GPU needed): validation, run names, dataset order, overrides of the STARK config,
initial weights and the dataset checks."""
import json
import tempfile
from pathlib import Path

from stark_ft.paths import Paths
from stark_ft.train.config import TrainConfig
from stark_ft.train.run import _load_model_cfg, describe, resolve_init


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
    assert tc.total_epochs == 500 and TrainConfig(stage=2).total_epochs == 50
    # runs are grouped by family (model + datasets): stage 1 and its stage 2
    assert tc.run_name == "st101_got10k_stage1_s42"
    assert TrainConfig(model_config="baseline", datasets=["coco"], epochs=3).run_name == "st50_coco_stage1_e3_s42"
    s2 = TrainConfig(stage=2, datasets=["coco"])
    assert s2.init == "st101_coco_stage1_s42" and s2.run_name == "st101_coco_stage2_s42"  # default: own stage 1
    assert TrainConfig(stage=2, datasets=["coco"], init="st101_coco_stage1_s42").run_name == s2.run_name
    quick = TrainConfig(stage=2, datasets=["coco"], epochs=1)  # the same parameters as stage 1 with epochs=1
    assert quick.init == TrainConfig(datasets=["coco"], epochs=1).run_name == "st101_coco_stage1_e1_s42"
    assert quick.run_name == "st101_coco_stage2_e1_s42"
    assert TrainConfig(stage=2, datasets=["coco"], init="st101_got10k+coco_stage1_s42").run_name == \
        "st101_coco_stage2_from-st101_got10k+coco_stage1_s42_s42"
    # runs on STARK's weights (trained on all four datasets) are marked
    assert TrainConfig(stage=2, datasets=["coco"], init="official").run_name == "st101_coco_stage2_on-official_s42"
    assert TrainConfig(stage=2, datasets=["got10k_full"], init="official",
                       model_config="baseline_R101_got10k_only").run_name == \
        "st101_got10k_full_stage2_on-official-got10k_s42"
    assert TrainConfig(datasets=["coco", "got10k"], dataset_ratios=[2, 1]).run_name == "st101_got10k+coco_r1-2_stage1_s42"
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


def _run(paths, name, stage, origin=None, finished=True, **kw):
    run_dir = paths.train_outputs / name
    run_dir.mkdir(parents=True)
    meta = {"config": TrainConfig(stage=stage, **kw).to_dict(), "run_name": name}
    if origin:
        meta["origin"] = origin
    (run_dir / "train_config.json").write_text(json.dumps(meta))
    if finished:
        (run_dir / "final.pth.tar").write_bytes(b"x")
    return run_dir


def test_init_and_dataset_checks():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        info = describe(TrainConfig(stage=2, init="official", datasets=["got10k", "coco"]), paths)
        assert len(info["dataset_problems"]) == 2 and info["init_error"] is None  # official: downloaded at the start
        assert "downloaded when the run starts" in info["init"] and info["origin"] == "official"
        assert info["steps_per_epoch"] == 468 and info["accumulation"] == 8
        # official stage-2 checkpoint: everything except the classification head is loaded
        official = paths.checkpoints / "stark_st2" / "baseline_R101" / "STARKST_ep0050.pth.tar"
        official.parent.mkdir(parents=True)
        official.write_bytes(b"x")
        assert resolve_init(TrainConfig(stage=2, init="official"), paths) == (official, ("cls_head.",), "official")
        assert resolve_init(TrainConfig(stage=1), paths) == (None, None, "imagenet")
        # stage 2 by default starts from our own stage-1 run of the same family; missing -> clear error
        info = describe(TrainConfig(stage=2, datasets=["coco"]), paths)
        assert "st101_coco_stage1_s42" in info["init_error"] and "Train it first" in info["init_error"]
        run = _run(paths, "st101_coco_stage1_s42", 1, datasets=["coco"])
        assert resolve_init(TrainConfig(stage=2, datasets=["coco"]), paths) == (run / "final.pth.tar", (), "imagenet")
        # the origin is inherited along the chain; an unfinished or stage-2 run cannot be the init
        _run(paths, "s1_on_official", 1, origin="official", datasets=["coco"])
        assert resolve_init(TrainConfig(stage=2, init="s1_on_official"), paths)[2] == "official"
        _run(paths, "s1_running", 1, finished=False)
        _run(paths, "s2_done", 2, datasets=["coco"])
        for init, error in (("s1_running", FileNotFoundError), ("s2_done", ValueError)):
            try:
                resolve_init(TrainConfig(stage=2, init=init), paths)
                raise AssertionError(f"init={init} accepted")
            except error:
                pass
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


if __name__ == "__main__":  # without pytest: python -m tests.test_train_config
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
