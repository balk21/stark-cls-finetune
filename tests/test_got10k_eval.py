"""GOT-10k tests (no GPU needed): sequence names, AO / SR as the GOT-10k toolkit, the submission file."""
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from stark_ft.paths import Paths
from stark_ft.test import datasets, direct
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.evaluation import got10k_ious, got10k_metrics


def _paths(tmp):
    tmp = Path(tmp)
    return Paths(checkpoints=tmp / "ckpt", dataset=tmp / "seq", outputs=tmp / "out", train_data=tmp / "train_data",
                 train_outputs=tmp / "train_out", archives=tmp / "archives")


def test_sequences_and_config():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _paths(tmp)
        val = datasets.resolve("got10k_val", "all", paths)
        assert len(val) == 180 and val[0] == "GOT-10k_Val_000001" and val[-1] == "GOT-10k_Val_000180"
        assert len(datasets.resolve("got10k_train", "all", paths)) == 9335
        assert datasets.resolve("got10k_test", ["GOT-10k_Test_000005"], paths) == ["GOT-10k_Test_000005"]
        for dataset, seqs in (("got10k_val", ["GOT-10k_Test_000001"]), ("got10k_test", ["GOT-10k_Test_000181"])):
            try:
                datasets.resolve(dataset, seqs, paths)
                raise AssertionError(f"{seqs} accepted for {dataset}")
            except ValueError:
                pass
        assert datasets.root("got10k_val", paths) == paths.train_data / "got10k" / "val"
        assert "extracted from the GOT-10k archives" in datasets.to_fetch("got10k_val", val, paths)
    cfg = ExperimentConfig(dataset="got10k_val", model_config="baseline_R101_got10k_only", ft_mode="none",
                           update_interval=200).validate()
    assert cfg.experiment_name == "st101got_base_int200_got10k-val"
    for bad in (dict(dataset="otb"), dict(dataset="got10k_val", run_redetection=True)):
        try:
            ExperimentConfig(**bad).validate()
            raise AssertionError(f"{bad} accepted")
        except ValueError:
            pass
    assert ExperimentConfig.from_dict({"sequences": ["bull"]}).dataset == "votlt2020"  # earlier experiments


def _toolkit_rect_iou(r1, r2, bound):
    """got10k.utils.metrics.rect_iou (GOT-10k toolkit), copied for the comparison."""
    r1, r2 = np.array(r1, float), np.array(r2, float)
    for r in (r1, r2):
        r[:, 0] = np.clip(r[:, 0], 0, bound[0])
        r[:, 1] = np.clip(r[:, 1], 0, bound[1])
        r[:, 2] = np.clip(r[:, 2], 0, bound[0] - r[:, 0])
        r[:, 3] = np.clip(r[:, 3], 0, bound[1] - r[:, 1])
    x1, y1 = np.maximum(r1[:, 0], r2[:, 0]), np.maximum(r1[:, 1], r2[:, 1])
    x2 = np.minimum(r1[:, 0] + r1[:, 2], r2[:, 0] + r2[:, 2])
    y2 = np.minimum(r1[:, 1] + r1[:, 3], r2[:, 1] + r2[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = r1[:, 2] * r1[:, 3] + r2[:, 2] * r2[:, 3] - inter
    return np.clip(inter / (union + np.finfo(float).eps), 0.0, 1.0)


def test_ao_sr_as_the_toolkit():
    rng = np.random.RandomState(0)
    size = (320, 240)
    gt = [list(rng.uniform([0, 0, 10, 10], [300, 220, 80, 80])) for _ in range(50)]
    pred = [list(np.array(g) + rng.normal(0, 8, 4)) for g in gt]
    pred[7] = [-30, -20, 100, 90]  # partly outside the image: clipped
    cover = np.ones(50, int)
    cover[[0, 3, 20]] = 0
    visible = [g if c > 0 else None for g, c in zip(gt, cover)]
    ours = got10k_ious(visible, pred, size)
    ref = _toolkit_rect_iou(pred[1:], gt[1:], size)[cover[1:] > 0]
    assert np.allclose(ours, ref) and len(ours) == 47
    m = got10k_metrics(ours)
    assert np.isclose(m["AO"], ref.mean()) and np.isclose(m["SR50"], (ref > 0.5).mean())
    assert np.isclose(m["SR75"], (ref > 0.75).mean())
    assert got10k_metrics([0.5, 0.75, 1.0]) == {"AO": 0.75, "SR50": 2 / 3, "SR75": 1 / 3}  # strictly greater


def test_submission_file():
    import cv2
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        seq = "GOT-10k_Test_000001"
        (tmp / "test" / seq).mkdir(parents=True)
        for i in range(1, 4):
            cv2.imwrite(str(tmp / "test" / seq / f"{i:08d}.jpg"), np.zeros((8, 8, 3), np.uint8))
        (tmp / "test" / seq / "groundtruth.txt").write_text("1,2,3,4\n")
        pred = tmp / "out" / "predictions" / seq
        pred.mkdir(parents=True)
        (pred / f"{seq}_001.txt").write_text("1\n2.0000,3.0000,4.0000,5.0000\n3.0000,4.0000,5.0000,6.0000\n")
        (pred / f"{seq}_001_time.value").write_text("0.5\n0.02\n0.03\n")
        path = direct.write_submission(tmp / "out", tmp / "test", [seq])
        with zipfile.ZipFile(path) as z:
            assert sorted(z.namelist()) == [f"{seq}/{seq}_001.txt", f"{seq}/{seq}_time.txt"]
            boxes = z.read(f"{seq}/{seq}_001.txt").decode().split()
            assert boxes == ["1.0000,2.0000,3.0000,4.0000", "2.0000,3.0000,4.0000,5.0000", "3.0000,4.0000,5.0000,6.0000"]
            assert z.read(f"{seq}/{seq}_time.txt").decode().split() == ["0.5", "0.02", "0.03"]


if __name__ == "__main__":  # without pytest: python -m tests.test_got10k_eval
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
