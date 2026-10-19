"""Drawing the boxes of an experiment: ground truth green, prediction red, 1 pixel, text top left (no GPU needed)."""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from stark_ft.paths import Paths
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.draw import draw_sequence

GT = [[20, 30, 40, 30], [22, 30, 40, 30], None]      # frame 2: target absent
PRED = [None, [60, 70, 30, 20], [100, 90, 20, 20]]  # frame 0: initialisation (no prediction)


def _experiment(tmp: Path):
    seq_dir = tmp / "seq" / "toy"
    (seq_dir / "color").mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(seq_dir / "color" / f"{i + 1:08d}.jpg"), np.full((160, 200, 3), 128, np.uint8))
    (seq_dir / "sequence").write_text("channels.color=color/%08d.jpg\n")
    (seq_dir / "groundtruth.txt").write_text("".join("nan,nan,nan,nan\n" if b is None else ",".join(map(str, b)) + "\n"
                                                     for b in GT))
    exp = tmp / "out" / "exp"
    pred = exp / "predictions" / "toy"
    pred.mkdir(parents=True)
    (pred / "toy_001.txt").write_text("1\n" + "".join(",".join(map(str, b)) + "\n" for b in PRED[1:]))
    (pred / "toy_001_confidence.value").write_text("\n0.9\n0.2\n")
    (pred / "toy_001_time.value").write_text("0.1\n0.02\n0.02\n")
    cfg = ExperimentConfig(ft_mode="none", sequences=["toy"])
    (exp / "experiment.json").write_text(json.dumps({"config": cfg.to_dict(), "sequences": ["toy"],
                                                     "dataset": str(tmp / "seq")}))
    return exp, Paths(checkpoints=tmp, dataset=tmp / "seq", outputs=tmp / "out", train_data=tmp, train_outputs=tmp)


def test_boxes_colours_and_text():
    with tempfile.TemporaryDirectory() as tmp:
        exp, paths = _experiment(Path(tmp))
        out = draw_sequence(exp, "toy", frames=[0, 1, 2], video=False, paths=paths)
        assert [p.name for p in out["images"]] == ["toy_000000.png", "toy_000001.png", "toy_000002.png"]
        assert out["video"] is None
        img = cv2.imread(str(out["images"][1]))  # BGR
        assert tuple(img[45, 22]) == (0, 255, 0)     # left edge of the ground-truth box: green
        assert tuple(img[80, 60]) == (0, 0, 255)     # left edge of the predicted box: red
        assert tuple(img[45, 23]) == (128, 128, 128)  # 1 pixel thick: next to the edge is the background again
        assert img[2:10, 2:60].mean() < 110          # dark text box in the top-left corner
        absent = cv2.imread(str(out["images"][2]))
        assert tuple(absent[100, 100]) == (0, 0, 255)
        assert not (absent == (0, 255, 0)).all(axis=2).any()  # target absent: no ground-truth box


def test_video():
    with tempfile.TemporaryDirectory() as tmp:
        exp, paths = _experiment(Path(tmp))
        out = draw_sequence(exp, "toy", frames=[], video=True, paths=paths)
        cap = cv2.VideoCapture(str(out["video"]))
        assert out["video"].is_file() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3


if __name__ == "__main__":  # without pytest: python -m tests.test_draw
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
