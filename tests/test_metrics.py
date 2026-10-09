"""mAP / AP50 / AP75 of a sequence: frames without the target are not evaluated, predictions below the score
threshold are not counted, boxes are rounded to integers."""
from stark_ft.test.evaluation import coco_ap

BOX = [10.0, 10.0, 40.0, 30.0]


def test_coco_ap_rules():
    good = [(i, BOX, BOX, 0.9) for i in range(1, 11)]
    assert coco_ap(good, 0.35) == {"mAP": 1.0, "AP50": 1.0, "AP75": 1.0}
    # a prediction where the target is absent is not evaluated
    assert coco_ap(good + [(11, None, BOX, 0.99)], 0.35)["mAP"] == 1.0
    # predictions below the score threshold are not counted: half of the frames are missed
    half = [(i, BOX, BOX, 0.9 if i <= 5 else 0.1) for i in range(1, 11)]
    assert coco_ap(half, 0.35)["AP50"] < 0.6 and coco_ap(half, 0.05)["AP50"] == 1.0
    # boxes are rounded to integers: a 0.4-pixel shift is no error
    shifted = [(i, BOX, [v + 0.4 for v in BOX], 0.9) for i in range(1, 11)]
    assert coco_ap(shifted, 0.35)["mAP"] == 1.0
    # a missing score counts as 1 (STARK-S has no score)
    assert coco_ap([(i, BOX, BOX, float("nan")) for i in range(1, 11)], 0.35)["mAP"] == 1.0


if __name__ == "__main__":  # without pytest: python -m tests.test_metrics
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
