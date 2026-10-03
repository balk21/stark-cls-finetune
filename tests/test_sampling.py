"""negative_search_box: the generated search crop must NOT overlap the target box at all."""
import math

import numpy as np

from lib.test.tracker.ft_sampling import jitter_box, negative_search_box


def _crop_window(box, search_factor):
    # same computation as lib/utils/processing_utils.sample_target
    x, y, w, h = box
    crop_sz = math.ceil(math.sqrt(w * h) * search_factor)
    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    return x1, y1, x1 + crop_sz, y1 + crop_sz


def _overlap(window, box):
    x1, y1, x2, y2 = window
    bx, by, bw, bh = box
    return max(0, min(x2, bx + bw) - max(x1, bx)) * max(0, min(y2, by + bh) - max(y1, by))


def test_negative_never_contains_target():
    rng = np.random.RandomState(0)
    for _ in range(20000):
        img_w, img_h = rng.randint(200, 2000), rng.randint(200, 1500)
        w = rng.uniform(4, img_w * 0.9)
        h = rng.uniform(4, img_h * 0.9)
        x = rng.uniform(-w * 0.2, img_w - w * 0.8)
        y = rng.uniform(-h * 0.2, img_h - h * 0.8)
        neg, cov = negative_search_box([x, y, w, h], img_w, img_h, 5.0)
        assert _overlap(_crop_window(neg, 5.0), [x, y, w, h]) == 0
        assert 0.0 <= cov <= 1.0
        assert abs(neg[2] - w) < 1e-9 and abs(neg[3] - h) < 1e-9


def test_old_rule_failed_for_square_targets():
    # The old rule (shift by 2*max(w,h) from the centre) left a square target inside the crop
    x, y, w, h = 500, 500, 50, 50
    old_neg = [x + 2 * max(w, h), y, w, h]
    assert _overlap(_crop_window(old_neg, 5.0), [x, y, w, h]) > 0


def test_jitter_deterministic_with_seed():
    a = jitter_box([10, 20, 30, 40], np.random.RandomState(3))
    b = jitter_box([10, 20, 30, 40], np.random.RandomState(3))
    assert a == b


if __name__ == "__main__":  # without pytest: python -m tests.test_sampling
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
