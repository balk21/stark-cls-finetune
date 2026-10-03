"""
Helpers that generate training samples (search regions) for fine-tuning.

All boxes are in [x, y, w, h] format (top-left corner + size).
"""
import math

import numpy as np


def jitter_box(bbox, rng: np.random.RandomState, center_jitter=4.5, scale_jitter=0.5):
    """NumPy equivalent of the search jitter used in STARK-ST2 training (processing.py::_get_jittered_box).

    The size is scaled by exp(N(0,1) * scale_jitter) and the centre is shifted uniformly within a
    window of width sqrt(w'h') * center_jitter.
    Note: with center_jitter=4.5 the target can move up to the border of the crop (as in training).
    """
    x, y, w, h = bbox
    jw = w * math.exp(rng.randn() * scale_jitter)
    jh = h * math.exp(rng.randn() * scale_jitter)
    max_offset = math.sqrt(jw * jh) * center_jitter
    cx = x + w / 2 + max_offset * (rng.rand() - 0.5)
    cy = y + h / 2 + max_offset * (rng.rand() - 0.5)
    return [cx - jw / 2, cy - jh / 2, jw, jh]


def _image_coverage(cx, cy, half, img_w, img_h):
    """Fraction of a square crop (centre (cx, cy), half side `half`) that lies inside the image."""
    ix = max(0.0, min(cx + half, img_w) - max(cx - half, 0.0))
    iy = max(0.0, min(cy + half, img_h) - max(cy - half, 0.0))
    return min(1.0, (ix * iy) / ((2 * half) ** 2))


def negative_search_box(bbox, img_w, img_h, search_factor, margin_ratio=0.02):
    """Returns a box in the same frame whose search crop does NOT contain the target at all.

    sample_target() crops a square region centred on the box with side ceil(sqrt(w*h) * search_factor).
    For the target to lie completely outside this crop, the distance between the centres must exceed
    (half crop side + half target size) along at least one axis.

    8 candidate directions (4 axis-aligned + 4 diagonal) are tried and the one whose crop has the largest
    fraction inside the image (coverage) is chosen. The box is NOT clamped to the image, because clamping
    could move the target back into the crop. The part outside the image is zero-padded by sample_target.

    Returns:
        (box, coverage): box = [x, y, w, h] (same w, h, so the crop size is unchanged),
                         coverage = fraction of the crop inside the image (0-1).
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    crop_sz = math.ceil(math.sqrt(w * h) * search_factor)
    half = crop_sz / 2
    margin = max(2.0, margin_ratio * crop_sz)  # slack for the round() in sample_target
    dx = half + w / 2 + margin
    dy = half + h / 2 + margin

    candidates = [
        (cx + dx, cy), (cx - dx, cy), (cx, cy + dy), (cx, cy - dy),
        (cx + dx, cy + dy), (cx + dx, cy - dy), (cx - dx, cy + dy), (cx - dx, cy - dy),
    ]
    best = max(candidates, key=lambda c: _image_coverage(c[0], c[1], half, img_w, img_h))
    coverage = _image_coverage(best[0], best[1], half, img_w, img_h)
    return [best[0] - w / 2, best[1] - h / 2, w, h], coverage
