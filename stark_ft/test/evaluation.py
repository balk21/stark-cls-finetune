"""
Detection-style evaluation (every frame = one "image", the target = a single object).

Metric families:
  1. COCO mAP / AP50 / AP75 (pycocotools) — the primary metrics
       - mAP is COCO's main "AP": precision averaged over recall and over the IoU thresholds
         0.50:0.05:0.95. AP50 / AP75 use a single IoU threshold (0.50 / 0.75).
       - The frames after the first (init) frame in which the target is visible are evaluated.
       - Predictions with a score below the score threshold (eval_score_thr) are not counted.
       - Coordinates are rounded to integers.
  2. "Found / not found" metrics at a fixed threshold (eval_score_thr, eval_iou_thr):
       the tracker "claims the target" when score >= threshold.
         target visible + claimed + IoU >= iou_thr  -> TP
         target visible + claimed + IoU <  iou_thr  -> FP + FN (wrong location)
         target visible + not claimed               -> FN
         target absent  + claimed                   -> FP
         target absent  + not claimed               -> TN
       Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1.
  3. The threshold that maximises F (VOT-LT protocol, same method as vot-toolkit 0.5.3 vot/analysis/tpr.py):
       - Candidate thresholds: the scores of all sequences are pooled and `resolution` thresholds are picked
         with determine_thresholds() (98 evenly spaced scores + inf and -inf).
       - P and R are computed per sequence at every threshold. The ONLY difference from vot-toolkit: P/R are
         not IoU-weighted but computed with the TP/FP/FN counts of item 2 (COCOeval matching, IoU >= iou_thr).
         If no prediction passes the threshold, P = 1 and R = 0 as in vot-toolkit.
       - P and R are averaged over sequences; F = 2PR / (P + R) is computed from these averages.
       - The threshold with the largest F is selected (first one on ties, i.e. the highest threshold);
         one threshold for all sequences.
"""
import contextlib
import io
import math
from pathlib import Path

import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError as e:  # pragma: no cover
    raise ImportError("pycocotools is required (it is part of the vot1 environment)") from e


# ------------------------------------------------------------------ reading
def parse_box(line: str):
    """Converts an 'x,y,w,h' line (or an 8-value polygon) to [x, y, w, h].
    Returns None for NaN, empty or single-value lines ('1' = init marker) and for boxes with w/h <= 0."""
    if line is None:
        return None
    parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p.strip()]
    if len(parts) < 4:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    if any(math.isnan(v) for v in vals):
        return None
    if len(vals) >= 8:
        xs, ys = vals[0:8:2], vals[1:8:2]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    else:
        x, y, w, h = vals[:4]
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def read_groundtruth(path):
    with open(path) as f:
        return [parse_box(line) for line in f.read().splitlines() if line.strip()]


def _read_values(path, n):
    vals = []
    if Path(path).is_file():
        with open(path) as f:
            for line in f.read().splitlines():
                try:
                    vals.append(float(line))
                except ValueError:
                    vals.append(float("nan"))
    vals += [float("nan")] * (n - len(vals))
    return vals[:n]


def read_vot_result(pred_dir, seq):
    """VOT long-term result: line 0 = init marker ('1'), the following lines are boxes.
    Returns: boxes (list, may contain None), scores (list), times (list), all as long as the sequence."""
    pred_dir = Path(pred_dir)
    with open(pred_dir / f"{seq}_001.txt") as f:
        lines = f.read().splitlines()
    boxes = [None] + [parse_box(l) for l in lines[1:]]
    n = len(lines)
    scores = _read_values(pred_dir / f"{seq}_001_confidence.value", n)
    times = _read_values(pred_dir / f"{seq}_001_time.value", n)
    scores[0] = float("nan")
    return boxes, scores, times


def iou(a, b):
    if a is None or b is None:
        return float("nan")
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _clip_box(b, w, h):
    x, y = min(max(b[0], 0.0), w), min(max(b[1], 0.0), h)
    return [x, y, min(max(b[2], 0.0), w - x), min(max(b[3], 0.0), h - y)]


def got10k_ious(gt_boxes, boxes, size):
    """IoUs as the GOT-10k toolkit computes them (ExperimentGOT10k / rect_iou with bound): the frames after the first
    where the target is visible, boxes clipped to the image (size = (width, height))."""
    eps = np.finfo(float).eps
    w, h = size
    out = []
    for g, p in zip(gt_boxes[1:], boxes[1:]):
        if g is None:
            continue
        if p is None:
            out.append(0.0)
            continue
        a, b = _clip_box(p, w, h), _clip_box(g, w, h)
        ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        inter = ix * iy
        out.append(min(max(inter / (a[2] * a[3] + b[2] * b[3] - inter + eps), 0.0), 1.0))
    return np.array(out, dtype=float)


def got10k_metrics(ious) -> dict:
    """AO (average overlap), SR0.50 and SR0.75 (success rates) of the GOT-10k benchmark."""
    ious = np.asarray(ious, dtype=float)
    if not len(ious):
        return {"AO": float("nan"), "SR50": float("nan"), "SR75": float("nan")}
    return {"AO": float(ious.mean()), "SR50": float((ious > 0.5).mean()), "SR75": float((ious > 0.75).mean())}


# ------------------------------------------------------------------ COCO
def _coco_ap(images, annotations, detections):
    """images: [img_id], annotations: [(img_id, box)], detections: [(img_id, box, score)]."""
    nan3 = {"mAP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
    if not annotations:
        return nan3
    if not detections:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}
    gt = COCO()
    gt.dataset = {
        "info": {"description": "VOT-LT GT"},
        "images": [{"id": int(i), "width": 0, "height": 0} for i in images],
        "annotations": [{"id": k + 1, "image_id": int(i), "category_id": 1, "bbox": list(map(float, b)),
                         "area": float(b[2] * b[3]), "iscrowd": 0} for k, (i, b) in enumerate(annotations)],
        "categories": [{"id": 1, "name": "object", "supercategory": "object"}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes([{"image_id": int(i), "category_id": 1, "bbox": list(map(float, b)), "score": float(s)}
                         for i, b, s in detections])
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.catIds = [1]
        ev.params.imgIds = sorted(int(i) for i in images)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # stats[0] = AP@[.50:.95] (reported as mAP), stats[1] = AP@.50, stats[2] = AP@.75
    return {"mAP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}


def coco_ap(records, score_thr):
    """mAP / AP50 / AP75. records: [(img_id, gt_box|None, pred_box|None, score)]; init frames already excluded.
    Frames without the target are not evaluated, predictions with score < score_thr are not counted (a missing
    score counts as 1), boxes are rounded to integers."""
    def rounded(b):  # parse_box has already dropped NaN and w/h <= 0 boxes
        return None if b is None else [int(round(v)) for v in b]

    images, anns, dets = [], [], []
    for img_id, g, p, s in records:
        if g is None:
            continue
        images.append(img_id)
        anns.append((img_id, rounded(g)))
        s = s if np.isfinite(s) else 1.0
        if p is not None and s >= score_thr:
            dets.append((img_id, rounded(p), s))
    return _coco_ap(images, anns, dets)


# ------------------------------------------------------------------ P/R/F1 at a fixed threshold
def operating_point(records, score_thr, iou_thr):
    tp = fp = fn = tn = 0
    for _, g, p, s in records:
        detected = p is not None and np.isfinite(s) and s >= score_thr
        if g is not None:
            if detected and iou(g, p) >= iou_thr:
                tp += 1
            elif detected:
                fp += 1
                fn += 1
            else:
                fn += 1
        else:
            if detected:
                fp += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    absent = sum(1 for r in records if r[1] is None)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision, "recall": recall, "F1": f1,
            "absent_reject_rate": (tn / absent) if absent else float("nan")}


# ------------------------------------------------------------------ F-max threshold (VOT-LT method)
def determine_thresholds(scores, resolution=100):
    """Identical to vot-toolkit 0.5.3 vot/analysis/tpr.py::determine_thresholds."""
    scores = sorted((s for s in scores if not math.isnan(s)), reverse=True)
    if len(scores) > resolution - 2:
        delta = math.floor(len(scores) / (resolution - 2))
        idxs = np.round(np.linspace(delta, len(scores) - delta, num=resolution - 2)).astype(int)
        thresholds = [scores[idx] for idx in idxs]
    else:
        thresholds = scores
    return [math.inf] + thresholds + [-math.inf]


def pr_curve(records, thresholds, iou_thr):
    """(P, R) at every threshold for one sequence. Same counting as operating_point(), vectorised."""
    scores = np.array([s if (p is not None and np.isfinite(s)) else -np.inf for _, _, p, s in records])
    has_pred = np.array([p is not None and np.isfinite(s) for _, _, p, s in records])
    visible = np.array([g is not None for _, g, _, _ in records])
    correct = np.array([g is not None and p is not None and iou(g, p) >= iou_thr for _, g, p, _ in records])
    n_visible = int(visible.sum())

    thr = np.asarray(thresholds, dtype=float)[:, None]           # (T, 1)
    detected = has_pred[None, :] & (scores[None, :] >= thr)      # (T, N)
    tp = (detected & correct[None, :]).sum(axis=1)
    n_det = detected.sum(axis=1)
    precision = np.where(n_det > 0, tp / np.maximum(n_det, 1), 1.0)
    recall = np.where(n_det > 0, tp / max(n_visible, 1), 0.0)
    return precision, recall


def f_curve_from_sequences(seq_records, iou_thr, resolution=100):
    """seq_records: {sequence: records}. Returns: (curve dict, best point dict, per-sequence (P, R) curves)."""
    all_scores = [s for recs in seq_records.values() for _, _, p, s in recs if p is not None and np.isfinite(s)]
    thresholds = determine_thresholds(all_scores, resolution)
    per_seq = {name: pr_curve(recs, thresholds, iou_thr) for name, recs in seq_records.items()}
    precision = np.mean([pr[0] for pr in per_seq.values()], axis=0)
    recall = np.mean([pr[1] for pr in per_seq.values()], axis=0)
    denom = precision + recall
    f = np.where(denom > 0, 2 * precision * recall / np.where(denom > 0, denom, 1), 0.0)
    best = int(np.argmax(f))  # like list.index(max(...)): first on ties (highest threshold)
    curve = {"threshold": thresholds, "precision": precision.tolist(), "recall": recall.tolist(), "F": f.tolist()}
    best_point = {"threshold": float(thresholds[best]), "precision": float(precision[best]),
                  "recall": float(recall[best]), "F": float(f[best]), "index": best,
                  "resolution": resolution, "iou_thr": iou_thr}
    return curve, best_point, per_seq


# ------------------------------------------------------------------ sequence level
def evaluate_sequence(gt_boxes, pred_boxes, scores, score_thr, iou_thr, img_id_offset=0):
    """Returns all metrics and the per-frame records for one sequence."""
    n = len(gt_boxes)
    pred_boxes = list(pred_boxes) + [None] * (n - len(pred_boxes))
    scores = list(scores) + [float("nan")] * (n - len(scores))
    records = [(img_id_offset + i, gt_boxes[i], pred_boxes[i], scores[i]) for i in range(1, n)]

    visible_ious = [iou(g, p) for _, g, p, _ in records if g is not None]
    visible_ious = [v if np.isfinite(v) else 0.0 for v in visible_ious]
    metrics = {
        "frames": n - 1,
        "visible_frames": sum(1 for r in records if r[1] is not None),
        "absent_frames": sum(1 for r in records if r[1] is None),
        "mean_iou_visible": float(np.mean(visible_ious)) if visible_ious else float("nan"),
    }
    metrics.update(coco_ap(records, score_thr))
    metrics.update(operating_point(records, score_thr, iou_thr))
    return metrics, records
