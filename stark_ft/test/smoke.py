"""
Smoke test without vot-toolkit: loads the model and tracks the first frames of a sequence.
"""
import time

from stark_ft import weights
from stark_ft.paths import get_paths, list_sequences
from stark_ft.test import vot_data
from stark_ft.test.config import ExperimentConfig


def smoke_test(cfg: ExperimentConfig = None, sequence: str = None, n_frames: int = 30):
    """Without vot-toolkit: loads the model and tracks the first n_frames frames of a sequence.
    Verifies within seconds that the setup (checkpoint, GPU, dataset) is correct. If no sequence is on disk yet,
    the smallest one (ballet, 57 MB) is downloaded."""
    import cv2
    import numpy as np

    from stark_ft.test.evaluation import iou, read_groundtruth
    from stark_ft.test.tracker_factory import build_tracker

    cfg = cfg or ExperimentConfig()
    paths = get_paths()
    seqs = list_sequences(paths.dataset)
    seq = sequence or (seqs[0] if seqs else vot_data.SMOKE_SEQUENCE)
    vot_data.ensure(vot_data.resolve([seq], paths.dataset), paths.dataset, paths.dataset_cache)
    seq_dir = paths.dataset / seq
    frames = sorted((seq_dir / "color").glob("*.jpg"))[:n_frames]
    gt = read_groundtruth(seq_dir / "groundtruth.txt")

    t0 = time.time()
    checkpoint = weights.resolve(cfg.weights, cfg.model, cfg.model_config, paths)  # official: downloaded if missing
    tracker = build_tracker(cfg, checkpoint)
    print(f"Model loaded ({time.time() - t0:.1f} s): {cfg.weights} ({checkpoint})")

    def rgb(p):
        return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)

    t0 = time.time()
    tracker.initialize(rgb(frames[0]), {"init_bbox": gt[0]})
    print(f"Initialize ({cfg.ft_mode}) {time.time() - t0:.2f} s")
    ious, confs, t0 = [], [], time.time()
    for i, p in enumerate(frames[1:], 1):
        out = tracker.track(rgb(p))
        confs.append(out["conf_score"])
        if gt[i] is not None:
            ious.append(iou(gt[i], out["target_bbox"]))
    fps = (len(frames) - 1) / max(time.time() - t0, 1e-6)
    print(f"Sequence: {seq} | {len(frames) - 1} frames | {fps:.1f} FPS | mean IoU: {np.mean(ious):.3f} | "
          f"mean score: {np.mean(confs):.3f}")
    return {"sequence": seq, "fps": fps, "mean_iou": float(np.mean(ious)), "mean_conf": float(np.mean(confs))}
