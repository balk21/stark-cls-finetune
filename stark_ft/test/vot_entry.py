"""
Entry point that vot-toolkit (TraX protocol) starts as a separate process for every sequence.

It is not run directly; it is registered in the trackers.ini written by runner.py as
    command = stark_ft.test.vot_entry
The experiment parameters are read from the experiment.json file pointed to by the
STARK_CLEAN_EXPERIMENT environment variable.

Frame indices: 0 = first (init) frame, i.e. the line number in the VOT result files minus 1.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import torch

from stark_ft.test.config import ExperimentConfig
from stark_ft.test.tracker_factory import build_tracker
from stark_ft.test.vot_trax import VOT, Rectangle

EVENTS_HEADER = "frame,event,conf_score\n"


def _read_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    exp_file = os.environ.get("STARK_CLEAN_EXPERIMENT")
    if not exp_file:
        raise RuntimeError("STARK_CLEAN_EXPERIMENT environment variable is not set (run via runner.py)")
    with open(exp_file) as f:
        meta = json.load(f)
    cfg = ExperimentConfig.from_dict(meta["config"])
    out_dir = Path(meta["output_dir"])

    torch.set_num_threads(1)
    from lib.test.tracker.stark_st_ft import seed_everything
    seed_everything(cfg.seed)

    # The model is loaded before the TraX handshake (so that loading does not count towards the timeout)
    tracker = build_tracker(cfg, Path(meta["checkpoint"]))

    handle = VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    seq_name = Path(imagefile).parent.parent.name  # <dataset>/<sequence>/color/00000001.jpg
    log_dir = out_dir / "tracker_logs" / seq_name
    log_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(tracker, "log_dir"):
        tracker.log_dir = str(log_dir)

    init_box = [selection.x, selection.y, selection.width, selection.height]
    tracker.initialize(_read_rgb(imagefile), {"init_bbox": init_box})

    frame_idx = 0
    with open(log_dir / "events.txt", "w") as events:
        events.write(EVENTS_HEADER)
        while True:
            imagefile = handle.frame()
            if not imagefile:
                break
            frame_idx += 1
            out = tracker.track(_read_rgb(imagefile))
            x, y, w, h = out["target_bbox"]
            conf = float(out["conf_score"])
            handle.report(Rectangle(x, y, w, h), conf)
            if out.get("template_updated"):
                events.write(f"{frame_idx},template_update,{conf:.6f}\n")
            if out.get("ft_updated"):
                events.write(f"{frame_idx},ft_update,{conf:.6f}\n")
    handle.quit()


if __name__ == "__main__":
    main()
