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
from stark_ft.test.evaluation import read_groundtruth
from stark_ft.test.tracker_factory import build_tracker, shot_samples
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

    seq_dir = Path(imagefile).parent.parent  # <dataset>/<sequence>/color/00000001.jpg
    log_dir = out_dir / "tracker_logs" / seq_dir.name
    log_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(tracker, "log_dir"):
        tracker.log_dir = str(log_dir)

    info = {"init_bbox": [selection.x, selection.y, selection.width, selection.height]}
    shot_frames = []
    if cfg.shots() > 1:  # ft_shots: more frames of this video with their ground-truth box (docs/test.md)
        images = sorted(Path(imagefile).parent.glob("*" + Path(imagefile).suffix))
        shot_frames, info["ft_shots"] = shot_samples(cfg, read_groundtruth(seq_dir / "groundtruth.txt"), images,
                                                     _read_rgb)
    tracker.initialize(_read_rgb(imagefile), info)

    frame_idx = 0
    with open(log_dir / "events.txt", "w") as events:
        events.write(EVENTS_HEADER)
        events.writelines(f"{i},ft_shot,nan\n" for i in shot_frames)
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
                events.write(f"{out['template_frame']},template_update,{out['template_conf']:.6f}\n")
            if out.get("ft_updated"):
                events.write(f"{out['template_frame']},ft_update,{out['template_conf']:.6f}\n")
    handle.quit()


if __name__ == "__main__":
    main()
