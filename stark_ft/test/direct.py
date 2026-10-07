"""
Runs a test on GOT-10k with the GOT-10k protocol (one pass over every sequence, no restarts), without vot-toolkit
(vot-toolkit needs the ground truth of every frame, which the GOT-10k test split does not have). The model is loaded
once; the tracker is initialised on the first frame of every sequence (this also resets the fine-tuned head and the
random seed, as a new vot-toolkit process would).

The results are written in the same files as the VOT runs (predictions/<seq>/<seq>_001.txt, _confidence.value,
_time.value; tracker_logs/<seq>/), so the analysis is the same. A sequence's files are written when it is complete;
running the experiment again continues with the sequences that have no results yet. For the test split,
got10k_submission.zip contains the results in the format of the GOT-10k evaluation server.
"""
import os
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import torch

from stark_ft import got10k
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.tracker_factory import build_tracker

EVENTS_HEADER = "frame,event,conf_score\n"
SUBMISSION = "got10k_submission.zip"


def _rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write(path: Path, text: str):
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    os.replace(tmp, path)


def _box(b):
    return ",".join(f"{v:.4f}" for v in b)


def run_sequences(cfg: ExperimentConfig, out_dir: Path, dataset_dir: Path, sequences, checkpoint: Path) -> dict:
    """Tracks every sequence without results. Returns {"complete": [...], "failed": {seq: error}}."""
    from lib.test.tracker.stark_st_ft import seed_everything
    torch.set_num_threads(1)  # as in the vot-toolkit tracker process
    seed_everything(cfg.seed)
    done = [s for s in sequences if (out_dir / "predictions" / s / f"{s}_001.txt").is_file()]
    todo = [s for s in sequences if s not in done]
    status = {"complete": list(done), "failed": {}}
    if done:
        print(f"{len(done)} sequence(s) already have results (skipped); {len(todo)} to run.", flush=True)
    tracker = None
    for i, seq in enumerate(todo, 1):
        try:
            if tracker is None:
                tracker = build_tracker(cfg, checkpoint)
            images, _, _, init_box = got10k.read_sequence(dataset_dir, seq)
            log_dir = out_dir / "tracker_logs" / seq
            log_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(tracker, "log_dir"):
                tracker.log_dir = str(log_dir)
            t0 = time.perf_counter()
            tracker.initialize(_rgb(images[0]), {"init_bbox": init_box})
            times, boxes, confs, events = [time.perf_counter() - t0], [], [], [EVENTS_HEADER]
            for k, path in enumerate(images[1:], 1):
                image = _rgb(path)
                t0 = time.perf_counter()
                out = tracker.track(image)
                times.append(time.perf_counter() - t0)
                conf = float(out["conf_score"])
                boxes.append(out["target_bbox"])
                confs.append(conf)
                if out.get("template_updated"):
                    events.append(f"{out['template_frame']},template_update,{out['template_conf']:.6f}\n")
                if out.get("ft_updated"):
                    events.append(f"{out['template_frame']},ft_update,{out['template_conf']:.6f}\n")
            pred = out_dir / "predictions" / seq
            pred.mkdir(parents=True, exist_ok=True)
            # VOT long-term format (as written by vot-toolkit): line 1 = "1" (initialisation frame)
            _write(pred / f"{seq}_001_confidence.value", "\n" + "".join(f"{c}\n" for c in confs))
            _write(pred / f"{seq}_001_time.value", "".join(f"{t}\n" for t in times))
            _write(log_dir / "events.txt", "".join(events))
            _write(pred / f"{seq}_001.txt", "1\n" + "".join(_box(b) + "\n" for b in boxes))  # written last
            status["complete"].append(seq)
            fps = (len(images) - 1) / max(sum(times[1:]), 1e-9)
            print(f"[{len(done) + i}/{len(sequences)}] {seq}: {len(images)} frames, {fps:.1f} FPS", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - one failing sequence must not stop the others
            status["failed"][seq] = f"{type(e).__name__}: {e}"
            print(f"[{len(done) + i}/{len(sequences)}] {seq}: FAILED", flush=True)
            traceback.print_exc()
    return status


def write_submission(out_dir: Path, dataset_dir: Path, sequences) -> Path:
    """got10k_submission.zip in the format of the GOT-10k server: <seq>/<seq>_001.txt (box of every frame, the
    first one is the given box) and <seq>/<seq>_time.txt (seconds per frame)."""
    path = out_dir / SUBMISSION
    tmp = path.with_name(path.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for seq in sequences:
            pred = out_dir / "predictions" / seq
            lines = (pred / f"{seq}_001.txt").read_text().splitlines()[1:]
            init = got10k.read_sequence(dataset_dir, seq)[3]
            z.writestr(f"{seq}/{seq}_001.txt", "\n".join([_box(init)] + lines) + "\n")
            z.writestr(f"{seq}/{seq}_time.txt", (pred / f"{seq}_001_time.value").read_text())
    os.replace(tmp, path)
    return path
