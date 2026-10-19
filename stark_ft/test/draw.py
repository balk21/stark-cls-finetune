"""
Draws the boxes of a test experiment on the frames of a sequence: ground truth green, prediction red (1 pixel), and
in the top-left corner the frame number, the IoU and the confidence score.

    <experiment>/boxes/<sequence>/<sequence>_<frame>.jpg   the chosen frames, or every frame (frames="all")
    <experiment>/boxes/<sequence>/<sequence>.mp4            every frame (H.264 if ffmpeg is installed, e.g. Colab)

Frame numbers as everywhere in the outputs: 0 = first frame (initialisation, ground-truth box), not evaluated.
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2

from stark_ft import got10k
from stark_ft.paths import Paths, get_paths
from stark_ft.test import datasets
from stark_ft.test.analysis import _dataset_dir
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.evaluation import iou, read_vot_result

GT_COLOR = (0, 255, 0)    # BGR: green
PRED_COLOR = (0, 0, 255)  # BGR: red
# JPEG without chroma subsampling (4:4:4): the 1-pixel coloured lines keep their colour (with the default 4:2:0
# they fade), at about a quarter of the size of PNG
JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]


def _rect(image, box, color):
    x, y, w, h = box
    cv2.rectangle(image, (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h))), color, 1)


def annotate(image, frame, gt, pred, score):
    """Draws on the (BGR) image in place."""
    if gt is not None:
        _rect(image, gt, GT_COLOR)
    if pred is not None:
        _rect(image, pred, PRED_COLOR)
    if frame == 0:
        text = "#0  init"
    else:
        overlap = f"{iou(gt, pred):.2f}" if gt is not None and pred is not None else "-"
        conf = f"{score:.2f}" if score is not None and math.isfinite(score) else "-"
        text = f"#{frame}  IoU {overlap}  conf {conf}"
    scale = max(0.3, image.shape[0] / 1800)  # small text, readable on 360p ... 1080p
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(image, (0, 0), (tw + 6, th + base + 6), (0, 0, 0), -1)
    cv2.putText(image, text, (3, th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _images(dataset: str, dataset_dir: Path, seq: str):
    if datasets.is_got10k(dataset):
        return got10k.read_sequence(dataset_dir, seq)[0]
    return sorted((Path(dataset_dir) / seq / "color").glob("*.jpg"))


def _h264(path: Path):
    """Re-encodes to H.264 (browsers / notebooks play it) if ffmpeg is installed; otherwise keeps MPEG-4."""
    if not shutil.which("ffmpeg"):
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(path)
    elif tmp.exists():
        tmp.unlink()


def draw_sequence(exp_dir, sequence: str, frames=(), video: bool = True, fps: int = 30, paths: Paths = None) -> dict:
    """Writes the chosen frames (frames="all": every frame) and, with video=True, a video of every frame with the
    boxes drawn. Returns {"images": [paths of the frames], "video": path or None}."""
    exp_dir = Path(exp_dir)
    paths = paths or get_paths()
    meta = json.loads((exp_dir / "experiment.json").read_text())
    cfg = ExperimentConfig.from_dict(meta["config"])
    if sequence not in meta["sequences"]:
        raise ValueError(f"{sequence!r} is not a sequence of this experiment: {meta['sequences']}")
    dataset_dir = _dataset_dir(meta, cfg, paths)
    images = _images(cfg.dataset, dataset_dir, sequence) if (Path(dataset_dir) / sequence).is_dir() else []
    if not images:  # e.g. a new Colab session: download / extract the sequence again
        datasets.ensure(cfg.dataset, [sequence], paths)
        dataset_dir = datasets.root(cfg.dataset, paths)
        images = _images(cfg.dataset, dataset_dir, sequence)
    gt, _ = datasets.ground_truth(cfg.dataset, dataset_dir, sequence)
    boxes, scores, _ = read_vot_result(exp_dir / "predictions" / sequence, sequence)
    if len(boxes) != len(images):
        raise RuntimeError(f"{sequence}: {len(boxes)} predicted frames, {len(images)} images (incomplete result?)")

    out_dir = exp_dir / "boxes" / sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = list(range(len(images))) if frames == "all" else sorted({int(f) for f in frames})
    bad = [f for f in wanted if not 0 <= f < len(images)]
    if bad:
        raise ValueError(f"frames {bad} out of range: {sequence} has frames 0 ... {len(images) - 1}")
    result = {"images": [], "video": None}
    writer = None
    todo = range(len(images)) if video else wanted
    wanted = set(wanted)
    for i in todo:
        image = cv2.imread(str(images[i]))
        annotate(image, i, gt[i] if i < len(gt) else None, boxes[i], scores[i])
        if i in wanted:
            path = out_dir / f"{sequence}_{i:06d}.jpg"
            cv2.imwrite(str(path), image, JPEG_PARAMS)
            result["images"].append(path)
        if video:
            if writer is None:
                result["video"] = out_dir / f"{sequence}.mp4"
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(result["video"]), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(image)
    if writer is not None:
        writer.release()
        _h264(result["video"])
    return result
