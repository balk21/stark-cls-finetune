"""
Setup helpers: environment check, checkpoint and dataset download, quick smoke test.
"""
import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from stark_ft import vot_data
from stark_ft.config import ExperimentConfig
from stark_ft.paths import get_paths, list_sequences

# Official STARK checkpoints (individual files in the Google Drive folders listed in STARK's MODEL_ZOO.md)
OFFICIAL_CHECKPOINTS = {
    ("stark_st", "baseline"): ("STARKST_ep0050.pth.tar", "1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w"),
    ("stark_st", "baseline_got10k_only"): ("STARKST_ep0050.pth.tar", "16eK2CxHXYNo3YZ7oEp2FBOojdV1pB0G6"),
    ("stark_st", "baseline_R101"): ("STARKST_ep0050.pth.tar", "1t3oQOF8XyqA3nnMgT4IzzueREEus81uP"),
    ("stark_st", "baseline_R101_got10k_only"): ("STARKST_ep0050.pth.tar", "1IhX1ecuzk8OfAxR74N8BTXG8BE43L3qD"),
    ("stark_s", "baseline"): ("STARKS_ep0500.pth.tar", "1zlecl8DJKZk2Waok52S4gkB0DVRoN2Vl"),
    ("stark_s", "baseline_got10k_only"): ("STARKS_ep0500.pth.tar", "1IJydxCXbpQF6p5MDIvvYWi6xAjDspEiu"),
}
TESTED_GPU_SERIES = "NVIDIA RTX 3000 / 4000 series (Ampere / Ada, compute capability 8.x)"


def check_environment(verbose=True) -> dict:
    info = {"python": sys.version.split()[0], "executable": sys.executable}
    problems = []
    for mod in ("torch", "torchvision", "cv2", "numpy", "pandas", "pycocotools", "yaml", "easydict", "openpyxl",
                "matplotlib", "vot", "trax"):
        try:
            m = importlib.import_module(mod)
            info[mod] = getattr(m, "__version__", "ok")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"MISSING ({e.__class__.__name__})"
            problems.append(f"could not import '{mod}'")
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = f"{cap[0]}.{cap[1]}"
            arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
            if arch_list and f"sm_{cap[0]}{cap[1]}" not in arch_list:
                problems.append(f"This PyTorch build was not compiled for {info['gpu']} (sm_{cap[0]}{cap[1]}). "
                                f"Tested GPUs: {TESTED_GPU_SERIES}")
        else:
            problems.append("CUDA is not available (a GPU is required)")
    except ImportError:
        pass
    if "vot1" not in sys.prefix:
        problems.append(f"The active Python does not look like the 'vot1' environment ({sys.prefix}).")
    paths = get_paths()
    info["paths"] = paths.as_dict()
    info["n_sequences"] = len(list_sequences(paths.dataset))
    info["problems"] = problems
    if verbose:
        for k, v in info.items():
            if k not in ("paths", "problems"):
                print(f"{k:20s}: {v}")
        print("paths:")
        for k, v in paths.as_dict().items():
            print(f"  {k:12s}: {v}")
        print("\nPROBLEMS:\n  - " + "\n  - ".join(problems) if problems else "\nThe environment looks ready.")
    return info


def _ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    return importlib.import_module("gdown")


def download_checkpoints(models=(("stark_st", "baseline_R101"),), force=False):
    """Downloads official checkpoints into <checkpoints>/<stark_st2|stark_s>/<model_config>/."""
    paths = get_paths()
    gdown = None
    for model, model_config in models:
        key = (model, model_config)
        if key not in OFFICIAL_CHECKPOINTS:
            raise ValueError(f"No official checkpoint for {key}. Options: {list(OFFICIAL_CHECKPOINTS)}")
        fname, file_id = OFFICIAL_CHECKPOINTS[key]
        cfg = ExperimentConfig(model=model, model_config=model_config, ft_mode="none")
        target = paths.checkpoints / cfg.model_dir / model_config / fname
        if target.is_file() and not force:
            print(f"Already present: {target}")
            continue
        gdown = gdown or _ensure_gdown()  # only installed when a download is actually needed
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        print(f"Downloading: {model}/{model_config} -> {target}")
        gdown.download(id=file_id, output=str(tmp), quiet=False)
        if not tmp.is_file() or tmp.stat().st_size < 1_000_000:
            raise RuntimeError(f"Download failed: {model}/{model_config}. The Google Drive quota may be exceeded; "
                               f"you can download the file in a browser and put it at {target} "
                               f"(https://drive.google.com/file/d/{file_id}).")
        tmp.rename(target)


def download_dataset(target=None, sequences="all"):
    """Downloads VOT-LT2020 (LTB50) sequences that are not there yet: "all" (50 sequences, 17.6 GB) or a list of
    names. Not needed before experiments: every experiment downloads the sequences it uses (stark_ft/vot_data.py)."""
    paths = get_paths()
    target = Path(target) if target else paths.dataset
    names = vot_data.resolve(sequences, target)
    fetched = vot_data.ensure(names, target, paths.dataset_cache)
    print(f"{len(names) - len(fetched)} of {len(names)} sequence(s) were already present, {len(fetched)} fetched. "
          f"Dataset: {target} ({len(list_sequences(target))} sequences)")
    return target


def disk_usage(path):
    total, used, free = shutil.disk_usage(path)
    return {"total_gb": total / 1e9, "free_gb": free / 1e9}


def smoke_test(cfg: ExperimentConfig = None, sequence: str = None, n_frames: int = 30):
    """Without vot-toolkit: loads the model and tracks the first n_frames frames of a sequence.
    Verifies within seconds that the setup (checkpoint, GPU, dataset) is correct. If no sequence is on disk yet,
    the smallest one (ballet, 57 MB) is downloaded."""
    import cv2
    import numpy as np

    from stark_ft.evaluation import iou, read_groundtruth
    from stark_ft.tracker_factory import build_tracker

    cfg = cfg or ExperimentConfig()
    paths = get_paths()
    seqs = list_sequences(paths.dataset)
    seq = sequence or (seqs[0] if seqs else vot_data.SMOKE_SEQUENCE)
    vot_data.ensure(vot_data.resolve([seq], paths.dataset), paths.dataset, paths.dataset_cache)
    seq_dir = paths.dataset / seq
    frames = sorted((seq_dir / "color").glob("*.jpg"))[:n_frames]
    gt = read_groundtruth(seq_dir / "groundtruth.txt")

    t0 = time.time()
    tracker = build_tracker(cfg, cfg.checkpoint_path(paths))
    print(f"Model loaded ({time.time() - t0:.1f} s): {cfg.checkpoint_path(paths)}")

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
