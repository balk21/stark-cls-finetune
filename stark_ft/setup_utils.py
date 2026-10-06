"""
Shared setup helpers: environment check and download of the official STARK checkpoints.
"""
import importlib
import subprocess
import sys

from stark_ft.paths import get_paths, list_sequences

# Official STARK checkpoints (individual files in the Google Drive folders listed in STARK's MODEL_ZOO.md):
# (model, model_config) -> (folder in <checkpoints>, file name, Google Drive file id)
OFFICIAL_CHECKPOINTS = {
    ("stark_st", "baseline"): ("stark_st2", "STARKST_ep0050.pth.tar", "1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w"),
    ("stark_st", "baseline_got10k_only"): ("stark_st2", "STARKST_ep0050.pth.tar", "16eK2CxHXYNo3YZ7oEp2FBOojdV1pB0G6"),
    ("stark_st", "baseline_R101"): ("stark_st2", "STARKST_ep0050.pth.tar", "1t3oQOF8XyqA3nnMgT4IzzueREEus81uP"),
    ("stark_st", "baseline_R101_got10k_only"): ("stark_st2", "STARKST_ep0050.pth.tar",
                                                "1IhX1ecuzk8OfAxR74N8BTXG8BE43L3qD"),
    ("stark_s", "baseline"): ("stark_s", "STARKS_ep0500.pth.tar", "1zlecl8DJKZk2Waok52S4gkB0DVRoN2Vl"),
    ("stark_s", "baseline_got10k_only"): ("stark_s", "STARKS_ep0500.pth.tar", "1IJydxCXbpQF6p5MDIvvYWi6xAjDspEiu"),
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
        folder, fname, file_id = OFFICIAL_CHECKPOINTS[key]
        target = paths.checkpoints / folder / model_config / fname
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
