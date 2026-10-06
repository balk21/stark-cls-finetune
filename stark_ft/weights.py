"""
Weights a test can use, kept apart by origin (the `weights` test parameter):

  "official"      STARK's published weights (github.com/researchmm/Stark), trained by STARK on LaSOT + GOT-10k + COCO +
                  TrackingNet (model_config "*_got10k_only": on GOT-10k only). Stored in
                  <checkpoints>/<stark_st2|stark_s>/<model_config>/ and downloaded automatically when first used.
  "<run name>"    a training run of this repository (<train_outputs>/<run name>/final.pth.tar). Its origin is stored
                  in the run's train_config.json:
                    "imagenet"  trained here from ImageNet: our stage 1 -> our stage 2, only on the run's datasets
                    "official"  stage 2 trained here on top of the official weights (run name contains "on-official")
                    "external"  trained here on top of another checkpoint file
  file / path     any other checkpoint: a file name is looked up in <checkpoints>/<stark_st2|stark_s>/<model_config>/,
                  a value containing "/" is a path (relative paths: repository root).
"""
import json
from pathlib import Path

from stark_ft.paths import REPO_ROOT, Paths, get_paths
from stark_ft.setup_utils import OFFICIAL_CHECKPOINTS, download_checkpoints

OFFICIAL = "official"
ORIGIN_TEXT = {
    "imagenet": "trained here from ImageNet (own stage 1 -> own stage 2)",
    "official": "trained here on top of the official STARK weights",
    "external": "trained here on top of another checkpoint file",
}
OFFICIAL_TEXT = "STARK's published weights, trained on LaSOT + GOT-10k + COCO + TrackingNet"
OFFICIAL_GOT10K_TEXT = "STARK's published weights, trained on GOT-10k only"


def _is_file_ref(weights: str) -> bool:
    return "/" in weights or "\\" in weights or weights.endswith((".pth.tar", ".pth"))


def run_name(weights: str) -> str:
    """The run name if `weights` refers to a training run ("train:<run>" is the earlier syntax), else ''."""
    if weights == OFFICIAL or _is_file_ref(weights):
        return ""
    return weights[len("train:"):] if weights.startswith("train:") else weights


def official_path(model: str, model_config: str, paths: Paths) -> Path:
    folder, fname, _ = OFFICIAL_CHECKPOINTS[(model, model_config)]
    return paths.checkpoints / folder / model_config / fname


def path_of(weights: str, model: str, model_config: str, paths: Paths) -> Path:
    if weights == OFFICIAL:
        return official_path(model, model_config, paths)
    run = run_name(weights)
    if run:
        return paths.train_outputs / run / "final.pth.tar"
    if "/" in weights or "\\" in weights:
        p = Path(weights).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    folder = OFFICIAL_CHECKPOINTS[(model, model_config)][0]
    return paths.checkpoints / folder / model_config / weights


def read_run(run_dir: Path) -> dict:
    meta = json.loads((Path(run_dir) / "train_config.json").read_text())
    cfg = meta["config"]
    if "origin" not in meta:  # runs started before the origin was recorded
        meta["origin"] = "imagenet" if cfg.get("stage") == 1 else (
            "official" if cfg.get("init") == OFFICIAL else "imagenet")
    return meta


def describe(weights: str, model: str, model_config: str, paths: Paths = None) -> dict:
    """What `weights` refers to: path, origin text, whether it can be used now (and why not)."""
    paths = paths or get_paths()
    path = path_of(weights, model, model_config, paths)
    info = {"weights": weights, "path": str(path), "exists": path.is_file(), "problem": None}
    if weights == OFFICIAL:
        info["origin"] = OFFICIAL_GOT10K_TEXT if "got10k_only" in model_config else OFFICIAL_TEXT
        if not path.is_file():
            info["note"] = "downloaded automatically when the run starts"
        return info
    run = run_name(weights)
    if not run:
        info["origin"] = "checkpoint file"
        if not path.is_file():
            info["problem"] = f"Checkpoint file not found: {path}"
        return info
    run_dir = paths.train_outputs / run
    if not (run_dir / "train_config.json").is_file():
        info["problem"] = (f"No training run {run!r} in {paths.train_outputs}. List the available weights with "
                           "`python -m stark_ft weights` (notebook: nb.list_weights()).")
        return info
    meta = read_run(run_dir)
    cfg = meta["config"]
    info["origin"] = ORIGIN_TEXT.get(meta["origin"], meta["origin"])
    info["datasets"] = cfg.get("datasets")
    if model != "stark_st":
        info["problem"] = f"{run!r} is a STARK-ST training run; set model='stark_st'"
    elif cfg.get("stage") != 2:
        info["problem"] = (f"{run!r} is a stage-1 run: its classification head is not trained, so it cannot be "
                           f"tested. Train stage 2 on top of it (the same parameters with stage=2).")
    elif cfg.get("model_config") != model_config:
        info["problem"] = (f"{run!r} was trained with model_config={cfg.get('model_config')!r}; "
                           "set the same model_config")
    elif not path.is_file():
        info["problem"] = f"{run!r} has not finished yet (no final.pth.tar)"
    return info


def resolve(weights: str, model: str, model_config: str, paths: Paths = None, download: bool = True) -> Path:
    """The checkpoint file for a test; downloads the official weights if they are missing."""
    paths = paths or get_paths()
    info = describe(weights, model, model_config, paths)
    if info["problem"]:
        raise FileNotFoundError(info["problem"])
    path = Path(info["path"])
    if weights == OFFICIAL and not path.is_file():
        if not download:
            raise FileNotFoundError(f"Official weights not downloaded yet: {path}")
        download_checkpoints([(model, model_config)])
    return path


def list_weights(paths: Paths = None) -> dict:
    """All weights, grouped by origin: official ones (downloaded or not) and the training runs of this repository."""
    from stark_ft.train.report import list_runs
    paths = paths or get_paths()
    official = [{"model": m, "model_config": c, "path": str(official_path(m, c, paths)),
                 "downloaded": official_path(m, c, paths).is_file(),
                 "trained_on": "GOT-10k" if "got10k_only" in c else "LaSOT + GOT-10k + COCO + TrackingNet"}
                for (m, c) in OFFICIAL_CHECKPOINTS]
    runs = {"imagenet": [], "official": [], "external": []}
    for st in list_runs(paths):
        runs.setdefault(st["origin"], []).append(st)
    return {"official": official, "runs": runs}
