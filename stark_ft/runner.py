"""
Runs an experiment end to end: VOT workspace setup -> `vot evaluate` -> collecting results -> analysis.

Output folder layout (outputs/<experiment name>/):
    experiment.json       All parameters, checkpoint/dataset paths, sequence list, timestamp
    run.log               vot-toolkit output
    run_status.json       Which sequences are complete / missing
    vot_workspace/        vot-toolkit workspace (config.yaml, trackers.ini, sequences/list.txt, results/)
    predictions/<seq>/    <seq>_001.txt (bbox), _confidence.value (score), _time.value (time), frames.csv
    tracker_logs/<seq>/   finetune_loss.txt, events.txt (template / fine-tuning update frames)
    plots/<seq>/          iou_conf.png, finetune_loss.png
    metrics/              metrics.xlsx, metrics.json, summary.txt, f_curve.csv, f_curve.png
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from stark_ft.config import ExperimentConfig
from stark_ft.paths import REPO_ROOT, Paths, get_paths, list_sequences

TRACKER_ID = "stark_clean"
VOT_RESULTS_SUBDIR = Path("results") / TRACKER_ID / "longterm"
# Code that determines the tracker output. If it changes, an experiment must not be resumed: completed sequences
# (skipped by vot-toolkit, which is why `vot evaluate` runs without -f) would come from the old code.
TRACKING_CODE = ("lib", "model_configs", "stark_ft/vot_entry.py", "stark_ft/tracker_factory.py", "stark_ft/vot_trax.py")
TRACKING_EXCLUDE = ("lib/train", "lib/config/stark_st1", "model_configs/stark_st1")  # training only


class ExperimentExistsError(RuntimeError):
    pass


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def tracking_code_hash() -> str:
    """Hash of the code that determines the tracker output (independent of git and of documentation changes)."""
    h = hashlib.sha1()
    for entry in TRACKING_CODE:
        root = REPO_ROOT / entry
        files = sorted(root.rglob("*")) if root.is_dir() else [root]
        for f in files:
            rel = f.relative_to(REPO_ROOT).as_posix()
            if any(rel == e or rel.startswith(e + "/") for e in TRACKING_EXCLUDE):
                continue
            if f.is_file() and f.suffix in (".py", ".yaml"):
                h.update(str(f.relative_to(REPO_ROOT)).encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def resolve_sequences(cfg: ExperimentConfig, paths: Paths):
    available = list_sequences(paths.dataset)
    if not available:
        raise FileNotFoundError(
            f"No VOT sequences found in the dataset folder: {paths.dataset}\n"
            "Download the dataset with 00_setup.ipynb or set the 'dataset' path in configs/paths.local.yaml.")
    if cfg.sequences == "all":
        return available
    missing = [s for s in cfg.sequences if s not in available]
    if missing:
        raise ValueError(f"Sequence(s) not in the dataset: {missing}\nAvailable sequences: {available}")
    return list(cfg.sequences)


def _stack_definition(cfg: ExperimentConfig) -> dict:
    """Copy of the VOT-LT2020 stack (vot/stack/votlt2020.yaml); redetection is optional.
    The `dataset` key is deliberately left empty, so vot-toolkit never tries to download anything."""
    experiments = {"longterm": {"type": "unsupervised", "repetitions": 1}}
    if cfg.run_redetection:
        experiments["redetection"] = {
            "type": "unsupervised",
            "transformers": [{"type": "redetection", "length": 200, "initialization": 5,
                              "padding": 2, "scaling": 3}],
        }
    return {"title": "VOT-LT2020 (stark-cls-finetune)", "experiments": experiments}


def _write_workspace(ws: Path, cfg: ExperimentConfig, paths: Paths, sequences, experiment_json: Path):
    (ws / "sequences").mkdir(parents=True, exist_ok=True)
    # list.txt contains absolute paths: nothing is written into the dataset folder and no copies/symlinks
    # are needed. vot-toolkit takes the sequence name from the folder name (basename).
    with open(ws / "sequences" / "list.txt", "w") as f:
        for s in sequences:
            f.write(f"{(paths.dataset / s).resolve()}\n")
    with open(ws / "config.yaml", "w") as f:
        yaml.safe_dump({"registry": ["./trackers.ini"], "stack": _stack_definition(cfg),
                        "sequences": "sequences"}, f, sort_keys=False)
    with open(ws / "trackers.ini", "w") as f:
        f.write(f"[{TRACKER_ID}]\n")
        f.write(f"label = {cfg.experiment_name}\n")
        f.write("protocol = traxpython\n")
        f.write("command = stark_ft.vot_entry\n")
        f.write(f"paths = {REPO_ROOT}\n")
        f.write(f"python = {sys.executable}\n")
        f.write(f"timeout = {int(cfg.tracker_timeout)}\n")
        # vot-toolkit overwrites LD_LIBRARY_PATH of the tracker process with `linkpaths` (empty by default).
        # Where the GPU driver is only found through LD_LIBRARY_PATH (e.g. /usr/lib64-nvidia on Google Colab),
        # the tracker would then not see the GPU, so the current value is passed on explicitly.
        lib_dirs = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
        if lib_dirs:
            f.write(f"linkpaths = {os.pathsep.join(lib_dirs)}\n")
        f.write(f"env_STARK_CLEAN_EXPERIMENT = {experiment_json}\n")


def _expected_lengths(paths: Paths, sequences):
    lengths = {}
    for s in sequences:
        with open(paths.dataset / s / "groundtruth.txt") as f:
            lengths[s] = sum(1 for line in f if line.strip())
    return lengths


def _collect_results(out_dir: Path, paths: Paths, sequences):
    """Copies the VOT results into predictions/ and returns the completion status."""
    src_root = out_dir / "vot_workspace" / VOT_RESULTS_SUBDIR
    expected = _expected_lengths(paths, sequences)
    status = {"complete": [], "incomplete": {}, "missing": []}
    for s in sequences:
        src = src_root / s
        bbox_file = src / f"{s}_001.txt"
        if not bbox_file.is_file():
            status["missing"].append(s)
            continue
        with open(bbox_file) as f:
            n = sum(1 for _ in f)
        dst = out_dir / "predictions" / s
        dst.mkdir(parents=True, exist_ok=True)
        for suffix in ("_001.txt", "_001_confidence.value", "_001_time.value"):
            if (src / f"{s}{suffix}").is_file():
                shutil.copy2(src / f"{s}{suffix}", dst / f"{s}{suffix}")
        if n == expected[s]:
            status["complete"].append(s)
        else:
            status["incomplete"][s] = {"lines": n, "expected": expected[s]}
    return status


def _print_tracker_errors(log_dir: Path, since: float, max_logs=2, max_lines=40):
    """Prints the end of the tracker error logs written by vot-toolkit during this run."""
    logs = sorted((p for p in log_dir.glob("*.log") if p.stat().st_mtime >= since), key=lambda p: p.stat().st_mtime)
    for log in logs[-max_logs:]:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"\n----- tracker output: {log.name} (last {min(len(lines), max_lines)} lines) -----")
        print("\n".join(lines[-max_lines:]))


def _stream_process(cmd, log_path: Path, env, verbose=True):
    with open(log_path, "ab") as log:
        log.write(f"\n===== {datetime.datetime.now().isoformat()} :: {' '.join(cmd)}\n".encode())
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(REPO_ROOT))
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            log.write(chunk)
            log.flush()
            if verbose:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        return proc.wait()


def prepare_experiment(cfg: ExperimentConfig, paths: Paths = None, overwrite: bool = False) -> Path:
    """Prepares the output folder and the VOT workspace. If an experiment with the same name and the SAME
    parameters was started before, it is resumed (vot-toolkit skips completed sequences)."""
    cfg.validate()
    paths = paths or get_paths()
    checkpoint = cfg.checkpoint_path(paths)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Download it with 00_setup.ipynb, put the file at this location, or pass a path via `checkpoint`.")
    sequences = resolve_sequences(cfg, paths)

    out_dir = paths.outputs / cfg.experiment_name
    meta_path = out_dir / "experiment.json"
    code_hash = tracking_code_hash()
    if out_dir.exists():
        same = False
        old_code = None
        if meta_path.is_file():
            with open(meta_path) as f:
                old = json.load(f)
            try:  # parameters added later are compared using their default values
                old_cfg = ExperimentConfig.from_dict(old.get("config", {})).tracking_dict()
            except ValueError:
                old_cfg = None
            same = old_cfg == cfg.tracking_dict() and old.get("checkpoint") == str(checkpoint)
            old_code = old.get("code_hash")
        if overwrite:
            if paths.outputs.resolve() not in out_dir.resolve().parents:
                raise RuntimeError(f"Safety check: {out_dir} is not inside the outputs folder; not deleted")
            shutil.rmtree(out_dir)
        elif not same:
            raise ExperimentExistsError(
                f"'{out_dir}' already exists with different parameters.\n"
                "Use a different `name`, or overwrite=True to delete the old results.")
        elif old_code is not None and old_code != code_hash:
            raise ExperimentExistsError(
                f"'{out_dir}' was started with a different version of the tracking code "
                f"({old_code} -> {code_hash}).\nResuming would mix results of two code versions. "
                "Use overwrite=True for a fresh run, or a different `name`.")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": cfg.to_dict(),
        "experiment_name": cfg.experiment_name,
        "output_dir": str(out_dir),
        "checkpoint": str(checkpoint),
        "dataset": str(paths.dataset),
        "sequences": sequences,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "code_hash": code_hash,
        "python": sys.executable,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    _write_workspace(out_dir / "vot_workspace", cfg, paths, sequences, meta_path)
    return out_dir


def run_experiment(cfg: ExperimentConfig, paths: Paths = None, overwrite: bool = False,
                   analyze: bool = True, verbose: bool = True) -> Path:
    """Runs the experiment and (if analyze=True) produces metrics and plots. Returns the output folder."""
    paths = paths or get_paths()
    out_dir = prepare_experiment(cfg, paths, overwrite=overwrite)
    with open(out_dir / "experiment.json") as f:
        sequences = json.load(f)["sequences"]

    print(f"Experiment  : {cfg.experiment_name}")
    print(f"Output      : {out_dir}")
    print(f"Sequences   : {len(sequences)}")
    print(f"Checkpoint  : {cfg.checkpoint_path(paths)}")
    print("-" * 70, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # --persist: if a sequence fails, the remaining sequences still run (failures are listed in run_status.json)
    cmd = [sys.executable, "-m", "vot", "evaluate", "--persist", "--workspace", str(out_dir / "vot_workspace"),
           TRACKER_ID]
    started = datetime.datetime.now().timestamp()
    returncode = _stream_process(cmd, out_dir / "run.log", env, verbose=verbose)

    status = _collect_results(out_dir, paths, sequences)
    status["vot_returncode"] = returncode
    status["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(out_dir / "run_status.json", "w") as f:
        json.dump(status, f, indent=2)

    print("-" * 70)
    print(f"Completed sequences: {len(status['complete'])}/{len(sequences)}")
    if status["missing"] or status["incomplete"]:
        print(f"WARNING - sequences without results : {status['missing']}")
        print(f"WARNING - sequences with missing lines: {status['incomplete']}")
        print(f"Tracker error output: {out_dir / 'vot_workspace' / 'logs'}  (general output: run.log)")
        print("Running the same experiment again skips completed sequences and retries the missing ones.")
        _print_tracker_errors(out_dir / "vot_workspace" / "logs", since=started)

    if analyze and status["complete"]:
        from stark_ft.analysis import analyze_experiment
        analyze_experiment(out_dir, paths=paths)
    return out_dir


def run_many(configs, paths: Paths = None, overwrite: bool = False, analyze: bool = True, verbose: bool = True):
    """Runs several experiments one after another; if one fails, the others still run."""
    results = {}
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(configs)}] {cfg.experiment_name}\n{'=' * 70}")
        try:
            results[cfg.experiment_name] = run_experiment(cfg, paths, overwrite, analyze, verbose)
        except Exception as e:  # noqa: BLE001 - one failure in a batch must not stop the others
            print(f"ERROR ({cfg.experiment_name}): {e}")
            results[cfg.experiment_name] = e
    return results
