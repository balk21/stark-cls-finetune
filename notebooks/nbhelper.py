"""
Notebook helpers — use ONLY the standard library + IPython.

They work in whatever Python kernel the notebook is opened with (including the default Jupyter of
Vast.ai / Colab): the actual work is done in a subprocess, `python -m stark_ft ...`, using the Python
of the `vot1` conda environment. On Google Colab, `colab_bootstrap()` (see colab_setup.py) prepares
that environment first.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_NAME = "vot1"
OUTPUT_DIR_MARKER = "OUTPUT_DIR="  # printed by `python -m stark_ft run`
_PATHS_CACHE = {}
_COLAB_INFO = {}  # result of the last colab_bootstrap() in this kernel

_CANDIDATE_PREFIXES = [
    "~/anaconda3", "~/miniconda3", "~/miniforge3", "~/mambaforge", "~/micromamba",
    "/opt/conda", "/opt/miniconda3", "/opt/miniforge3", "/usr/local/miniconda3", "/root/miniconda3",
    "/content/micromamba",  # Google Colab (colab_setup.py)
]
if os.environ.get("MAMBA_ROOT_PREFIX"):
    _CANDIDATE_PREFIXES.insert(0, os.environ["MAMBA_ROOT_PREFIX"])


# ------------------------------------------------------------------ locating the environment
def find_conda():
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if exe:
        return exe
    for prefix in _CANDIDATE_PREFIXES:
        p = Path(os.path.expanduser(prefix)) / "bin" / "conda"
        if p.is_file():
            return str(p)
    return None


def find_env_python(env_name=ENV_NAME):
    """Finds the python of the vot1 environment. Can also be given manually via the VOT1_PYTHON variable."""
    if os.environ.get("VOT1_PYTHON"):
        return os.environ["VOT1_PYTHON"]
    if Path(sys.prefix).name == env_name:
        return sys.executable
    conda = find_conda()
    if conda:
        try:
            out = subprocess.check_output([conda, "env", "list", "--json"], text=True, stderr=subprocess.DEVNULL)
            for env in json.loads(out).get("envs", []):
                if Path(env).name == env_name and (Path(env) / "bin" / "python").is_file():
                    return str(Path(env) / "bin" / "python")
        except Exception:  # noqa: BLE001
            pass
    for prefix in _CANDIDATE_PREFIXES:
        p = Path(os.path.expanduser(prefix)) / "envs" / env_name / "bin" / "python"
        if p.is_file():
            return str(p)
    return None


def in_colab() -> bool:
    return "google.colab" in sys.modules


def colab_bootstrap(**kwargs) -> dict:
    """Prepares a Google Colab session (environment, checkpoints, dataset, paths). See colab_setup.bootstrap."""
    import importlib
    import colab_setup
    info = importlib.reload(colab_setup).bootstrap(**kwargs)  # reload: the code may have changed (git pull)
    _PATHS_CACHE.clear()
    _COLAB_INFO.update(info)
    return info


def python():
    py = find_env_python()
    if py is None:
        raise RuntimeError(f"Conda environment '{ENV_NAME}' not found. Run the environment cell of 00_setup.ipynb "
                           "or set the VOT1_PYTHON environment variable to its python path.")
    return py


# ------------------------------------------------------------------ subprocesses
RESUME_NOTE = "Running the same experiment again skips the completed sequences."


def stream(cmd, cwd=REPO_ROOT, env=None, check=True, stop_note=RESUME_NOTE):
    """Runs a command and streams its output live into the notebook. If the cell is interrupted, the whole
    process group is terminated."""
    full_env = os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"
    full_env["MPLBACKEND"] = "Agg"  # Jupyter's inline backend may not exist in the subprocess
    full_env.setdefault("CONDA_PLUGINS_AUTO_ACCEPT_TOS", "yes")
    if env:
        full_env.update(env)
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd), env=full_env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    captured = []
    try:
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            text = chunk.decode("utf-8", errors="replace")
            captured.append(text)
            sys.stdout.write(text)
            sys.stdout.flush()
        rc = proc.wait()
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGTERM)
        time.sleep(2)
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        print(f"\n>>> Stopped. {stop_note}")
        raise
    if check and rc != 0:
        raise RuntimeError(f"Command failed (exit code {rc}). See the error message above.")
    return "".join(captured)


def _cli(*args, check=True, stop_note=RESUME_NOTE):
    return stream([python(), "-m", "stark_ft", *args], check=check, stop_note=stop_note)


def cli(*args, check=True):
    """Runs `python -m stark_ft <args>` in the vot1 environment (output is streamed live)."""
    _cli(*args, check=check)


def paths() -> dict:
    """Paths resolved from configs/paths*.yaml and environment variables (checkpoints, dataset, outputs,
    train_data, train_outputs)."""
    if not _PATHS_CACHE:
        out = subprocess.check_output(
            [python(), "-c", "import json; from stark_ft.paths import get_paths; print(json.dumps(get_paths().as_dict()))"],
            cwd=str(REPO_ROOT), text=True)
        _PATHS_CACHE.update(json.loads(out.strip().splitlines()[-1]))
    return dict(_PATHS_CACHE)


def outputs_dir() -> Path:
    return Path(paths()["outputs"])


# ------------------------------------------------------------------ experiments
def _config_file(params: dict) -> Path:
    config_dir = outputs_dir() / "_notebook_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"params_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def describe(params: dict) -> dict:
    """Validates the parameters; shows the experiment name, output folder, checkpoint and sequences."""
    path = _config_file(params)
    cli("show", "--config", path, check=False)
    out = subprocess.run([python(), "-m", "stark_ft", "show", "--config", str(path), "--json"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {}


def run(params: dict, overwrite=False, analyze=True) -> Path:
    """Runs the experiment and returns its output folder."""
    path = _config_file(params)
    args = ["run", "--config", path]
    if overwrite:
        args.append("--overwrite")
    if not analyze:
        args.append("--no-analyze")
    text = _cli(*args)
    for line in reversed(text.splitlines()):
        if line.startswith(OUTPUT_DIR_MARKER):
            return Path(line[len(OUTPUT_DIR_MARKER):].strip())
    raise RuntimeError("Output folder not found (check the output above).")


def run_many(param_list, overwrite=False, analyze=True):
    """Runs several experiments one after another; if one fails, the others still run."""
    results = {}
    for i, params in enumerate(param_list, 1):
        print(f"\n{'#' * 80}\n# [{i}/{len(param_list)}] {params}\n{'#' * 80}")
        try:
            results[i] = run(params, overwrite=overwrite, analyze=analyze)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {e}")
            results[i] = e
    return results


def analyze(out_dir, score_thr=None, iou_thr=None, thr_resolution=None, plots=True):
    args = ["analyze", out_dir]
    if score_thr is not None:
        args += ["--score-thr", score_thr]
    if iou_thr is not None:
        args += ["--iou-thr", iou_thr]
    if thr_resolution is not None:
        args += ["--thr-resolution", thr_resolution]
    if not plots:
        args.append("--no-plots")
    cli(*args)


# ------------------------------------------------------------------ display
def show_summary(out_dir):
    path = Path(out_dir) / "metrics" / "summary.txt"
    print(path.read_text(encoding="utf-8") if path.is_file() else f"No summary: {path}")


def show_f_curve(out_dir):
    """F-max threshold plot (P / R / F vs. threshold and the PR curve)."""
    from IPython.display import Image, display
    img = Path(out_dir) / "metrics" / "f_curve.png"
    if img.is_file():
        display(Image(filename=str(img)))
    else:
        print(f"No plot: {img}")


def show_plots(out_dir, sequences=None, kinds=("iou_conf", "finetune_loss"), max_sequences=5):
    from IPython.display import Image, Markdown, display
    plot_root = Path(out_dir) / "plots"
    seqs = sequences or sorted(p.name for p in plot_root.iterdir() if p.is_dir())[:max_sequences]
    for seq in seqs:
        display(Markdown(f"#### {seq}"))
        for kind in kinds:
            img = plot_root / seq / f"{kind}.png"
            if img.is_file():
                display(Image(filename=str(img)))


def list_outputs():
    root = outputs_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "metrics" / "metrics.json").is_file())


def compare(names, excel_name=None, plot_name=None):
    from IPython.display import Image, display
    out_dir = outputs_dir() / "_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = ["compare", *names]
    if excel_name:
        args += ["--out", out_dir / excel_name]
    if plot_name:
        args += ["--plot", out_dir / plot_name]
    cli(*args)
    if plot_name and (out_dir / plot_name).is_file():
        display(Image(filename=str(out_dir / plot_name)))


def show_side_by_side(names, sequence, kind="iou_conf"):
    """Shows the same sequence's plot for several experiments, one below the other."""
    from IPython.display import Image, Markdown, display
    for n in names:
        img = outputs_dir() / n / "plots" / sequence / f"{kind}.png"
        display(Markdown(f"**{n}**"))
        if img.is_file():
            display(Image(filename=str(img)))
        else:
            print(f"  (missing: {img})")


# ------------------------------------------------------------------ training
TRAIN_RESUME_NOTE = "Running the same training again resumes from the last finished epoch."


def _train_config_file(params: dict) -> Path:
    config_dir = Path(paths()["train_outputs"]) / "_notebook_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def training_datasets(params: dict):
    """Datasets a training run reads (training + validation)."""
    return sorted(set(params.get("datasets", ["got10k"])) | set(params.get("val_datasets", ["got10k"])))


def prepare_training_data(datasets, got10k_urls=(), archives=None, delete_archives=False):
    """Downloads / extracts the training datasets ("got10k", "coco") into `train_data`. On Colab the archives are
    kept on Google Drive (<drive_root>/train_archives), so later sessions only extract them. got10k_urls: download
    links (Google Drive share links work) or paths of GOT-10k archives / folders (used in place)."""
    archives = archives or _COLAB_INFO.get("train_archives")
    args = ["prepare-train-data", "--datasets", *datasets]
    if archives:
        args += ["--archives", archives]
    for url in got10k_urls:
        args += ["--got10k-url", url]
    if delete_archives:
        args.append("--delete-archives")
    cli(*args)


def setup_training(params: dict, got10k_urls=(), **bootstrap_kwargs) -> dict:
    """Everything a training run needs: on Colab the session (environment, checkpoint, paths; no VOT dataset),
    then the training datasets. Returns the dry-run information of the run."""
    if in_colab():
        official = params.get("stage", 2) == 2 and params.get("init") in (None, "official")
        ckpts = (("stark_st", params.get("model_config", "baseline_R101")),) if official else ()
        bootstrap_kwargs.setdefault("vot_dataset", False)
        colab_bootstrap(checkpoints=ckpts, **bootstrap_kwargs)
    prepare_training_data(training_datasets(params), got10k_urls)
    return describe_training(params)


def describe_training(params: dict) -> dict:
    """Validates the training parameters and shows what the run would do (folder, epochs, datasets, init)."""
    cli("train", "--config", _train_config_file(params), "--dry-run", check=False)
    try:
        return _training_info(params)
    except RuntimeError:
        return {}


def _training_info(params: dict) -> dict:
    out = subprocess.run([python(), "-m", "stark_ft", "train", "--config", str(_train_config_file(params)),
                          "--json"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        raise RuntimeError(f"Invalid training parameters:\n{out.stderr[-2000:]}") from None


def train(params: dict, overwrite=False) -> Path:
    """Trains (or resumes) the run and returns its folder."""
    args = ["train", "--config", _train_config_file(params)]
    if overwrite:
        args.append("--overwrite")
    text = _cli(*args, stop_note=TRAIN_RESUME_NOTE)
    for line in reversed(text.splitlines()):
        if line.startswith(OUTPUT_DIR_MARKER):
            return Path(line[len(OUTPUT_DIR_MARKER):].strip())
    raise RuntimeError("Run folder not found (check the output above).")


def show_training(run):
    """Progress, last losses and the history plot of a training run (run name, folder or the params dict)."""
    from IPython.display import Image, display
    if isinstance(run, dict):
        run = _training_info(run)["run_name"]
    cli("train-report", run)
    img = Path(paths()["train_outputs"]) / str(run) / "history.png"
    img = img if img.is_file() else Path(str(run)) / "history.png"
    if img.is_file():
        display(Image(filename=str(img)))


def list_training_runs():
    cli("train-list")
