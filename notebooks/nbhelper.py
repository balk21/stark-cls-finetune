"""
Notebook helpers (standard library + IPython only).

The notebooks run in any Python kernel. The work is done in subprocesses (`python -m stark_ft ...`) with the Python of
the `vot1` conda environment. On Google Colab, init() first restores that environment from Google Drive
(colab_setup.py), and the data helpers keep their downloads on Drive.
"""
import importlib
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
ENV_FILE = REPO_ROOT / "environment" / "vot1_environment.yml"
OUTPUT_DIR_MARKER = "OUTPUT_DIR="  # printed by `python -m stark_ft test / train`
_PATHS_CACHE = {}
_COLAB = {}  # Colab session info (colab_setup.session), empty elsewhere

_CANDIDATE_PREFIXES = [
    "~/anaconda3", "~/miniconda3", "~/miniforge3", "~/mambaforge", "~/micromamba",
    "/opt/conda", "/opt/miniconda3", "/opt/miniforge3", "/usr/local/miniconda3", "/root/miniconda3",
    "/content/micromamba",  # Google Colab (colab_setup.py)
]
if os.environ.get("MAMBA_ROOT_PREFIX"):
    _CANDIDATE_PREFIXES.insert(0, os.environ["MAMBA_ROOT_PREFIX"])


# ------------------------------------------------------------------ environment
def in_colab() -> bool:
    return "google.colab" in sys.modules


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
    """The python of the vot1 environment (can also be given with the VOT1_PYTHON environment variable)."""
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


def python():
    py = find_env_python()
    if py is None:
        raise RuntimeError(f"Conda environment '{ENV_NAME}' not found. Run nb.init() (it creates it), or set the "
                           "VOT1_PYTHON environment variable to its python.")
    return py


def create_environment():
    conda = find_conda()
    if conda is None:
        raise RuntimeError("conda not found. Install Miniconda / Miniforge (or use a machine image with conda) and "
                           f"run this cell again, or create the environment yourself from {ENV_FILE}.")
    print(f"Creating the '{ENV_NAME}' environment (once, 10-20 min) ...", flush=True)
    stream([conda, "env", "create", "-n", ENV_NAME, "-f", ENV_FILE])


def _colab():
    import colab_setup
    return importlib.reload(colab_setup)  # reload: the code may have changed (git pull)


def init(colab_drive="/content/drive/MyDrive/LOKAP"):
    """Finds the vot1 environment (creates it if missing). On Colab: mounts Google Drive and restores the
    environment from `colab_drive`; outputs, training runs and downloads are kept in that Drive folder."""
    _PATHS_CACHE.clear()
    _COLAB.clear()
    if in_colab():
        _COLAB.update(_colab().session(drive_root=colab_drive))
    elif find_env_python() is None:
        create_environment()
    gpu = subprocess.run([python(), "-c", "import torch; print(torch.cuda.get_device_name(0) "
                          "if torch.cuda.is_available() else 'NOT AVAILABLE')"], capture_output=True, text=True)
    p = paths()
    print(f"vot1 python   : {python()}\nGPU           : {gpu.stdout.strip() or gpu.stderr.strip()[-300:]}\n"
          f"test outputs  : {p['outputs']}\ntraining runs : {p['train_outputs']}")


# ------------------------------------------------------------------ subprocesses
RESUME_NOTE = "Running the same experiment again skips the completed sequences."
TRAIN_RESUME_NOTE = "Running the same training again resumes from the last finished epoch."


def stream(cmd, cwd=REPO_ROOT, env=None, check=True, stop_note=RESUME_NOTE):
    """Runs a command and streams its output into the notebook. Stopping the cell stops the whole process group."""
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
    """Runs `python -m stark_ft <args>` in the vot1 environment."""
    _cli(*args, check=check)


def _json_cli(*args) -> dict:
    out = subprocess.run([python(), "-m", "stark_ft", *map(str, args), "--json"], cwd=str(REPO_ROOT),
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        raise RuntimeError(f"Invalid parameters:\n{out.stderr[-2000:]}") from None


def paths() -> dict:
    """Paths from configs/paths*.yaml and environment variables (checkpoints, dataset, outputs, train_data,
    train_outputs)."""
    if not _PATHS_CACHE:
        out = subprocess.check_output(
            [python(), "-c", "import json; from stark_ft.paths import get_paths; print(json.dumps(get_paths().as_dict()))"],
            cwd=str(REPO_ROOT), text=True)
        _PATHS_CACHE.update(json.loads(out.strip().splitlines()[-1]))
    return dict(_PATHS_CACHE)


def _params_file(params: dict, folder: Path, prefix: str) -> Path:
    folder = folder / "_notebook_configs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
    path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------ data
def download_checkpoints(model_configs=("baseline_R101",), model="stark_st"):
    """Official STARK checkpoints (skipped if present). On Colab they are kept on Google Drive."""
    if _COLAB:
        _colab().download_checkpoints(Path(_COLAB["drive_root"]), Path(_COLAB["local_root"]), model, model_configs)
    else:
        cli("download-checkpoints", "--model", model, "--model-config", *model_configs)


def download_vot_dataset(sequences="all"):
    """VOT-LT2020 sequences in advance ("all": 17.6 GB, or a list). Optional: a test downloads the sequences it uses.
    Sequences already present are skipped; on Colab a copy of each is kept on Google Drive."""
    cli("download-dataset", "--sequences", *(["all"] if sequences == "all" else sequences))


def training_datasets(params: dict):
    """Datasets a training run reads (training + validation)."""
    return sorted(set(params.get("datasets", ["got10k"])) | set(params.get("val_datasets", ["got10k"])))


def prepare_training_data(datasets, got10k_sources=(), delete_archives=False):
    """Downloads / extracts the training datasets ("got10k", "coco"); `datasets` may also be the training
    parameters. got10k_sources: GOT-10k download links or archive paths. On Colab the archives are kept on Drive."""
    if isinstance(datasets, dict):
        datasets = training_datasets(datasets)
    args = ["prepare-train-data", "--datasets", *datasets]
    if _COLAB:
        args += ["--archives", _COLAB["train_archives"]]
    for src in got10k_sources:
        args += ["--got10k-url", src]
    if delete_archives:
        args.append("--delete-archives")
    cli(*args)


# ------------------------------------------------------------------ test
def outputs_dir() -> Path:
    return Path(paths()["outputs"])


def describe(params: dict) -> dict:
    """Checks the test parameters; shows the experiment name, output folder, checkpoint and sequences."""
    path = _params_file(params, outputs_dir(), "params")
    cli("show", "--config", path, check=False)
    try:
        return _json_cli("show", "--config", path)
    except RuntimeError:
        return {}


def test(params: dict, overwrite=False, analyze=True) -> Path:
    """Runs a test experiment (vot-toolkit + analysis) and returns its output folder."""
    args = ["test", "--config", _params_file(params, outputs_dir(), "params")]
    if overwrite:
        args.append("--overwrite")
    if not analyze:
        args.append("--no-analyze")
    text = _cli(*args)
    for line in reversed(text.splitlines()):
        if line.startswith(OUTPUT_DIR_MARKER):
            return Path(line[len(OUTPUT_DIR_MARKER):].strip())
    raise RuntimeError("Output folder not found (see the output above).")


def test_many(param_list, overwrite=False, analyze=True):
    """Runs several experiments one after another; a failing one does not stop the others."""
    results = {}
    for i, params in enumerate(param_list, 1):
        print(f"\n{'#' * 80}\n# [{i}/{len(param_list)}] {params}\n{'#' * 80}")
        try:
            results[i] = test(params, overwrite=overwrite, analyze=analyze)
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


def show_summary(out_dir):
    path = Path(out_dir) / "metrics" / "summary.txt"
    print(path.read_text(encoding="utf-8") if path.is_file() else f"No summary: {path}")


def show_f_curve(out_dir):
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
    """The same sequence's plot for several experiments, one below the other."""
    from IPython.display import Image, Markdown, display
    for n in names:
        img = outputs_dir() / n / "plots" / sequence / f"{kind}.png"
        display(Markdown(f"**{n}**"))
        if img.is_file():
            display(Image(filename=str(img)))
        else:
            print(f"  (missing: {img})")


# ------------------------------------------------------------------ train
def _train_file(params: dict) -> Path:
    return _params_file(params, Path(paths()["train_outputs"]), "train")


def describe_training(params: dict) -> dict:
    """Checks the training parameters and data; shows the run folder, epochs, steps and initial weights."""
    path = _train_file(params)
    cli("train", "--config", path, "--dry-run", check=False)
    try:
        return _json_cli("train", "--config", path)
    except RuntimeError:
        return {}


def train(params: dict, overwrite=False) -> Path:
    """Trains (or resumes) and returns the run folder."""
    args = ["train", "--config", _train_file(params)]
    if overwrite:
        args.append("--overwrite")
    text = _cli(*args, stop_note=TRAIN_RESUME_NOTE)
    for line in reversed(text.splitlines()):
        if line.startswith(OUTPUT_DIR_MARKER):
            return Path(line[len(OUTPUT_DIR_MARKER):].strip())
    raise RuntimeError("Run folder not found (see the output above).")


def show_training(run):
    """Progress, last losses and the history plot of a run (run name, run folder or the training parameters)."""
    from IPython.display import Image, display
    if isinstance(run, dict):
        run = _json_cli("train", "--config", _train_file(run))["run_name"]
    cli("train-report", run)
    img = Path(paths()["train_outputs"]) / str(run) / "history.png"
    img = img if img.is_file() else Path(str(run)) / "history.png"
    if img.is_file():
        display(Image(filename=str(img)))


def list_training_runs():
    cli("train-list")
