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
    from colab_setup import bootstrap
    info = bootstrap(**kwargs)
    _PATHS_CACHE.clear()
    return info


def python():
    py = find_env_python()
    if py is None:
        raise RuntimeError(f"Conda environment '{ENV_NAME}' not found. Run the environment cell of 00_setup.ipynb "
                           "or set the VOT1_PYTHON environment variable to its python path.")
    return py


# ------------------------------------------------------------------ subprocesses
def stream(cmd, cwd=REPO_ROOT, env=None, check=True):
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
        print("\n>>> Stopped. Running the same experiment again skips the completed sequences.")
        raise
    if check and rc != 0:
        raise RuntimeError(f"Command failed (exit code {rc}). See the error message above.")
    return "".join(captured)


def _cli(*args, check=True):
    return stream([python(), "-m", "stark_ft", *args], check=check)


def cli(*args, check=True):
    """Runs `python -m stark_ft <args>` in the vot1 environment (output is streamed live)."""
    _cli(*args, check=check)


def paths() -> dict:
    """Paths resolved from configs/paths*.yaml and environment variables (checkpoints, dataset, outputs)."""
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
