"""
Google Colab bootstrap — uses ONLY the standard library.

Colab sessions are ephemeral, so every session has to restore the environment and the checkpoints.
`bootstrap()` is idempotent: it only does what is missing in the current session.

  First session (slow, once):
    - installs micromamba and creates the `vot1` environment from environment/vot1_environment.yml
      (exactly the same packages as on Vast.ai / a local conda machine), then caches it on Google Drive;
    - downloads the official checkpoint(s) to Drive.
  Later sessions (a few minutes):
    - restores the environment from the Drive cache to the local disk.
  VOT-LT2020 sequences are NOT downloaded here: every experiment downloads only the sequences it uses
  (stark_ft/vot_data.py) and keeps a copy of each on Drive (<drive_root>/cache/votlt2019_sequences/<sequence>.tar),
  so later sessions restore them from Drive. bootstrap(vot_sequences="all" or a list) fetches them up front instead.
  Training datasets (03_train.ipynb) are prepared separately with `python -m stark_ft prepare-train-data`; their
  archives are kept on Drive (<drive_root>/train_archives) and extracted to the local disk in every session.

Layout:
  <drive_root>/cache/vot1_env_<hash>.tar        cached conda environment (~7.5 GB)
  <drive_root>/cache/votlt2019_sequences/       one tar per downloaded VOT sequence (all 50: 17.6 GB)
  <drive_root>/checkpoints/...                  checkpoints (same layout as checkpoints/ in the repository)
  <drive_root>/outputs/                         experiment outputs (persist across sessions, so runs can resume)
  <drive_root>/train_archives/{coco,got10k}/    training dataset archives (COCO ~19.6 GB; GOT-10k: see README)
  <drive_root>/training/                        training runs (persist across sessions, so training can resume)
  <local_root>/micromamba/envs/vot1/            the environment (local disk)
  <local_root>/data/votlt2020/sequences/        the VOT sequences (local disk; reading images from Drive is too slow)
  <local_root>/checkpoints/                     local copy of the Drive checkpoints
  <local_root>/train_data/                      extracted training datasets (local disk)
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "environment" / "vot1_environment.yml"
DRIVE_ROOT = "/content/drive/MyDrive/LOKAP"  # the notebooks pass their DRIVE_ROOT (first cell)
LOCAL_ROOT = "/content"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
DEFAULT_CHECKPOINTS = (("stark_st", "baseline_R101"),)


def in_colab() -> bool:
    return "google.colab" in sys.modules


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(cmd, env=None, cwd=None, quiet=False):
    """Runs a command and streams its output (stdlib only). quiet=True: the output is only shown if it fails."""
    full_env = os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"
    full_env["MPLBACKEND"] = "Agg"
    if env:
        full_env.update(env)
    proc = subprocess.Popen([str(c) for c in cmd], env=full_env, cwd=str(cwd or REPO_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    captured = []
    for chunk in iter(lambda: proc.stdout.read1(4096), b""):
        text = chunk.decode("utf-8", errors="replace")
        if quiet:
            captured.append(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
    if proc.wait() != 0:
        if quiet:
            sys.stdout.write("".join(captured)[-20000:])
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")


def _tar_create(src_parent: Path, member: str, dst: Path):
    """Writes <src_parent>/<member> into the tar file dst (via a temporary file, so a half-written cache
    is never used)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    _run(["tar", "-cf", tmp, "-C", src_parent, member])
    os.replace(tmp, dst)


def _tar_extract(src: Path, dst_parent: Path):
    dst_parent.mkdir(parents=True, exist_ok=True)
    _run(["tar", "-xf", src, "-C", dst_parent])


def _env_hash() -> str:
    return hashlib.sha1(ENV_FILE.read_bytes()).hexdigest()[:10]


def _cuda_version_for_solver():
    """The CUDA version reported by the driver (for micromamba's __cuda virtual package)."""
    if os.environ.get("CONDA_OVERRIDE_CUDA"):
        return os.environ["CONDA_OVERRIDE_CUDA"]
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
    if exe:
        try:
            m = re.search(r"CUDA Version:\s*([\d.]+)", subprocess.check_output([exe], text=True))
            if m:
                return m.group(1)
        except Exception:  # noqa: BLE001
            pass
    return None


def _install_micromamba(mamba_root: Path) -> Path:
    exe = mamba_root / "bin" / "micromamba"
    if exe.is_file():
        return exe
    _log("Downloading micromamba ...")
    mamba_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "micromamba.tar.bz2"
        urllib.request.urlretrieve(MICROMAMBA_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extract("bin/micromamba", path=mamba_root)
    exe.chmod(0o755)
    return exe


def setup_environment(drive_root: Path, local_root: Path, rebuild=False) -> Path:
    """Returns the python of the vot1 environment; restores it from Drive or creates it."""
    mamba_root = local_root / "micromamba"
    env_dir = mamba_root / "envs" / "vot1"
    python = env_dir / "bin" / "python"
    cache = drive_root / "cache" / f"vot1_env_{_env_hash()}.tar"
    if not (python.is_file() and not rebuild):
        if cache.is_file() and not rebuild:
            _log(f"Restoring the environment from Drive ({cache.stat().st_size / 1e9:.1f} GB) ...")
            _tar_extract(cache, mamba_root / "envs")  # the cache contains the folder "vot1/"
        else:
            _log("Creating the vot1 environment (first time only, ~5-10 min) ...")
            exe = _install_micromamba(mamba_root)
            env = {"MAMBA_ROOT_PREFIX": str(mamba_root)}
            cuda = _cuda_version_for_solver()
            if cuda:
                env["CONDA_OVERRIDE_CUDA"] = cuda
            if env_dir.exists():
                shutil.rmtree(env_dir)
            _run([exe, "create", "-y", "-q", "-n", "vot1", "-f", ENV_FILE], env=env, quiet=True)
            _run([exe, "clean", "-y", "-a", "-q"], env=env, quiet=True)  # drop the package cache (~3 GB)
    if not cache.is_file():
        # Also covers a session where the environment was created but caching was interrupted
        _log(f"Caching the environment on Drive: {cache} ...")
        for old in cache.parent.glob("vot1_env_*.tar"):
            old.unlink()  # caches of older environment files
        _tar_create(mamba_root / "envs", "vot1", cache)
    if not python.is_file():
        raise RuntimeError(f"The environment is incomplete ({python} is missing). Run bootstrap(rebuild_env=True).")
    _log(f"Environment ready: {python}")
    return python


def setup_checkpoints(drive_root: Path, local_root: Path, checkpoints):
    """Downloads missing official checkpoints to Drive and copies every Drive checkpoint to the local disk."""
    drive_ckpt = drive_root / "checkpoints"
    local_ckpt = local_root / "checkpoints"
    for model in sorted({m for m, _ in checkpoints}):
        cfgs = [c for m, c in checkpoints if m == model]
        # Runs with the notebook's own Python: it only needs PyYAML and gdown (both preinstalled on Colab)
        _run([sys.executable, "-m", "stark_ft", "download-checkpoints", "--model", model, "--model-config", *cfgs],
             env={"STARK_CLEAN_CHECKPOINTS": str(drive_ckpt)})
    copied = 0
    for src in drive_ckpt.rglob("*.pth.tar"):
        dst = local_ckpt / src.relative_to(drive_ckpt)
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    _log(f"Checkpoints ready: {local_ckpt} ({copied} copied from Drive)")
    return local_ckpt


def setup_dataset(drive_root: Path, local_root: Path):
    """The VOT sequence folder (local disk) and its cache on Drive. Nothing is downloaded here."""
    seq_dir = local_root / "data" / "votlt2020" / "sequences"
    cache = drive_root / "cache" / "votlt2019_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    cached = len(list(cache.glob("*.tar"))) if cache.is_dir() else 0
    _log(f"VOT sequences: fetched when an experiment needs them ({cached} of 50 cached on Drive in {cache})")
    old = drive_root / "cache" / "votlt2020_sequences.tar"
    if old.is_file():
        _log(f"Note: {old} ({old.stat().st_size / 1e9:.1f} GB) is no longer used; you can delete it.")
    return seq_dir, cache


def check_gpu(python: Path) -> bool:
    """Checks that PyTorch in the vot1 environment can use the GPU."""
    code = ("import torch; ok = torch.cuda.is_available(); "
            "print('GPU in vot1:', torch.cuda.get_device_name(0) if ok else 'NOT AVAILABLE')")
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip()[-500:])
    ok = "NOT AVAILABLE" not in result.stdout and result.returncode == 0
    if not ok:
        print("WARNING: PyTorch in the vot1 environment cannot use the GPU. Check that a GPU runtime is selected "
              "(Runtime -> Change runtime type -> GPU) and run this cell again.")
    return ok


def write_paths(**paths):
    """Writes configs/paths.local.yaml (keys of stark_ft/paths.py; None values are left out)."""
    path = REPO_ROOT / "configs" / "paths.local.yaml"
    lines = ["# Written by notebooks/colab_setup.py (Google Colab session)"]
    lines += [f"{k}: {v}" for k, v in paths.items() if v is not None]
    path.write_text("\n".join(lines) + "\n")
    return path


def bootstrap(drive_root=DRIVE_ROOT, local_root=LOCAL_ROOT, mount=None, checkpoints=DEFAULT_CHECKPOINTS,
              rebuild_env=False, vot_sequences=()) -> dict:
    """Prepares the current Colab session. Safe to call in every notebook; only missing steps are executed.
    vot_sequences: VOT sequences to fetch now ("all" or a list of names); by default none, since every experiment
    fetches the sequences it uses."""
    t0 = time.time()
    if mount is None:
        mount = in_colab()
    if in_colab() and not shutil.which("nvidia-smi"):
        print("WARNING: no GPU in this session. Choose Runtime -> Change runtime type -> GPU (T4, L4 or A100), "
              "then run this cell again.")
    if mount and not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive  # noqa: import only available on Colab
        drive.mount("/content/drive")
    drive_root, local_root = Path(drive_root), Path(local_root)
    _log(f"Google Drive folder: {drive_root}")
    (drive_root / "outputs").mkdir(parents=True, exist_ok=True)

    python = setup_environment(drive_root, local_root, rebuild=rebuild_env)
    gpu_ok = check_gpu(python)
    ckpt = setup_checkpoints(drive_root, local_root, checkpoints)
    dataset, dataset_cache = setup_dataset(drive_root, local_root)
    info = {"checkpoints": ckpt, "dataset": dataset, "dataset_cache": dataset_cache, "outputs": drive_root / "outputs",
            "train_data": local_root / "train_data", "train_outputs": drive_root / "training"}
    paths_file = write_paths(**info)
    if vot_sequences:
        names = ["all"] if vot_sequences == "all" else list(vot_sequences)
        _run([python, "-m", "stark_ft", "download-dataset", "--sequences", *names])
    os.environ["VOT1_PYTHON"] = str(python)  # used by nbhelper.python()
    _log(f"Session ready in {time.time() - t0:.0f} s. Paths: {paths_file}")
    info = {k: (str(v) if v is not None else None) for k, v in info.items()}
    info.update(python=str(python), gpu=gpu_ok, train_archives=str(drive_root / "train_archives"))
    return info
