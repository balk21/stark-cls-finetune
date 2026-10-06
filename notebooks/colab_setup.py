"""
Google Colab support (standard library only). Used by nbhelper.init() / the data helpers on Colab.

Colab gives a new machine in every session. session() mounts Google Drive and restores the vot1 environment from a
Drive cache (creating it the first time), so that the notebooks work as on any other machine. Downloads (checkpoints,
VOT sequences, training archives) are kept on Drive, so later sessions only copy / extract them.

Layout (<drive_root>: default MyDrive/LOKAP, set in the first cell of the notebooks):
  <drive_root>/cache/vot1_env_<hash>.tar        the conda environment (~7.8 GB)
  <drive_root>/cache/votlt2019_sequences/       one tar per VOT-LT2020 sequence used (dataset_cache; all 50: 17.6 GB)
  <drive_root>/checkpoints/                     official STARK weights / own checkpoint files (as checkpoints/)
  <drive_root>/train_archives/{coco,got10k}/    training dataset archives
  <drive_root>/outputs/                         test outputs (persist, so experiments can resume)
  <drive_root>/training/                        training runs (persist, so training can resume)
  <local_root>/...                              environment and data used in the session (local disk)
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
DRIVE_ROOT = "/content/drive/MyDrive/LOKAP"
LOCAL_ROOT = "/content"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"


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
        raise RuntimeError(f"The environment is incomplete ({python} is missing). "
                           "Run colab_setup.session(rebuild_env=True).")
    _log(f"Environment ready: {python}")
    return python


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


def session(drive_root=DRIVE_ROOT, local_root=LOCAL_ROOT, mount=True, rebuild_env=False) -> dict:
    """Prepares the Colab session: Google Drive, the vot1 environment and the paths."""
    t0 = time.time()
    if in_colab() and not shutil.which("nvidia-smi"):
        print("WARNING: no GPU in this session. Choose Runtime -> Change runtime type -> GPU (e.g. A100 or L4), "
              "then run this cell again.")
    if mount and not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive  # noqa: import only available on Colab
        drive.mount("/content/drive")
    drive_root, local_root = Path(drive_root), Path(local_root)
    _log(f"Google Drive folder: {drive_root}")
    (drive_root / "outputs").mkdir(parents=True, exist_ok=True)
    python = setup_environment(drive_root, local_root, rebuild=rebuild_env)
    gpu_ok = check_gpu(python)
    paths = {"checkpoints": drive_root / "checkpoints",                     # official weights: downloaded here once
             "dataset": local_root / "data" / "votlt2020" / "sequences",       # VOT sequences: local disk
             "dataset_cache": drive_root / "cache" / "votlt2019_sequences",    # copies of the sequences used: Drive
             "outputs": drive_root / "outputs",
             "train_data": local_root / "train_data",
             "train_outputs": drive_root / "training"}
    write_paths(**paths)
    os.environ["VOT1_PYTHON"] = str(python)  # used by nbhelper.python()
    _log(f"Session ready in {time.time() - t0:.0f} s")
    return {"drive_root": str(drive_root), "local_root": str(local_root), "python": str(python), "gpu": gpu_ok,
            "train_archives": str(drive_root / "train_archives"), **{k: str(v) for k, v in paths.items()}}
