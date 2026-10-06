"""
Progress of training runs: status, last losses and the history plot (history.png).
"""
import json
from pathlib import Path

import numpy as np

from stark_ft.paths import Paths, get_paths
from stark_ft.train.config import TrainConfig


def find_run(name_or_path, paths: Paths = None) -> Path:
    """A run folder given by its name (in train_outputs) or its path."""
    p = Path(str(name_or_path))
    if (p / "train_config.json").is_file():
        return p
    run_dir = (paths or get_paths()).train_outputs / str(name_or_path)
    if not (run_dir / "train_config.json").is_file():
        raise FileNotFoundError(f"No training run {str(name_or_path)!r} (looked for {run_dir}/train_config.json)")
    return run_dir


def _history(run_dir: Path):
    import pandas as pd
    path = run_dir / "history.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    h = pd.read_csv(path)
    return h.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def run_status(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "train_config.json").read_text())
    tc = TrainConfig.from_dict(meta["config"])
    status = {"run_name": meta.get("run_name", run_dir.name), "run_dir": str(run_dir), "stage": tc.stage,
              "model_config": tc.model_config, "datasets": tc.datasets, "init": meta.get("init"),
              "gpu": meta.get("gpu"), "total_epochs": tc.total_epochs, "epochs_done": 0,
              "finished": (run_dir / "final.pth.tar").is_file(), "last": {}}
    h = _history(run_dir)
    if h is not None and len(h):
        done = int(h["epoch"].iloc[-1])
        epoch_s = float(h["seconds"].tail(10).mean())
        status.update(epochs_done=done, epoch_seconds=epoch_s,
                      remaining_hours=max(0, status["total_epochs"] - done) * epoch_s / 3600)
        for col in h.columns:
            if "/" in col:
                values = h[col].dropna()
                if len(values):
                    status["last"][col] = {"epoch": int(h.loc[values.index[-1], "epoch"]),
                                           "value": float(values.iloc[-1])}
    return status


def report(name_or_path, plot=True, paths: Paths = None) -> dict:
    """Status of a training run; writes history.png into the run folder."""
    run_dir = find_run(name_or_path, paths)
    status = run_status(run_dir)
    h = _history(run_dir)
    if plot and h is not None and len(h):
        plot_training_history(h, run_dir / "history.png", status["run_name"])
        status["plot"] = str(run_dir / "history.png")
    return status


def list_runs(paths: Paths = None):
    root = (paths or get_paths()).train_outputs
    if not root.is_dir():
        return []
    return [run_status(p) for p in sorted(root.iterdir()) if (p / "train_config.json").is_file()]


def plot_training_history(history, out_path, run_name: str):
    """Per-epoch training / validation statistics of a training run (its history.csv). Dashed lines: LR drops."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    epoch = history["epoch"].to_numpy(dtype=float)
    metrics = []
    for col in history.columns:
        split, _, metric = col.partition("/")
        if split in ("train", "val") and metric and metric not in metrics:
            metrics.append(metric)
    cols = 2 if len(metrics) > 1 else 1
    rows = max(1, int(np.ceil(len(metrics) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.6 * rows), squeeze=False)
    lr = history["lr"].to_numpy(dtype=float)
    drops = epoch[1:][lr[1:] != lr[:-1]] if len(lr) > 1 else []
    for ax, metric in zip(axes.flat, metrics):
        for split, style in (("train", dict(color="tab:blue")),
                             ("val", dict(color="tab:orange", marker="o", markersize=3))):
            col = f"{split}/{metric}"
            if col in history.columns:
                y = history[col].to_numpy(dtype=float)
                ok = np.isfinite(y)
                if ok.any():
                    ax.plot(epoch[ok], y[ok], label=split, **style)
        for d in drops:
            ax.axvline(d - 0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize="small")
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Training history — {run_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
