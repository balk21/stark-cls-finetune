"""
Plots. All functions save to a file and close the figure (notebooks display them with
IPython.display.Image).
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def plot_iou_conf(frames: pd.DataFrame, events: pd.DataFrame, out_path, seq: str, exp_name: str,
                  score_thr: float = None, opt_thr: float = None):
    """IoU and confidence score vs. frame number.

    frames: columns frame, conf, iou, gt_visible (frame 0 = init frame)
    events: columns frame, event ('template_update' | 'ft_update')
    """
    df = frames[frames["frame"] > 0]
    x = df["frame"].to_numpy()
    absent = ~df["gt_visible"].to_numpy(dtype=bool)
    iou_vals = df["iou"].to_numpy(dtype=float)
    conf = df["conf"].to_numpy(dtype=float)
    vis_iou = np.nan_to_num(iou_vals[~absent], nan=0.0)
    mean_iou = float(vis_iou.mean()) if len(vis_iou) else float("nan")

    upd = events.loc[events["event"] == "template_update", "frame"].to_numpy() if len(events) else []
    ftu = events.loc[events["event"] == "ft_update", "frame"].to_numpy() if len(events) else []

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x, -0.05, 1.05, where=absent, color="gray", alpha=0.2, hatch="///", step="mid", linewidth=0)
    ax.plot(x, conf, color="#1f77b4", linewidth=1.5, alpha=0.9)
    ax.plot(x, iou_vals, color="#ff7f0e", linewidth=1.5, alpha=0.9)
    if len(upd):
        ax.vlines(upd, 0, 1, colors="red", linestyles="--", linewidth=1.2, alpha=0.7)
    if len(ftu):
        ax.vlines(ftu, 0, 1, colors="green", linestyles=":", linewidth=1.5, alpha=0.8)
    if score_thr is not None:
        ax.axhline(score_thr, color="black", linestyle="-.", linewidth=0.8, alpha=0.6)
    if opt_thr is not None and np.isfinite(opt_thr):
        ax.axhline(opt_thr, color="purple", linestyle="-", linewidth=1.0, alpha=0.7)

    ax.set_title(f"Sequence: {seq}  |  Mean IoU (visible frames): {mean_iou:.3f}  |  "
                 f"Template updates: {len(upd)}  |  Fine-tuning updates: {len(ftu)}\n{exp_name}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Frame number (0 = init frame)")
    ax.set_ylabel("Value (0-1)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(1, int(frames["frame"].max())))
    ax.grid(True, linestyle="--", alpha=0.5)

    handles = [mlines.Line2D([], [], color="#1f77b4", linewidth=2, label="Confidence score"),
               mlines.Line2D([], [], color="#ff7f0e", linewidth=2, label="IoU"),
               mpatches.Patch(facecolor="gray", alpha=0.3, hatch="///", label="Target absent (GT = NaN)"),
               mlines.Line2D([], [], color="red", linestyle="--", label="Template update")]
    if len(ftu):
        handles.append(mlines.Line2D([], [], color="green", linestyle=":", label="Fine-tuning update"))
    if score_thr is not None:
        handles.append(mlines.Line2D([], [], color="black", linestyle="-.", label=f"Fixed threshold ({score_thr:g})"))
    if opt_thr is not None and np.isfinite(opt_thr):
        handles.append(mlines.Line2D([], [], color="purple", label=f"F-max threshold ({opt_thr:.3f})"))
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, ncol=3, fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_finetune_loss(ft: pd.DataFrame, out_path, seq: str, exp_name: str):
    """Fine-tuning loss and positive/negative sample probabilities.

    x axis: consecutive optimisation steps (all sessions back to back). Vertical lines mark session starts;
    the frame number of the session is written at the top.
    """
    ft = ft.reset_index(drop=True)
    step = np.arange(1, len(ft) + 1)
    loss = ft["loss"].to_numpy(dtype=float)  # matplotlib 3.3 cannot plot pandas Series directly
    pos_prob = ft["pos_prob"].to_numpy(dtype=float)
    neg_prob = ft["neg_prob"].to_numpy(dtype=float)
    session_start = ft.index[(ft["frame"] != ft["frame"].shift()) | (ft["session"] != ft["session"].shift())]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(step, loss, color="tab:blue", marker="." if len(ft) < 300 else None, linewidth=1.2)
    if (loss > 0).all():
        ax1.set_yscale("log")
    ax1.set_ylabel("BCE loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(step, pos_prob, color="tab:green", label="Positive sample probability (target = 1)")
    if np.isfinite(neg_prob).any():
        ax2.plot(step, neg_prob, color="tab:red", label="Negative sample probability (target = 0)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("sigmoid(logit)")
    ax2.set_xlabel("Optimisation step (sessions back to back)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize="small")

    show_labels = len(session_start) <= 25
    for i in session_start:
        for ax in (ax1, ax2):
            ax.axvline(i + 0.5, color="gray", linewidth=0.8, alpha=0.6)
        if show_labels:
            label = "init" if ft.loc[i, "session"] == "init" else f"f{int(ft.loc[i, 'frame'])}"
            ax1.annotate(label, (i + 1, 1.0), xycoords=("data", "axes fraction"), fontsize=7,
                         rotation=90, va="top", color="dimgray")
    n_online = int((ft.loc[session_start, "session"] == "online").sum())
    fig.suptitle(f"Fine-tuning — {seq}  |  init sessions: {int((ft['session'] == 'init').any())}  |  "
                 f"online sessions: {n_online}\n{exp_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_f_curve(curve: dict, best: dict, out_path, exp_name: str, fixed_thr: float = None):
    """VOT-LT style threshold sweep: P / R / F vs. threshold on the left, precision-recall curve on the right."""
    thr = np.asarray(curve["threshold"], dtype=float)
    p, r, f = (np.asarray(curve[k], dtype=float) for k in ("precision", "recall", "F"))
    finite = np.isfinite(thr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(thr[finite], p[finite], color="tab:green", label="Precision")
    ax1.plot(thr[finite], r[finite], color="tab:red", label="Recall")
    ax1.plot(thr[finite], f[finite], color="tab:blue", linewidth=2, label="F")
    if np.isfinite(best["threshold"]):
        ax1.axvline(best["threshold"], color="purple", label=f"F-max threshold = {best['threshold']:.3f}")
    if fixed_thr is not None:
        ax1.axvline(fixed_thr, color="black", linestyle="-.", label=f"Fixed threshold = {fixed_thr:g}")
    ax1.set_xlabel("Score threshold")
    ax1.set_ylabel("Mean over sequences")
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize="small", loc="lower left")

    ax2.plot(r, p, color="tab:blue", marker=".", markersize=3)
    ax2.scatter([best["recall"]], [best["precision"]], color="purple", zorder=3, s=50,
                label=f"F = {best['F']:.3f} (P = {best['precision']:.3f}, R = {best['recall']:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize="small", loc="lower left")
    fig.suptitle(f"F-max threshold (VOT method, IoU >= {best['iou_thr']:g})\n{exp_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, metrics, out_path=None, title="Experiment comparison"):
    """summary: index = experiment name, columns = metrics (output of compare.load_summaries)."""
    metrics = [m for m in metrics if m in summary.columns]
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(summary) * len(metrics) / 2), 5))
    summary[metrics].plot.bar(ax=ax, rot=20)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    for patch in ax.patches:  # matplotlib 3.3 compatible (bar_label needs 3.4+)
        h = patch.get_height()
        if np.isfinite(h):
            ax.annotate(f"{h:.3f}", (patch.get_x() + patch.get_width() / 2, h), ha="center", va="bottom",
                        fontsize=7, rotation=90, xytext=(0, 2), textcoords="offset points")
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(history: pd.DataFrame, out_path, run_name: str):
    """Per-epoch training / validation statistics of a training run (its history.csv). Dashed lines: LR drops."""
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
