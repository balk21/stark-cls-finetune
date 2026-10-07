"""
Produces metric tables and plots from the outputs of an experiment.

Can be called again without re-running the tracking, e.g. with a different threshold:
    analyze_experiment("outputs/<experiment>", score_thr=0.5)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stark_ft.test import datasets
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.evaluation import (coco_standard, evaluate_sequence, f_curve_from_sequences, got10k_ious,
                                      got10k_metrics, iou, operating_point, read_vot_result)
from stark_ft.paths import Paths, get_paths
from stark_ft.test.plots import plot_f_curve, plot_finetune_loss, plot_iou_conf

# Order of the summary tables: the primary metrics (mAP / AP50 / AP75) first
SUMMARY_METRICS = ["mAP", "AP50", "AP75", "precision_opt", "recall_opt", "F_opt", "precision", "recall", "F1",
                   "absent_reject_rate", "mean_iou_visible", "legacy_mAP", "legacy_AP50", "legacy_AP75"]


GOT10K_METRICS = ["AO", "SR50", "SR75"]  # GOT-10k benchmark metrics (GOT-10k datasets only)


def _dataset_dir(meta: dict, cfg: ExperimentConfig, paths: Paths) -> Path:
    """If the analysis runs on another machine the recorded path may not exist; then the current setting is used."""
    recorded = Path(meta.get("dataset", ""))
    return recorded if recorded.is_dir() else datasets.root(cfg.dataset, paths)


def _read_csv(path, columns):
    path = Path(path)
    if path.is_file():
        df = pd.read_csv(path)
        if len(df.columns):
            return df
    return pd.DataFrame(columns=columns)


def build_frame_table(gt_boxes, boxes, scores, times):
    n = len(gt_boxes)
    rows = []
    for i in range(n):
        g = gt_boxes[i]
        p = boxes[i] if i < len(boxes) else None
        rows.append({
            "frame": i,
            "x": p[0] if p else np.nan, "y": p[1] if p else np.nan,
            "w": p[2] if p else np.nan, "h": p[3] if p else np.nan,
            "conf": scores[i] if i < len(scores) else np.nan,
            "time": times[i] if i < len(times) else np.nan,
            "gt_visible": g is not None,
            "gt_x": g[0] if g else np.nan, "gt_y": g[1] if g else np.nan,
            "gt_w": g[2] if g else np.nan, "gt_h": g[3] if g else np.nan,
            "iou": iou(g, p) if (g is not None and i > 0) else np.nan,
        })
    return pd.DataFrame(rows)


def _f_from(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def analyze_experiment(out_dir, paths: Paths = None, score_thr: float = None, iou_thr: float = None,
                       thr_resolution: int = None, plots: bool = True, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    paths = paths or get_paths()
    with open(out_dir / "experiment.json") as f:
        meta = json.load(f)
    cfg = ExperimentConfig.from_dict(meta["config"])
    score_thr = cfg.eval_score_thr if score_thr is None else score_thr
    iou_thr = cfg.eval_iou_thr if iou_thr is None else iou_thr
    thr_resolution = cfg.eval_thr_resolution if thr_resolution is None else thr_resolution
    dataset = _dataset_dir(meta, cfg, paths)
    exp_name = meta["experiment_name"]
    # Ground truth of the sequences with results, e.g. in a new Colab session: restored / downloaded / extracted
    with_results = [s for s in meta["sequences"] if (out_dir / "predictions" / s / f"{s}_001.txt").is_file()]
    datasets.ensure(cfg.dataset, with_results, paths)
    if not datasets.has_ground_truth(cfg.dataset):
        return _no_ground_truth(out_dir, meta, with_results, verbose)
    is_got10k = datasets.is_got10k(cfg.dataset)
    summary_metrics = (GOT10K_METRICS if is_got10k else []) + SUMMARY_METRICS
    got10k_pooled = []

    # ---- pass 1: per-sequence metrics (fixed threshold) ----
    per_seq, all_records, seq_records, plot_data, skipped = [], [], {}, {}, []
    offset = 0
    for seq in meta["sequences"]:
        pred_dir = out_dir / "predictions" / seq
        if not (pred_dir / f"{seq}_001.txt").is_file():
            skipped.append(seq)
            continue
        gt_boxes, image_size = datasets.ground_truth(cfg.dataset, dataset, seq)
        boxes, scores, times = read_vot_result(pred_dir, seq)
        if len(boxes) != len(gt_boxes):
            skipped.append(seq)
            if verbose:
                print(f"  {seq}: incomplete result ({len(boxes)}/{len(gt_boxes)} frames), skipped")
            continue

        frames = build_frame_table(gt_boxes, boxes, scores, times)
        frames.to_csv(pred_dir / "frames.csv", index=False, float_format="%.6f")

        metrics, records = evaluate_sequence(gt_boxes, boxes, scores, score_thr, iou_thr, img_id_offset=offset)
        if is_got10k:
            seq_ious = got10k_ious(gt_boxes, boxes, image_size)
            got10k_pooled.append(seq_ious)
            metrics.update(got10k_metrics(seq_ious))
        offset += len(gt_boxes) + 1
        all_records.extend(records)
        seq_records[seq] = records

        log_dir = out_dir / "tracker_logs" / seq
        events = _read_csv(log_dir / "events.txt", ["frame", "event", "conf_score"])
        ft = _read_csv(log_dir / "finetune_loss.txt", ["frame", "session", "epoch", "loss", "pos_prob", "neg_prob"])
        metrics["template_updates"] = int((events["event"] == "template_update").sum()) if len(events) else 0
        metrics["ft_updates"] = int((events["event"] == "ft_update").sum()) if len(events) else 0
        metrics["mean_time_ms"] = float(np.nanmean(frames["time"].to_numpy()[1:]) * 1000) if len(frames) > 1 else np.nan
        per_seq.append({"sequence": seq, **metrics})
        if plots:
            plot_data[seq] = (frames, events, ft)

        if verbose:
            got = f"AO={metrics['AO']:.3f}  " if is_got10k else ""
            print(f"  {seq:<14s} {got}mAP={metrics['mAP']:.3f}  AP50={metrics['AP50']:.3f}  "
                  f"AP75={metrics['AP75']:.3f}  |  F1={metrics['F1']:.3f} (threshold {score_thr:g})")

    if not per_seq:
        raise RuntimeError(f"No completed sequence to analyse: {out_dir}")

    # ---- pass 2: the threshold that maximises F (VOT-LT method, one threshold for all sequences) ----
    curve, best, per_seq_pr = f_curve_from_sequences(seq_records, iou_thr, thr_resolution)
    for row in per_seq:
        p, r = per_seq_pr[row["sequence"]]
        row["precision_opt"] = float(p[best["index"]])
        row["recall_opt"] = float(r[best["index"]])
        row["F_opt"] = _f_from(row["precision_opt"], row["recall_opt"])

    table = pd.DataFrame(per_seq).set_index("sequence")
    mean_row = table.mean(numeric_only=True)
    mean_over = {k: float(mean_row[k]) for k in summary_metrics if k in mean_row}
    # VOT definition: F is not the mean of per-sequence F values; it is computed from the mean P and R
    mean_over["F1"] = _f_from(mean_over["precision"], mean_over["recall"])
    mean_over["F_opt"] = best["F"]

    pooled = {"mAP": np.nan, "AP50": np.nan, "AP75": np.nan}
    if is_got10k:  # the official GOT-10k numbers: all frames of all sequences pooled
        pooled.update(got10k_metrics(np.concatenate(got10k_pooled)))
    pooled.update(coco_standard(all_records))
    pooled.update(operating_point(all_records, score_thr, iou_thr))
    pooled_opt = operating_point(all_records, best["threshold"], iou_thr)
    pooled.update({"precision_opt": pooled_opt["precision"], "recall_opt": pooled_opt["recall"],
                   "F_opt": pooled_opt["F1"]})

    summary = {
        "experiment": exp_name,
        "n_sequences": len(per_seq),
        "skipped_sequences": skipped,
        "score_thr": score_thr,
        "iou_thr": iou_thr,
        # The threshold that maximises F, and the sequence-averaged P, R, F at that threshold
        "optimal_threshold": {k: best[k] for k in ("threshold", "precision", "recall", "F", "resolution", "iou_thr")},
        # Mean over sequences (same idea as the MEAN row of the old coco_eval.py; F values use the VOT definition)
        "mean_over_sequences": mean_over,
        # All sequences as a single dataset (weighted by frames)
        "pooled": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in pooled.items()},
    }

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "config": meta["config"], "per_sequence": table.reset_index().to_dict("records")},
                  f, indent=2, ensure_ascii=False, default=float)
    curve_df = pd.DataFrame(curve)
    curve_df.to_csv(metrics_dir / "f_curve.csv", index=False, float_format="%.6f")

    # ---- plots ----
    if plots:
        plot_f_curve(curve, best, metrics_dir / "f_curve.png", exp_name, score_thr)
        for seq, (frames, events, ft) in plot_data.items():
            plot_dir = out_dir / "plots" / seq
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_iou_conf(frames, events, plot_dir / "iou_conf.png", seq, exp_name, score_thr, best["threshold"])
            if len(ft):
                plot_finetune_loss(ft, plot_dir / "finetune_loss.png", seq, exp_name)

    # ---- tables ----
    summary_df = pd.DataFrame({"mean over sequences": mean_over,
                               "pooled": {k: summary["pooled"].get(k, np.nan) for k in summary_metrics}})
    summary_df = summary_df.reindex(summary_metrics)
    params_df = pd.DataFrame(sorted(meta["config"].items()), columns=["parameter", "value"]).astype(str)
    with pd.ExcelWriter(metrics_dir / "metrics.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary")
        pd.DataFrame([summary["optimal_threshold"]]).to_excel(writer, sheet_name="optimal_threshold", index=False)
        table.reset_index().to_excel(writer, sheet_name="per_sequence", index=False)
        curve_df.to_excel(writer, sheet_name="f_curve", index=False)
        params_df.to_excel(writer, sheet_name="parameters", index=False)

    seq_cols = (GOT10K_METRICS if is_got10k else []) + [
        "mAP", "AP50", "AP75", "F_opt", "precision_opt", "recall_opt", "F1",
        "absent_reject_rate", "mean_iou_visible", "absent_frames", "template_updates", "ft_updates"]
    got = summary["pooled"]
    got10k_line = ([f"GOT-10k AO / SR0.50 / SR0.75 (all frames pooled, as the GOT-10k toolkit): "
                    f"{got['AO']:.4f} / {got['SR50']:.4f} / {got['SR75']:.4f}"] if is_got10k else [])
    lines = [f"Experiment: {exp_name}",
             f"Sequences: {len(per_seq)}  (skipped: {skipped or 'none'})", *got10k_line,
             f"mAP / AP50 / AP75 (mean over sequences): {mean_over['mAP']:.4f} / {mean_over['AP50']:.4f} / "
             f"{mean_over['AP75']:.4f}",
             f"F-max threshold (VOT method, {thr_resolution} candidates): score >= {best['threshold']:.4f}  ->  "
             f"P={best['precision']:.4f}  R={best['recall']:.4f}  F={best['F']:.4f}",
             f"Fixed threshold: score >= {score_thr:g}  ->  P={mean_over['precision']:.4f}  R={mean_over['recall']:.4f}  "
             f"F={mean_over['F1']:.4f}      (IoU >= {iou_thr:g})",
             "",
             summary_df.to_string(float_format=lambda v: f"{v:.4f}"),
             "",
             "Per sequence (*_opt: at the F-max threshold shared by all sequences):",
             table[seq_cols].to_string(float_format=lambda v: f"{v:.3f}")]
    (metrics_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print("\n" + "\n".join(lines[:8 + len(got10k_line)]))  # per-sequence lines were already printed above
    return {"summary": summary, "per_sequence": table, "summary_table": summary_df, "f_curve": curve_df}


def _no_ground_truth(out_dir: Path, meta: dict, with_results, verbose: bool) -> dict:
    """GOT-10k test split: no local ground truth; the results are scored by the GOT-10k server."""
    zip_path = out_dir / "got10k_submission.zip"
    times = []
    for seq in with_results:
        times += read_vot_result(out_dir / "predictions" / seq, seq)[2][1:]
    lines = [f"Experiment: {meta['experiment_name']}",
             f"Sequences with results: {len(with_results)} of {len(meta['sequences'])}",
             f"Speed: {1.0 / np.nanmean(times):.1f} FPS" if times else "Speed: -",
             "The GOT-10k test split has no local ground truth: AO / SR are computed by the GOT-10k server.",
             (f"Upload {zip_path} at http://got-10k.aitestunion.com/submit_instructions" if zip_path.is_file()
              else "got10k_submission.zip is written when all sequences have results.")]
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print("\n" + "\n".join(lines))
    return {"summary": None}
