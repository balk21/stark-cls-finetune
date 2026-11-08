"""
Comparing the metrics of several experiments.
"""
import json
import math
from dataclasses import fields
from pathlib import Path

import pandas as pd
from openpyxl.styles import Font

from stark_ft.paths import get_paths
from stark_ft.test.config import NO_UPDATE_INTERVAL, ExperimentConfig


def list_experiments(outputs_dir=None):
    """Names of analysed experiments (those that have metrics/metrics.json)."""
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    if not outputs_dir.is_dir():
        return []
    return sorted(p.name for p in outputs_dir.iterdir() if (p / "metrics" / "metrics.json").is_file())


AP_METRICS = ("mAP", "AP50", "AP75")


def _as_current(d: dict, earlier: bool) -> dict:
    """metrics.json written by earlier versions: "AP" = "mAP"; their legacy_* values are this version's
    mAP / AP50 / AP75 (the other AP values there were computed differently and are not used)."""
    d = {{"AP": "mAP", "legacy_AP": "legacy_mAP"}.get(k, k): v for k, v in d.items()}
    if earlier:
        for k in AP_METRICS:
            d[k] = d.pop(f"legacy_{k}", float("nan"))
        d = {k: v for k, v in d.items() if not k.startswith("legacy_")}
    return d


def _load(name, outputs_dir):
    with open(Path(outputs_dir) / name / "metrics" / "metrics.json") as f:
        data = json.load(f)
    mean = data["summary"]["mean_over_sequences"]
    earlier = "legacy_mAP" in mean or "legacy_AP" in mean
    for kind in ("mean_over_sequences", "pooled"):
        data["summary"][kind] = _as_current(data["summary"][kind], earlier)
    data["per_sequence"] = [_as_current(r, earlier) for r in data["per_sequence"]]
    return data


def load_summaries(names, kind="mean_over_sequences", outputs_dir=None) -> pd.DataFrame:
    """kind: 'optimal_threshold' (F-max threshold and P/R/F at it), 'mean_over_sequences' (mean over sequences)
    or 'pooled' (all frames pooled)."""
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    rows = {}
    for n in names:
        data = _load(n, outputs_dir)
        rows[n] = {**data["summary"][kind], "n_sequences": data["summary"]["n_sequences"]}
    return pd.DataFrame(rows).T


def load_parameters(names, outputs_dir=None, only_different=True) -> pd.DataFrame:
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    df = pd.DataFrame({n: _load(n, outputs_dir)["config"] for n in names}).T.astype(str)
    if only_different and len(df) > 1:
        df = df.drop(columns=["name"]).loc[:, lambda d: d.nunique() > 1]
    return df


def per_sequence(names, metric="mAP", outputs_dir=None) -> pd.DataFrame:
    """Rows = sequences, columns = experiments; for a single metric."""
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    cols = {}
    for n in names:
        data = _load(n, outputs_dir)
        cols[n] = {r["sequence"]: r.get(metric) for r in data["per_sequence"]}
    return pd.DataFrame(cols)


EXCEL_COLUMNS = ["experiment", "lr", "epoch", "sequence", "method", *AP_METRICS]
WEIGHTED = [f"{k} weighted" for k in AP_METRICS]  # mean weighted by the length (frames) of each video
SUMMARY_COLUMNS = [*EXCEL_COLUMNS, *WEIGHTED]
# Sequence groups of group_table(): every sequence | the target is visible in every frame | absent in some frames
GROUPS = ("all", "visible", "absent")
GROUP_COLUMNS = ["experiment", "group", "FT", "TU", "videos", *AP_METRICS, *WEIGHTED]


def _video_frames(row: dict) -> float:
    """Length of the video of a per-sequence row; metrics.json of earlier versions only has `frames`, the evaluated
    frames (all but the first)."""
    if row.get("video_frames") is not None:
        return float(row["video_frames"])
    return float(row["frames"]) + 1 if row.get("frames") is not None else math.nan


def _settings(config: dict):
    """(lr, epoch, method) of an experiment for the Excel rows. epoch = fine-tuning steps on the first frame + at
    every template update (15+15 online, 15+0 init, 0+0 without fine-tuning); method = template update rule."""
    c = {f.name: f.default for f in fields(ExperimentConfig)}
    c.update(config)
    ft = c["ft_mode"] if c["model"] == "stark_st" else "none"
    lr = "-" if ft == "none" else f"{c['ft_lr']:g}"
    epoch = {"none": "0+0", "init": f"{c['ft_epochs_init']}+0"}.get(ft, f"{c['ft_epochs_init']}+{c['ft_epochs_online']}")
    if c["model"] != "stark_st":
        method = "-"  # STARK-S: no template update
    elif c["update_interval"] >= NO_UPDATE_INTERVAL:
        method = "noupd"
    else:
        method = {"stark": "orj", "max": "max"}[c["update_mode"]]
    return lr, epoch, method


def _sheet_name(i, lr, epoch, method):
    return f"{i} {method}" + (f" lr{lr}" if lr != "-" else "") + f" {epoch}"  # at most 31 characters


def _write(writer, sheet_name, table, mean_rows=0, number_format="0.0000"):
    table.to_excel(writer, sheet_name=sheet_name, index=False)
    sheet = writer.sheets[sheet_name]
    for col, name in enumerate(table.columns, 1):
        width = max(len(str(v)) for v in [name, *table[name].tolist()]) + 2
        sheet.column_dimensions[sheet.cell(row=1, column=col).column_letter].width = min(width, 60)
        if name in AP_METRICS or name in WEIGHTED:
            for row in range(2, len(table) + 2):
                sheet.cell(row=row, column=col).number_format = number_format
    for row in range(len(table) + 2 - mean_rows, len(table) + 2):
        for cell in sheet[row]:
            cell.font = Font(bold=True)


def _means(ap: pd.DataFrame, frames: pd.Series):
    """(plain mean, mean weighted by the video length) of the rows of `ap`; the weighted mean is NaN if a length is
    unknown (earlier metrics.json) or there is no row."""
    weighted = (ap.mul(frames.to_numpy(), axis=0).sum() / frames.sum() if len(ap) and frames.notna().all()
                else pd.Series(math.nan, index=ap.columns))
    weighted.index = [f"{k} weighted" for k in ap.columns]
    return ap.mean(), weighted


def _ft_tu(config: dict):
    """("✓" | "-", "✓" | "-"): whether the experiment fine-tunes (FT) and updates the template (TU)."""
    c = {f.name: f.default for f in fields(ExperimentConfig)}
    c.update(config)
    st = c["model"] == "stark_st"
    ft = st and c["ft_mode"] != "none"
    tu = st and c["update_interval"] < NO_UPDATE_INTERVAL
    return ("✓" if ft else "-"), ("✓" if tu else "-")


def group_table(names, outputs_dir=None) -> pd.DataFrame:
    """Three rows per experiment (in the given order), one per sequence group: "all" sequences, "visible" (the target
    is visible in every frame) and "absent" (the target is absent in some frames). Columns: FT, TU, number of videos,
    mAP / AP50 / AP75 in % as the plain mean over the group's sequences and as the mean weighted by the video length."""
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    rows = []
    for name in names:
        data = _load(name, outputs_dir)
        ft, tu = _ft_tu(data["config"])
        seqs = data["per_sequence"]
        known = all("absent_frames" in r for r in seqs)  # earlier metrics.json may not have it
        members = {"all": seqs,
                   "visible": [r for r in seqs if r.get("absent_frames") == 0] if known else [],
                   "absent": [r for r in seqs if (r.get("absent_frames") or 0) > 0] if known else []}
        for group in GROUPS:
            part = members[group]
            ap = pd.DataFrame([{k: r.get(k, math.nan) for k in AP_METRICS} for r in part],
                              columns=list(AP_METRICS)).astype(float) * 100
            mean, weighted = _means(ap, pd.Series([_video_frames(r) for r in part], dtype=float))
            rows.append({"experiment": name, "group": group, "FT": ft, "TU": tu, "videos": len(part),
                         **mean.to_dict(), **weighted.to_dict()})
    return pd.DataFrame(rows, columns=GROUP_COLUMNS)


def _tex(text: str) -> str:
    return "".join("\\" + ch if ch in "_%&#$" else ch for ch in str(text))


def latex_rows(table: pd.DataFrame, weighted=False, labels=None, group_names=("target visible", "target absent")):
    """Rows of a LaTeX tabular "name & FT & TU & mAP & AP50 & AP75 \\\\" from group_table(), in % with 2 decimals; a
    \\hline after every experiment. labels: {experiment: row name} (default: the experiment name); group_names: added
    to the row name for the "visible" and the "absent" group."""
    labels = labels or {}
    suffix = {"all": "", "visible": f" {group_names[0]}", "absent": f" {group_names[1]}"}
    cols = WEIGHTED if weighted else list(AP_METRICS)
    mark = {"✓": r"\checkmark", "-": "--"}
    lines = []
    for name, rows in table.groupby("experiment", sort=False):
        label = labels.get(name, _tex(name))
        for _, r in rows.iterrows():
            values = " & ".join("" if pd.isna(r[c]) else f"{r[c]:.2f}" for c in cols)
            lines.append(f"{label}{suffix[r['group']]} & {mark[r['FT']]} & {mark[r['TU']]} & {values} \\\\")
        lines.append(r"\hline")
    return "\n".join(lines)


def export_latex(names, out_path, labels=None, group_names=("target visible", "target absent"), outputs_dir=None):
    """A .tex file with the rows of latex_rows() twice: the plain means over the sequences of each group and the means
    weighted by the video length."""
    table = group_table(names, outputs_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    head = "% mAP / AP50 / AP75 in %: all sequences / target visible in every frame / target absent in some frames\n"
    out_path.write_text(head + "% plain mean over the sequences\n" + latex_rows(table, False, labels, group_names)
                        + "\n% mean weighted by the video length (frames)\n"
                        + latex_rows(table, True, labels, group_names) + "\n", encoding="utf-8")
    return out_path


def export_comparison(names, out_path, outputs_dir=None):
    """Excel. Sheet "summary": one row per experiment (in the given order) with lr, epoch, sequence, method and
    mAP / AP50 / AP75 over its sequences, as the plain mean and as the mean weighted by the length (frames) of each
    video. If an experiment has several sequences, every experiment also gets its own sheet with one row per
    sequence and the two means as the last two rows, and sheet "groups" has group_table() (in %)."""
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary, per_experiment = [], []
    for i, name in enumerate(names, 1):
        data = _load(name, outputs_dir)
        lr, epoch, method = _settings(data["config"])
        rows = pd.DataFrame([{"experiment": name, "lr": lr, "epoch": epoch, "sequence": r["sequence"],
                              "method": method, **{k: r.get(k, math.nan) for k in AP_METRICS}}
                             for r in data["per_sequence"]], columns=EXCEL_COLUMNS)
        ap = rows[list(AP_METRICS)].astype(float)
        mean, weighted = _means(ap, pd.Series([_video_frames(r) for r in data["per_sequence"]], dtype=float))
        sequence = rows["sequence"].iloc[0] if len(rows) == 1 else f"{len(rows)} sequences"
        summary.append({"experiment": name, "lr": lr, "epoch": epoch, "sequence": sequence, "method": method,
                        **mean.to_dict(), **weighted.to_dict()})
        blank = {c: "" for c in EXCEL_COLUMNS}
        mean_rows = pd.DataFrame([{**blank, "sequence": "mean", **mean.to_dict()},
                                  {**blank, "sequence": "weighted mean", **dict(zip(AP_METRICS, weighted))}])
        per_experiment.append((_sheet_name(i, lr, epoch, method), pd.concat([rows, mean_rows], ignore_index=True),
                               len(rows)))
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _write(writer, "summary", pd.DataFrame(summary, columns=SUMMARY_COLUMNS))
        if any(n > 1 for _, _, n in per_experiment):  # some experiment has several sequences
            for sheet_name, table, _ in per_experiment:
                _write(writer, sheet_name, table, mean_rows=2)
            _write(writer, "groups", group_table(names, outputs_dir), number_format="0.00")
    return out_path
