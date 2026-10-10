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


def _write(writer, sheet_name, table, mean_row=False):
    table.to_excel(writer, sheet_name=sheet_name, index=False)
    sheet = writer.sheets[sheet_name]
    for col, name in enumerate(table.columns, 1):
        width = max(len(str(v)) for v in [name, *table[name].tolist()]) + 2
        sheet.column_dimensions[sheet.cell(row=1, column=col).column_letter].width = min(width, 60)
        if name in AP_METRICS:
            for row in range(2, len(table) + 2):
                sheet.cell(row=row, column=col).number_format = "0.0000"
    if mean_row:
        for cell in sheet[len(table) + 1]:
            cell.font = Font(bold=True)


def export_comparison(names, out_path, outputs_dir=None):
    """Excel. Sheet "summary": one row per experiment (in the given order) with lr, epoch, sequence, method and
    mAP / AP50 / AP75 (for several sequences: their plain mean). If an experiment has several sequences, every
    experiment also gets its own sheet with one row per sequence and the plain (not frame-weighted) mean as the last
    row."""
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
        mean = rows[list(AP_METRICS)].astype(float).mean()  # plain mean over the sequences
        sequence = rows["sequence"].iloc[0] if len(rows) == 1 else f"{len(rows)} sequences"
        summary.append({"experiment": name, "lr": lr, "epoch": epoch, "sequence": sequence, "method": method,
                        **mean.to_dict()})
        mean_row = {**{c: "" for c in EXCEL_COLUMNS}, "sequence": "mean", **mean.to_dict()}
        per_experiment.append((_sheet_name(i, lr, epoch, method), pd.concat([rows, pd.DataFrame([mean_row])],
                                                                             ignore_index=True)))
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _write(writer, "summary", pd.DataFrame(summary, columns=EXCEL_COLUMNS))
        if any(len(t) > 2 for _, t in per_experiment):  # some experiment has several sequences
            for sheet_name, table in per_experiment:
                _write(writer, sheet_name, table, mean_row=True)
    return out_path
