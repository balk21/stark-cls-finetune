"""
Comparing the metrics of several experiments.
"""
import json
from pathlib import Path

import pandas as pd

from stark_ft.paths import get_paths


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


EXCEL_METRICS = ["mAP", "AP50", "AP75"]


def export_comparison(names, out_path, outputs_dir=None):
    """Excel with one row per experiment (in the given order): mAP, AP50, AP75 (mean over sequences)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = load_summaries(names, "mean_over_sequences", outputs_dir)[EXCEL_METRICS].astype(float)
    table.index.name = "experiment"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="results")
        sheet = writer.sheets["results"]
        sheet.column_dimensions["A"].width = max(len(n) for n in [table.index.name, *names]) + 2
        for row in sheet.iter_rows(min_row=2, min_col=2):
            for cell in row:
                cell.number_format = "0.0000"
    return out_path
