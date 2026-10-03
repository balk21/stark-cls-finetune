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


def _load(name, outputs_dir):
    with open(Path(outputs_dir) / name / "metrics" / "metrics.json") as f:
        return json.load(f)


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


def per_sequence(names, metric="F1", outputs_dir=None) -> pd.DataFrame:
    """Rows = sequences, columns = experiments; for a single metric."""
    outputs_dir = Path(outputs_dir) if outputs_dir else get_paths().outputs
    cols = {}
    for n in names:
        data = _load(n, outputs_dir)
        cols[n] = {r["sequence"]: r.get(metric) for r in data["per_sequence"]}
    return pd.DataFrame(cols)


def export_comparison(names, out_path, outputs_dir=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        load_summaries(names, "mean_over_sequences", outputs_dir).to_excel(writer, sheet_name="mean_over_sequences")
        load_summaries(names, "pooled", outputs_dir).to_excel(writer, sheet_name="pooled")
        load_parameters(names, outputs_dir, only_different=False).to_excel(writer, sheet_name="parameters")
        load_summaries(names, "optimal_threshold", outputs_dir).to_excel(writer, sheet_name="optimal_threshold")
        for m in ("AP", "F_opt", "F1", "precision", "recall"):
            per_sequence(names, m, outputs_dir).to_excel(writer, sheet_name=f"per_seq_{m}")
    return out_path
