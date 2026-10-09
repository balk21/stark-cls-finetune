"""Comparison Excel: mAP / AP50 / AP75 of every experiment, one row each, in the given order."""
import json
import tempfile
from pathlib import Path

import openpyxl

from stark_ft.test.compare import export_comparison


def _experiment(outputs, name, m, earlier=False):
    summary = {"mAP": m, "AP50": m + 0.1, "AP75": m - 0.1, "F_opt": 0.5}
    if earlier:  # metrics.json of earlier versions: these values as legacy_*, other AP values under the plain names
        summary = {"AP": 0.9, "AP50": 0.9, "AP75": 0.9, "F_opt": 0.5,
                   "legacy_AP": m, "legacy_AP50": m + 0.1, "legacy_AP75": m - 0.1}
    d = outputs / name / "metrics"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps({
        "summary": {"mean_over_sequences": summary, "pooled": summary, "n_sequences": 1},
        "config": {}, "per_sequence": []}))


def test_excel_has_one_row_per_experiment():
    with tempfile.TemporaryDirectory() as tmp:
        outputs = Path(tmp)
        for name, m, old in (("b_exp", 0.5, False), ("a_exp", 0.3, True), ("c_exp", 0.7, False)):
            _experiment(outputs, name, m, old)
        path = export_comparison(["b_exp", "a_exp", "c_exp"], outputs / "cmp.xlsx", outputs_dir=outputs)
        wb = openpyxl.load_workbook(path)
        assert wb.sheetnames == ["results"]
        rows = list(wb["results"].iter_rows(values_only=True))
        assert rows[0] == ("experiment", "mAP", "AP50", "AP75")
        assert [r[0] for r in rows[1:]] == ["b_exp", "a_exp", "c_exp"]  # the given order
        assert [round(v, 6) for v in rows[2][1:]] == [0.3, 0.4, 0.2]


if __name__ == "__main__":  # without pytest: python -m tests.test_compare
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
