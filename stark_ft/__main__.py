"""
Command-line interface. The notebooks call this interface as well, using the vot1 Python.

    python -m stark_ft check
    python -m stark_ft show    [--config experiment.json] [--set key=value ...] [--json]
    python -m stark_ft smoke   [--config ...] [--set ...] [--sequence ballet] [--frames 30]
    python -m stark_ft run     [--config ...] [--set ...] [--overwrite] [--no-analyze]
    python -m stark_ft analyze outputs/<experiment> [--score-thr 0.35] [--iou-thr 0.5] [--thr-resolution 100] [--no-plots]
    python -m stark_ft list
    python -m stark_ft compare <exp1> <exp2> ... [--out comparison.xlsx] [--plot comparison.png]
    python -m stark_ft download-checkpoints [--model stark_st] [--model-config baseline_R101 ...]
    python -m stark_ft download-dataset
"""
import argparse
import json
import os
import sys

# The CLI is never interactive; the MPLBACKEND that Jupyter passes on to subprocesses
# (e.g. matplotlib_inline) may not exist in the vot1 environment.
os.environ["MPLBACKEND"] = "Agg"

import yaml  # noqa: E402

from stark_ft.config import ExperimentConfig  # noqa: E402

OUTPUT_DIR_MARKER = "OUTPUT_DIR="  # parsed by notebooks/nbhelper.py


def _build_config(args) -> ExperimentConfig:
    data = {}
    if args.config:
        data = ExperimentConfig.from_file(args.config).to_dict()
    for item in args.set or []:
        key, _, value = item.partition("=")
        data[key.strip()] = yaml.safe_load(value)  # "1e-4" -> float, "[a, b]" -> list, "true" -> bool
    return ExperimentConfig.from_dict(data).validate()


def _add_config_args(p):
    p.add_argument("--config", help="YAML/JSON experiment file")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="Override a parameter (repeatable)")


def cmd_show(args):
    from stark_ft.paths import get_paths
    from stark_ft.runner import resolve_sequences
    cfg = _build_config(args)
    paths = get_paths()
    ckpt = cfg.checkpoint_path(paths)
    info = {
        "experiment_name": cfg.experiment_name,
        "output_dir": str(paths.outputs / cfg.experiment_name),
        "output_exists": (paths.outputs / cfg.experiment_name).exists(),
        "checkpoint": str(ckpt),
        "checkpoint_exists": ckpt.is_file(),
        "config": cfg.to_dict(),
    }
    try:
        info["sequences"] = resolve_sequences(cfg, paths)
    except Exception as e:  # noqa: BLE001
        info["sequences_error"] = str(e)
    if args.json:
        print(json.dumps(info, ensure_ascii=False))
        return 0
    print(f"Experiment  : {info['experiment_name']}")
    exists_note = "  (EXISTS - resumed if the parameters are the same)" if info["output_exists"] else ""
    print(f"Output      : {info['output_dir']}{exists_note}")
    print(f"Checkpoint  : {info['checkpoint']}  ({'found' if info['checkpoint_exists'] else 'NOT FOUND'})")
    if "sequences" in info:
        seqs = info["sequences"]
        print(f"Sequences   : {len(seqs)} -> {', '.join(seqs[:8])}{' ...' if len(seqs) > 8 else ''}")
    else:
        print(f"Sequences   : ERROR - {info['sequences_error']}")
    print("Parameters:")
    for k, v in info["config"].items():
        print(f"  {k:22s} = {v!r}")
    return 0 if info["checkpoint_exists"] and "sequences" in info else 1


def cmd_compare(args):
    import pandas as pd

    from stark_ft import compare
    from stark_ft.plots import plot_metric_comparison
    names = args.names
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print("== Parameter differences ==")
    print(compare.load_parameters(names).to_string())
    print("\n== mAP / AP50 / AP75 (mean over sequences) ==")
    print(compare.load_summaries(names, "mean_over_sequences")[["mAP", "AP50", "AP75"]].to_string(float_format=fmt))
    print("\n== F-max threshold (VOT method, P/R averaged over sequences) ==")
    print(compare.load_summaries(names, "optimal_threshold")[["threshold", "precision", "recall", "F"]]
          .to_string(float_format=fmt))
    for kind, title in (("mean_over_sequences", "Mean over sequences"), ("pooled", "Pooled (all frames)")):
        print(f"\n== {title} ==")
        df = compare.load_summaries(names, kind)
        cols = [c for c in ["mAP", "AP50", "AP75", "precision_opt", "recall_opt", "F_opt", "precision", "recall",
                            "F1", "absent_reject_rate", "mean_iou_visible", "legacy_mAP", "legacy_AP50",
                            "n_sequences"] if c in df.columns]
        print(df[cols].to_string(float_format=fmt))
    if len(names) >= 2:
        for metric in ("mAP", "AP50", "AP75"):
            table = compare.per_sequence(names, metric)
            print(f"\n== {metric} per sequence (reference: {names[0]}, difference = experiment - reference) ==")
            diff = table.sub(table[names[0]], axis=0).drop(columns=names[0])
            print(pd.concat([table[[names[0]]], diff.add_prefix("Δ ")], axis=1).to_string(float_format=fmt))
    if args.out:
        print(f"\nExcel: {compare.export_comparison(names, args.out)}")
    if args.plot:
        df = compare.load_summaries(names, "mean_over_sequences")
        plot_metric_comparison(df, ["mAP", "AP50", "AP75", "F_opt", "absent_reject_rate"], args.plot)
        print(f"Plot: {args.plot}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m stark_ft")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Check the environment and paths")
    p = sub.add_parser("show", help="Validate the parameters and show where the experiment will be written")
    _add_config_args(p)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("smoke", help="Quick test without VOT (a few frames)")
    _add_config_args(p)
    p.add_argument("--sequence")
    p.add_argument("--frames", type=int, default=30)
    p = sub.add_parser("run", help="Run an experiment (+ analysis)")
    _add_config_args(p)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-analyze", action="store_true")
    p = sub.add_parser("analyze", help="Produce metrics and plots from existing results")
    p.add_argument("output_dir")
    p.add_argument("--score-thr", type=float)
    p.add_argument("--iou-thr", type=float)
    p.add_argument("--thr-resolution", type=int, help="Number of candidates in the F-max threshold search (default 100)")
    p.add_argument("--no-plots", action="store_true")
    sub.add_parser("list", help="List analysed experiments")
    p = sub.add_parser("compare", help="Compare experiments")
    p.add_argument("names", nargs="+")
    p.add_argument("--out", help="Excel output (.xlsx)")
    p.add_argument("--plot", help="Bar chart (.png)")
    p = sub.add_parser("download-checkpoints", help="Download official STARK checkpoints")
    p.add_argument("--model", default="stark_st", choices=["stark_st", "stark_s"])
    p.add_argument("--model-config", nargs="+", default=["baseline_R101"])
    p.add_argument("--force", action="store_true")
    sub.add_parser("download-dataset", help="Download the VOT-LT2020 sequences")
    args = parser.parse_args(argv)

    if args.cmd == "check":
        from stark_ft.setup_utils import check_environment
        return 1 if check_environment()["problems"] else 0
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "smoke":
        from stark_ft.setup_utils import smoke_test
        smoke_test(_build_config(args), args.sequence, args.frames)
        return 0
    if args.cmd == "run":
        from stark_ft.runner import run_experiment
        out_dir = run_experiment(_build_config(args), overwrite=args.overwrite, analyze=not args.no_analyze)
        print(f"\n{OUTPUT_DIR_MARKER}{out_dir}")
        return 0
    if args.cmd == "analyze":
        from stark_ft.analysis import analyze_experiment
        analyze_experiment(args.output_dir, score_thr=args.score_thr, iou_thr=args.iou_thr,
                           thr_resolution=args.thr_resolution, plots=not args.no_plots)
        return 0
    if args.cmd == "list":
        from stark_ft.compare import list_experiments
        names = list_experiments()
        print("\n".join(names) if names else "No analysed experiments.")
        return 0
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "download-checkpoints":
        from stark_ft.setup_utils import download_checkpoints
        download_checkpoints([(args.model, c) for c in args.model_config], force=args.force)
        return 0
    if args.cmd == "download-dataset":
        from stark_ft.setup_utils import download_dataset
        download_dataset()
        return 0


if __name__ == "__main__":
    sys.exit(main())
