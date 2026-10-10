"""
Command-line interface (the notebooks call it as well, with the Python of the vot1 environment).

Shared
    python -m stark_ft check
    python -m stark_ft weights                       # official STARK weights and the runs trained here, by origin
    python -m stark_ft download-checkpoints [--model stark_st] [--model-config baseline_R101 ...]
Train
    python -m stark_ft prepare-train-data --datasets got10k coco [--got10k-url URL|PATH ...] [--archives DIR]
    python -m stark_ft train [--config configs/train_example.yaml] [--set key=value ...] [--dry-run] [--overwrite]
    python -m stark_ft train-report <run name>
    python -m stark_ft train-list
Test
    python -m stark_ft download-dataset [--sequences all | bull ballet ...]   (optional: tests download what they use)
    python -m stark_ft smoke [--config ...] [--set ...] [--sequence ballet] [--frames 30]
    python -m stark_ft show [--config configs/test_example.yaml] [--set key=value ...]
    python -m stark_ft test [--config configs/test_example.yaml] [--set key=value ...] [--overwrite] [--no-analyze]
    python -m stark_ft analyze outputs/<experiment> [--score-thr 0.35] [--iou-thr 0.5] [--thr-resolution 100]
    python -m stark_ft list
    python -m stark_ft compare <exp1> <exp2> ... [--out comparison.xlsx] [--plot comparison.png]
"""
import argparse
import json
import os
import sys

# The CLI is never interactive; the MPLBACKEND that Jupyter passes on to subprocesses
# (e.g. matplotlib_inline) may not exist in the vot1 environment.
os.environ["MPLBACKEND"] = "Agg"

import yaml  # noqa: E402

OUTPUT_DIR_MARKER = "OUTPUT_DIR="  # parsed by notebooks/nbhelper.py


def _params(cls, args):
    """Parameters from --config (YAML / JSON) and --set key=value (values parsed as YAML: 1e-4, [a, b], true)."""
    data = cls.from_file(args.config).to_dict() if args.config else {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        data[key.strip()] = yaml.safe_load(value)
    return cls.from_dict(data).validate()


def _add_config_args(p, kind):
    p.add_argument("--config", help=f"YAML / JSON file with {kind} parameters")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="Set a parameter (repeatable)")


# ====================================================================== train
def cmd_train(args):
    from stark_ft.train.config import TrainConfig
    from stark_ft.train.run import describe, run_training
    tc = _params(TrainConfig, args)
    info = describe(tc)
    if args.json:
        print(json.dumps(info, ensure_ascii=False))
        return 0
    if args.dry_run:
        exists = "  (EXISTS - resumed if the parameters are the same)" if info["run_exists"] else ""
        print(f"Run         : {info['run_name']}{exists}")
        print(f"Folder      : {info['run_dir']}")
        print(f"Export      : {info['export']}")
        print(f"Epochs      : {info['epochs']}, {info['steps_per_epoch']} optimizer steps/epoch, "
              f"{info['accumulation']} micro-batches/step")
        print(f"Init        : {info['init'] or info['init_error']}")
        for d, root in info["dataset_roots"].items():
            print(f"Dataset     : {d:12s} {root}")
        for p in info["dataset_problems"]:
            print(f"PROBLEM     : {p}")
        print("Parameters:")
        for k, v in info["config"].items():
            print(f"  {k:22s} = {v!r}")
        return 1 if info["dataset_problems"] or info["init_error"] else 0
    out = run_training(tc, overwrite=args.overwrite)
    print(f"\n{OUTPUT_DIR_MARKER}{out}")
    return 0


def _print_status(st, indent="", name=None):
    state = "finished" if st["finished"] else f"{st['epochs_done']}/{st['total_epochs']} epochs"
    line = f"{indent}{name or st['run_name']}: stage {st['stage']}, {st['model_config']}, {'+'.join(st['datasets'])}"
    if st["stage"] == 2:
        line += f", from {st['init_run']}"
    line += f" - {state}"
    if not st["finished"] and st.get("remaining_hours") is not None:
        line += f" (~{st['remaining_hours']:.1f} h left at {st['epoch_seconds'] / 60:.1f} min/epoch)"
    print(line)


def _print_runs(runs_by_origin, testable_only=False):
    from stark_ft.weights import ORIGIN_TEXT
    any_run = False
    for origin, runs in runs_by_origin.items():
        if not runs:
            continue
        any_run = True
        print(f"Runs {ORIGIN_TEXT.get(origin, origin)}:")
        for st in runs:
            if testable_only and not (st["stage"] == 2 and st["finished"]):
                note = "stage 1: init of stage 2, not testable" if st["stage"] == 1 else "not finished"
                print(f"    ({st['run_name']}: {note})")
            else:
                _print_status(st, indent="  ", name=f"weights={st['run_name']!r}" if testable_only else None)
    if not any_run:
        print("No training runs.")


def cmd_train_report(args):
    from stark_ft.train.report import report
    st = report(args.run, plot=not args.no_plot)
    if args.json:
        print(json.dumps(st, ensure_ascii=False))
        return 0
    _print_status(st)
    print(f"Folder      : {st['run_dir']}")
    print(f"GPU         : {st['gpu']}")
    print(f"Init        : {st['init'] or 'ImageNet'}")
    for col, v in st["last"].items():
        print(f"  {col:24s} {v['value']:.5f}  (epoch {v['epoch']})")
    if st.get("plot"):
        print(f"Plot        : {st['plot']}")
    from stark_ft.weights import ORIGIN_TEXT
    print(f"Origin      : {ORIGIN_TEXT.get(st['origin'], st['origin'])}")
    if st["finished"]:
        hint = (f"test with weights={st['run_name']!r}" if st["stage"] == 2
                else f"stage 2 of this family: same parameters with stage=2 (init={st['run_name']!r})")
        print(f"Use         : {hint}")
    return 0


# ====================================================================== test
def _test_params(args):
    from stark_ft.test.config import ExperimentConfig
    return _params(ExperimentConfig, args)


def cmd_show(args):
    from stark_ft.paths import get_paths
    from stark_ft.test import datasets
    from stark_ft.test.runner import resolve_sequences
    cfg = _test_params(args)
    paths = get_paths()
    from stark_ft import weights
    w = weights.describe(cfg.weights, cfg.model, cfg.model_config, paths)
    info = {
        "experiment_name": cfg.experiment_name,
        "output_dir": str(paths.outputs / cfg.experiment_name),
        "output_exists": (paths.outputs / cfg.experiment_name).exists(),
        "weights": w,
        "config": cfg.to_dict(),
        "unused": cfg.unused_fields(),
    }
    try:
        info["sequences"] = resolve_sequences(cfg, paths, fetch=False)
        info["to_fetch"] = datasets.to_fetch(cfg.dataset, info["sequences"], paths)
    except Exception as e:  # noqa: BLE001
        info["sequences_error"] = str(e)
    if args.json:
        print(json.dumps(info, ensure_ascii=False))
        return 0
    print(f"Experiment  : {info['experiment_name']}")
    exists_note = "  (EXISTS - resumed if the parameters are the same)" if info["output_exists"] else ""
    print(f"Output      : {info['output_dir']}{exists_note}")
    print(f"Weights     : {cfg.weights} - {w.get('origin', '')}")
    print(f"              {w['path']}  ({'PROBLEM: ' + w['problem'] if w['problem'] else w.get('note', 'found')})")
    if "sequences" in info:
        seqs = info["sequences"]
        print(f"Dataset     : {cfg.dataset}")
        print(f"Sequences   : {len(seqs)} -> {', '.join(seqs[:8])}{' ...' if len(seqs) > 8 else ''}")
        if info["to_fetch"]:
            print(f"Data        : {info['to_fetch']}; fetched when the run starts")
    else:
        print(f"Sequences   : ERROR - {info['sequences_error']}")
    print("Parameters:")
    for k, v in info["config"].items():
        print(f"  {k:22s} = {v!r}{'   (not used with these settings)' if k in info['unused'] else ''}")
    return 0 if not w["problem"] and "sequences" in info else 1


def cmd_compare(args):
    import pandas as pd

    from stark_ft.test import compare
    from stark_ft.test.plots import plot_metric_comparison
    names = args.names
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    print("== Parameter differences ==")
    print(compare.load_parameters(names).to_string())
    pooled = compare.load_summaries(names, "pooled")
    if "AO" in pooled.columns and pooled["AO"].notna().any():
        print("\n== GOT-10k AO / SR0.50 / SR0.75 (all frames pooled) ==")
        print(pooled[["AO", "SR50", "SR75"]].to_string(float_format=fmt))
    print("\n== mAP / AP50 / AP75 (mean over sequences) ==")
    print(compare.load_summaries(names, "mean_over_sequences")[["mAP", "AP50", "AP75"]].to_string(float_format=fmt))
    print("\n== F-max threshold (VOT method, P/R averaged over sequences) ==")
    print(compare.load_summaries(names, "optimal_threshold")[["threshold", "precision", "recall", "F"]]
          .to_string(float_format=fmt))
    for kind, title in (("mean_over_sequences", "Mean over sequences"), ("pooled", "Pooled (all frames)")):
        print(f"\n== {title} ==")
        df = compare.load_summaries(names, kind)
        cols = [c for c in ["AO", "SR50", "SR75", "mAP", "AP50", "AP75", "precision_opt", "recall_opt", "F_opt",
                            "precision", "recall", "F1", "absent_reject_rate", "mean_iou_visible",
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


# ====================================================================== main
def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m stark_ft", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="command")

    # shared
    sub.add_parser("check", help="Check the environment and the paths")
    sub.add_parser("weights", help="Official STARK weights and the runs trained here (for the test parameter weights)")
    p = sub.add_parser("download-checkpoints", help="Download official STARK checkpoints")
    p.add_argument("--model", default="stark_st", choices=["stark_st", "stark_s"])
    p.add_argument("--model-config", nargs="+", default=["baseline_R101"])
    p.add_argument("--force", action="store_true")

    # train
    p = sub.add_parser("prepare-train-data", help="Train: download / extract training datasets (got10k, coco)")
    p.add_argument("--datasets", nargs="+", required=True, choices=["got10k", "got10k_full", "coco"])
    p.add_argument("--archives", help="Folder for the archives (default: the `archives` path, <train_data>/_archives)")
    p.add_argument("--got10k-url", action="append", default=[],
                   help="GOT-10k archive: download link (Google Drive share links work) or path (repeatable)")
    p.add_argument("--delete-archives", action="store_true", help="Delete the archives after extracting them")
    p = sub.add_parser("train", help="Train: STARK-ST stage 1 or 2 on a combination of datasets")
    _add_config_args(p, "training")
    p.add_argument("--overwrite", action="store_true", help="Delete an existing run with the same name first")
    p.add_argument("--dry-run", action="store_true", help="Only validate and show what would be done")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("train-report", help="Train: progress of a run (writes history.png)")
    p.add_argument("run", help="Run name (in train_outputs) or run folder")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--json", action="store_true")
    sub.add_parser("train-list", help="Train: list the training runs")

    # test
    p = sub.add_parser("download-dataset", help="Test: download VOT-LT2020 sequences in advance (optional)")
    p.add_argument("--sequences", nargs="+", default=["all"], help="'all' (default, 17.6 GB) or sequence names")
    p = sub.add_parser("smoke", help="Test: quick check without vot-toolkit (a few frames)")
    _add_config_args(p, "test")
    p.add_argument("--sequence")
    p.add_argument("--frames", type=int, default=30)
    p = sub.add_parser("show", help="Test: validate the parameters, show the output folder")
    _add_config_args(p, "test")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("test", aliases=["run"], help="Test: run an experiment on VOT-LT2020 (+ analysis)")
    _add_config_args(p, "test")
    p.add_argument("--overwrite", action="store_true", help="Delete an existing experiment with the same name first")
    p.add_argument("--no-analyze", action="store_true")
    p = sub.add_parser("analyze", help="Test: metrics and plots from existing results")
    p.add_argument("output_dir")
    p.add_argument("--score-thr", type=float)
    p.add_argument("--iou-thr", type=float)
    p.add_argument("--thr-resolution", type=int, help="Number of candidates in the F-max threshold search (100)")
    p.add_argument("--no-plots", action="store_true")
    sub.add_parser("list", help="Test: list analysed experiments")
    p = sub.add_parser("compare", help="Test: compare experiments")
    p.add_argument("names", nargs="+")
    p.add_argument("--out", help="Excel output (.xlsx): lr, epoch, sequence, method, mAP / AP50 / AP75 per experiment "
                                 "(several sequences: also one sheet per experiment with their mean)")
    p.add_argument("--plot", help="Bar chart (.png)")
    args = parser.parse_args(argv)

    # shared
    if args.cmd == "check":
        from stark_ft.setup_utils import check_environment
        return 1 if check_environment()["problems"] else 0
    if args.cmd == "weights":
        from stark_ft.weights import list_weights
        w = list_weights()
        print("Official STARK weights (github.com/researchmm/Stark), weights='official' with the model_config:")
        for o in w["official"]:
            state = "downloaded" if o["downloaded"] else "downloaded automatically when first used"
            print(f"  {o['model']:9s} model_config={o['model_config']!r:29s} trained on {o['trained_on']:38s} {state}")
        print()
        _print_runs(w["runs"], testable_only=True)
        return 0
    if args.cmd == "download-checkpoints":
        from stark_ft.setup_utils import download_checkpoints
        download_checkpoints([(args.model, c) for c in args.model_config], force=args.force)
        return 0
    # train
    if args.cmd == "prepare-train-data":
        from stark_ft.paths import get_paths
        from stark_ft.train.data import prepare
        paths = get_paths()
        prepare(args.datasets, paths.train_data, args.archives or paths.archives, args.got10k_url,
                args.delete_archives)
        return 0
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "train-report":
        return cmd_train_report(args)
    if args.cmd == "train-list":
        from stark_ft.weights import list_weights
        _print_runs(list_weights()["runs"])
        return 0
    # test
    if args.cmd == "download-dataset":
        from stark_ft.test.data import download_dataset
        download_dataset(sequences="all" if args.sequences == ["all"] else args.sequences)
        return 0
    if args.cmd == "smoke":
        from stark_ft.test.smoke import smoke_test
        smoke_test(_test_params(args), args.sequence, args.frames)
        return 0
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd in ("test", "run"):
        from stark_ft.test.runner import run_experiment
        out_dir = run_experiment(_test_params(args), overwrite=args.overwrite, analyze=not args.no_analyze)
        print(f"\n{OUTPUT_DIR_MARKER}{out_dir}")
        return 0
    if args.cmd == "analyze":
        from stark_ft.test.analysis import analyze_experiment
        analyze_experiment(args.output_dir, score_thr=args.score_thr, iou_thr=args.iou_thr,
                           thr_resolution=args.thr_resolution, plots=not args.no_plots)
        return 0
    if args.cmd == "list":
        from stark_ft.test.compare import list_experiments
        names = list_experiments()
        print("\n".join(names) if names else "No analysed experiments.")
        return 0
    if args.cmd == "compare":
        return cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
