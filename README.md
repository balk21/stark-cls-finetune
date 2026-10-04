# stark-cls-finetune — inference-time fine-tuning of the STARK-ST classification head

**English** | [Türkçe](README.tr.md)

This repository adds **video-specific fine-tuning of the classification head (cls_head)** at test (inference)
time to the **STARK-ST** variant of the [STARK](https://github.com/researchmm/Stark) tracker (ICCV 2021), and
evaluates it on **VOT-LT2020** with **detection-style (COCO)** metrics.

It also contains the **original STARK-ST training** (both stages), runnable on a single GPU with the original
effective batch size and on **any combination of training datasets** (see [§8](#8-training-stark-st-base-training)).

The whole workflow runs from Jupyter notebooks. The code contains no hard-coded file paths; the same repository
runs on a Vast.ai server, on Google Colab and on a personal computer.

---

## Contents
1. [Quick start (Vast.ai / Google Colab)](#1-quick-start)
2. [On your own computer](#2-on-your-own-computer)
3. [Folder layout](#3-folder-layout)
4. [Method](#4-method)
5. [Parameters](#5-parameters)
6. [Outputs and file formats](#6-outputs-and-file-formats)
7. [Metrics](#7-metrics)
8. [Training STARK-ST (base training)](#8-training-stark-st-base-training)
9. [Command line (CLI)](#9-command-line-cli)
10. [Troubleshooting](#10-troubleshooting)
11. [Known limitations and open issues](#11-known-limitations-and-open-issues)
12. [Changes from the earlier research code](#12-changes-from-the-earlier-research-code)
13. [License and citation](#13-license-and-citation)

---

## 1. Quick start

### 1.1 Vast.ai

**Requirements**

| | |
|---|---|
| GPU | NVIDIA **RTX 3000 / 4000 series** (tested). RTX 5000 series (Blackwell) GPUs do **not** work with the PyTorch 2.4.1 in the environment. |
| GPU memory | ≥ 8 GB |
| Disk | ≥ 40 GB (dataset ≈ 17 GB + conda environment + outputs) |
| Software | An image with `conda` (e.g. Vast.ai's PyTorch templates with conda) and Jupyter |

**Steps**

1. Rent a machine with the specs above on Vast.ai and open Jupyter.
2. Open a terminal in Jupyter and clone the repository:
   ```bash
   git clone https://github.com/balk21/stark-cls-finetune.git
   ```
3. Open `notebooks/00_setup.ipynb` and run the cells in order:
   the `vot1` environment is created, the checkpoint and the dataset are downloaded and a short smoke test runs.
4. Edit the parameters in `notebooks/01_run_experiment.ipynb` and run the experiment.
5. Compare experiments with `notebooks/02_compare.ipynb`.
6. Optional: train STARK-ST itself with `notebooks/03_train.ipynb` (see [§8](#8-training-stark-st-base-training)).

> **Notebook kernel:** any Python 3 kernel works. The notebooks do the actual work in the background with the
> Python of the `vot1` environment (`notebooks/nbhelper.py`). You do not need to change the kernel.

### 1.2 Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/00_setup_colab.ipynb)

1. Open `notebooks/00_setup_colab.ipynb` in Colab with the badge above and choose
   *Runtime → Change runtime type → GPU* (T4, L4 or A100).
2. Run the cells. Google Drive is mounted, and the **same `vot1` environment** as on Vast.ai is created with
   micromamba (same package versions). Note: results are only identical on the same GPU type; see
   [Results across GPUs](#results-across-gpus).
3. Then open `01_run_experiment.ipynb` [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/01_run_experiment.ipynb) and `02_compare.ipynb`
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/02_compare.ipynb) in Colab.

Colab gives a fresh machine in every session, so everything that is slow to prepare is cached on Google Drive in
`MyDrive/LOKAP/` and restored by the first cell of every notebook (`notebooks/colab_setup.py`). The folder is set by
`DRIVE_ROOT` at the top of the first cell; use the same value in every notebook.

| | First session (once) | Later sessions |
|---|---|---|
| `vot1` environment | created (~5–10 min), cached on Drive (`cache/vot1_env_<hash>.tar`, 7.8 GB) | restored (~1–3 min) |
| Checkpoint | downloaded to Drive (`checkpoints/`) | copied (seconds) |
| VOT-LT2020 dataset | downloaded (20–60 min), cached on Drive (`cache/votlt2020_sequences.tar`, 17 GB) | restored (a few min) |
| Outputs | written to Drive (`outputs/`) | kept; interrupted runs resume |

**Google Drive space needed:** ≈ 26 GB + experiment outputs. Your own checkpoints can be uploaded to
`MyDrive/LOKAP/checkpoints/<stark_st2|stark_s>/<model_config>/`.

**Training on Colab:** open `03_train.ipynb` [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/03_train.ipynb)
directly; it does not need the VOT dataset (see [§8](#8-training-stark-st-base-training)).

## 2. On your own computer

The same notebooks are used. If the checkpoints or the dataset already exist elsewhere, instead of downloading them,
copy `configs/paths.local.example.yaml` to `configs/paths.local.yaml` and set the paths:

```yaml
checkpoints: /home/user/stark/checkpoints/train   # <checkpoints>/stark_st2/baseline_R101/STARKST_ep0050.pth.tar
dataset: /home/user/vot/votlt2020/sequences        # <dataset>/<sequence>/{color/, groundtruth.txt, sequence}
outputs: /home/user/stark_outputs
train_data: /data/tracking_train                   # training only: <train_data>/{got10k/train, coco, ...} (§8.3)
train_outputs: /home/user/stark_training           # training only: one folder per training run (§8.5)
```

This file is ignored by git. The same settings can be given with the environment variables
`STARK_CLEAN_CHECKPOINTS`, `STARK_CLEAN_DATASET`, `STARK_CLEAN_OUTPUTS`, `STARK_CLEAN_TRAIN_DATA` and
`STARK_CLEAN_TRAIN_OUTPUTS` (priority: environment variable > `paths.local.yaml` > `paths.yaml`).

**Nothing is written into the dataset folder.** The `list.txt` that vot-toolkit needs is created in each
experiment's own `vot_workspace/` folder, with absolute paths to the sequences.

## 3. Folder layout

```
stark-cls-finetune/
├── notebooks/
│   ├── 00_setup.ipynb            Setup on a conda machine (environment, checkpoint, dataset, smoke test)
│   ├── 00_setup_colab.ipynb      Setup on Google Colab (environment / dataset cached on Google Drive)
│   ├── 01_run_experiment.ipynb   Parameters → run → results
│   ├── 02_compare.ipynb          Comparing experiments
│   ├── 03_train.ipynb            Training STARK-ST (stage 1 / 2) on a combination of datasets
│   ├── nbhelper.py               Notebook helpers (standard library only)
│   └── colab_setup.py            Google Colab session bootstrap (standard library only)
├── stark_ft/                     Experiment framework
│   ├── config.py                 ExperimentConfig: ALL parameters and their validation
│   ├── paths.py                  Path resolution
│   ├── runner.py                 VOT workspace setup + `vot evaluate` + collecting results
│   ├── vot_entry.py              Tracker process that vot-toolkit starts for every sequence
│   ├── tracker_factory.py        Builds the tracker object from the parameters
│   ├── evaluation.py             COCO mAP/AP50/AP75, P/R/F1, F-max threshold, legacy COCO
│   ├── analysis.py               Metric tables + plot generation
│   ├── plots.py                  Plots
│   ├── compare.py                Comparing experiments
│   ├── setup_utils.py            Environment check, downloads, smoke test
│   ├── training.py               TrainConfig: training parameters, run folders, resuming, export
│   ├── train_data.py             Downloading / extracting the training datasets
│   └── __main__.py               CLI (python -m stark_ft ...)
├── lib/                          STARK core
│   ├── models/stark/             Network architecture (original STARK)
│   ├── config/                   Model config defaults (original STARK; stark_st1/ is used by stage-1 training)
│   ├── train/                    STARK training code (from the original: data loading, actors, trainer)
│   ├── test/tracker/
│   │   ├── stark_st.py           STARK-ST tracker (original + update bookkeeping)
│   │   ├── stark_st_ft.py        ★ Tracker with fine-tuning
│   │   ├── ft_sampling.py        ★ Positive jitter and negative region generation
│   │   └── stark_s.py            STARK-S tracker
│   └── utils/
├── model_configs/                STARK model YAMLs (stark_st1/, stark_st2/, stark_s/)
├── configs/
│   ├── paths.yaml                Default paths
│   ├── paths.local.example.yaml  Template for machine-specific paths
│   └── experiments/example.yaml  Example experiment file for the CLI
├── environment/vot1_environment.yml
├── tests/                        python -m tests.<name> (no GPU needed)
├── checkpoints/   data/   outputs/   (ignored by git)
```

## 4. Method

### 4.1 The confidence score of STARK-ST

Besides the target box, STARK-ST outputs a **confidence score** in every frame. It is the sigmoid output of a
3-layer MLP (**cls_head**) applied to the transformer decoder output `hs` (256-dimensional).
The score is used in two places:

1. **Template update decision:** every `update_interval` frames, if the score is greater than `update_conf_thr`
   (0.5), the dynamic template is updated with the prediction in that frame.
2. **Long-term evaluation:** the tracker is expected to output a low score in frames where the target is not
   visible. In the metrics of this repository, the score is the basis of the "target found / not found" decision.

### 4.2 Fine-tuning

During base training, cls_head learns a generic "target present / absent" decision. In this method cls_head is
reset to its base weights at the start of every video and briefly trained specifically for that video's target.
The backbone, the transformer and the box head are **frozen**; gradients only flow through cls_head.

One fine-tuning **session** works as follows:

```
frame + target box
  ├─ positive box  = the box (+ the ST2 training jitter if ft_pos_jitter)            → label 1
  └─ negative box  = a region of the same frame that does NOT contain the target     → label 0  (posneg only)
        │
        ▼  for each box: crop search region → backbone → merge with templates → transformer → hs
        ▼
  cls_head(hs) → BCEWithLogitsLoss → AdamW (ft_lr, ft_weight_decay) → grad clip (ft_grad_clip_norm)
  (with 1–2 samples, 1 epoch = 1 optimisation step; a new optimiser for every session)
```

**Modes (`ft_mode`)**

| Mode | When fine-tuning happens | Source of the box |
|---|---|---|
| `none` | Never (plain STARK-ST; baseline) | — |
| `init` | Only on the first frame, `ft_epochs_init` steps | Ground truth (the initial box given by VOT) |
| `online` | `ft_epochs_init` steps on the first frame **+** `ft_epochs_online` steps on every frame with a template update | First frame: GT. Afterwards: **the tracker's own prediction** |

In `online` mode, fine-tuning is triggered only on frames with a template update. So with `update_interval=99999`,
`online` is effectively the same as `init`.

**Sample types (`ft_samples`)**

| Value | Description |
|---|---|
| `pos` | Positive sample only. Training with label "1" only pushes the head's score up in general; the score also rises on frames without the target (see §11). |
| `posneg` | Positive + negative. The negative is a region of the same frame, shifted so that it lies **completely outside** the search crop of the target box (8 directions are tried; the one whose crop has the largest fraction inside the image is chosen). This is a **provisional** solution until a proper negative-sampling method is found. |

## 5. Parameters

All parameters are defined in the `ExperimentConfig` class in `stark_ft/config.py`. In the notebook they are given
as `PARAMS = dict(...)`; on the CLI as a YAML file or `--set key=value`.
Parameters that are not given take their default values.

### Identity and model

| Parameter | Default | Description |
|---|---|---|
| `name` | `None` | Name of the output folder (`outputs/<name>/`). If `None`, it is generated from the parameters, e.g. `st101_online_pos_lr0.0001_i15_o1_int100_s0`. Must not contain spaces or `/\:*?"<>\|`. |
| `model` | `"stark_st"` | `"stark_st"` (with confidence score, can be fine-tuned) or `"stark_s"` (no score; a constant 1.0 is reported; `ft_mode="none"` required). |
| `model_config` | `"baseline_R101"` | A YAML in `model_configs/<stark_st2\|stark_s>/`. `stark_st`: `baseline_R101` (ST101), `baseline` (ST50), `baseline_R101_got10k_only`, `baseline_got10k_only`. `stark_s`: `baseline`, `baseline_got10k_only`. |
| `checkpoint` | `None` | If `None`, the official file name is used (`STARKST_ep0050.pth.tar` / `STARKS_ep0500.pth.tar`). If only a file name is given, it is looked up in `<checkpoints>/<stark_st2\|stark_s>/<model_config>/`, e.g. `"STARKSTcoco_ep0050.pth.tar"`. A value containing `/` is used as a path (relative paths are resolved against the repository root). `"train:<run name>"` uses the result of a training run (`<train_outputs>/<run name>/final.pth.tar`, §8.5). |

### Data

| Parameter | Default | Description |
|---|---|---|
| `sequences` | `"all"` | `"all"` or a list of sequence names, e.g. `["bull", "ballet"]`. A sequence name is a folder name in the dataset folder. |

### Template update (`stark_st` only)

| Parameter | Default | Description |
|---|---|---|
| `update_interval` | `100` | A template update is attempted every N frames. `99999` = no updates. (`TEST.UPDATE_INTERVALS` in the model YAMLs is **not used** in this repository; the value always comes from this parameter.) |
| `update_conf_thr` | `0.5` | The score must be **greater** than this for an update (original STARK: 0.5). |
| `max_template_updates` | `-1` | Maximum number of template updates in a sequence. `-1` = unlimited. |

### Fine-tuning (`stark_st` only)

| Parameter | Default | Description |
|---|---|---|
| `ft_mode` | `"online"` | `"none"`, `"init"`, `"online"` (see §4.2). |
| `ft_samples` | `"pos"` | `"pos"` or `"posneg"` (see §4.2). |
| `ft_lr` | `1e-4` | AdamW learning rate (base Stage-2 training: 1e-4). |
| `ft_epochs_init` | `15` | Number of optimisation steps on the first frame. `0` = no fine-tuning on the first frame. |
| `ft_epochs_online` | `1` | Number of steps in every online session (`ft_mode="online"`). |
| `ft_weight_decay` | `1e-4` | AdamW weight decay (same as base training). |
| `ft_grad_clip_norm` | `0.1` | Gradient norm clipping (same as base training). |
| `ft_pos_jitter` | `True` | Apply the ST2 training search jitter to the positive box? If `False`, the box is used as is. |
| `ft_center_jitter` | `4.5` | Jitter centre-shift factor (ST2: 4.5). The centre is shifted uniformly within a window of width `sqrt(w·h)·4.5`. |
| `ft_scale_jitter` | `0.5` | Jitter scale factor (ST2: 0.5). The size is scaled by `exp(N(0,1)·0.5)`. |
| `max_ft_updates` | `-1` | Maximum number of online fine-tuning sessions in a sequence. `-1` = unlimited. |
| `seed` | `0` | Random seed (jitter, PyTorch, deterministic cuDNN). The same seed and parameters give **exactly the same** result. Repeat with different seeds to measure variance. |

### Evaluation

| Parameter | Default | Description |
|---|---|---|
| `eval_score_thr` | `0.35` | **Fixed** threshold: score threshold for the "tracker found the target" decision. Only affects the `precision / recall / F1` columns; does not affect mAP / AP50 / AP75 or the F-max threshold result. |
| `eval_iou_thr` | `0.5` | Minimum IoU for a found box to count as correct (both at the fixed threshold and in the F-max search). |
| `eval_thr_resolution` | `100` | Number of candidate thresholds in the F-max threshold search (vot-toolkit default: 100). |

These three values can be changed without re-running the tracking: `nb.analyze(OUT, score_thr=0.5, thr_resolution=200)`.
Changing only these does not make the experiment "different" (it does not affect the resume rule).

### vot-toolkit

| Parameter | Default | Description |
|---|---|---|
| `run_redetection` | `False` | Also run the second experiment of the VOT-LT2020 stack (`redetection`)? Roughly doubles the run time; the metrics in this repository only use the `longterm` results. |
| `tracker_timeout` | `300` | Timeout (seconds) for a single tracker response, including model loading. |

### Experiment name, resuming and overwriting

- Running again with the same `name` + the **same** parameters **resumes**: vot-toolkit skips completed sequences.
  To continue an interrupted run, simply run the cell again. (The `eval_*` parameters are not part of this check.)
- The same `name` + **different** parameters raises an error and the old results are kept. To overwrite, use
  `nb.run(PARAMS, overwrite=True)` (CLI: `--overwrite`).
- Resuming is also refused if the **tracking code changed** since the experiment was started (e.g. after a
  `git pull`): a hash of `lib/`, `model_configs/` and the tracker entry point is stored in `experiment.json`, so results
  of two code versions are never mixed. Use `overwrite=True` for a fresh run.
- This is why `vot evaluate` is called **without `-f`**: every experiment has its own workspace, so there are no stale
  results from other experiments, and skipping completed sequences is what makes resuming possible. `vot analysis`
  (and its `--nocache` option) is not used; the metrics are always recomputed from the raw results.

## 6. Outputs and file formats

Every experiment is written to `outputs/<experiment name>/`.

| File | Content |
|---|---|
| `experiment.json` | All parameters, checkpoint and dataset paths, sequence list, git commit, date |
| `run.log` | Full vot-toolkit output. If a sequence fails, the tracker's error output is written to `vot_workspace/logs/`. |
| `run_status.json` | Completed / missing sequences |
| `predictions/<seq>/<seq>_001.txt` | Box per frame `x,y,w,h` (top-left corner + size, pixels). **The first line is `1`** (init frame marker, VOT format). |
| `predictions/<seq>/<seq>_001_confidence.value` | Confidence score per frame (first line empty). |
| `predictions/<seq>/<seq>_001_time.value` | Time per frame (seconds). |
| `predictions/<seq>/frames.csv` | Everything together: `frame, x, y, w, h, conf, time, gt_visible, gt_x, gt_y, gt_w, gt_h, iou` |
| `tracker_logs/<seq>/finetune_loss.txt` | CSV: `frame, session, epoch, loss, pos_prob, neg_prob, n_pos, n_neg, neg_coverage`. One row per optimisation step. `pos_prob`/`neg_prob`: sigmoid output at that step (before the update). `neg_coverage`: fraction of the negative crop inside the image. |
| `tracker_logs/<seq>/events.txt` | CSV: `frame, event, conf_score`. `event` = `template_update` or `ft_update`. |
| `plots/<seq>/iou_conf.png` | IoU and confidence score vs. frame number. Grey hatched area: target absent. Red dashed: template update. Green dotted: fine-tuning update. Black dash-dot: fixed threshold (`eval_score_thr`). Purple: F-max threshold. |
| `plots/<seq>/finetune_loss.png` | Loss on top (log scale), positive/negative sample probabilities below. Vertical lines are session starts (`init`, `f<frame>`). |
| `metrics/summary.txt` | Readable summary (first lines: mAP / AP50 / AP75, then the F-max threshold and fixed threshold results) + per-sequence table |
| `metrics/metrics.xlsx` | Sheets `summary`, `optimal_threshold`, `per_sequence`, `f_curve`, `parameters` |
| `metrics/f_curve.csv` | Sequence-averaged `precision`, `recall`, `F` at every candidate threshold |
| `metrics/f_curve.png` | P / R / F vs. threshold on the left (purple: selected threshold, black: fixed threshold), precision–recall curve on the right |
| `metrics/metrics.json` | The same data, machine-readable |
| `vot_workspace/` | The experiment's own VOT workspace (`config.yaml`, `trackers.ini`, raw `results/`) |

**Frame numbering:** everywhere, `frame = 0` is the first (init) frame. In the VOT result files this is the line
number minus 1. The first frame is not evaluated (the GT is given to the tracker there).

## 7. Metrics

The evaluation is **detection**-style: every frame is an "image", the target is a single object. The computation
uses the `pycocotools` library (`stark_ft/evaluation.py`).

**Primary metrics: mAP, AP50 and AP75** (§7.1). They are listed first in every summary, table and comparison.

### 7.1 mAP / AP50 / AP75 (standard COCO usage)

- **mAP** is COCO's main "AP" (`stats[0]` of `COCOeval`): precision averaged over recall levels and over the IoU
  thresholds 0.50, 0.55, …, 0.95. **AP50** and **AP75** use a single IoU threshold (0.50 / 0.75).
- In the summaries, "mean over sequences" is the mean of the per-sequence values (as in the old `coco_eval.py`
  MEAN row); "pooled" treats all frames of all sequences as one dataset.
- **All** frames except the first one are evaluated.
- Frames where the target is not visible (GT = NaN) are added as images without annotations. Every prediction on
  these frames counts as a **false positive**.
- **No score threshold is applied**: all predictions are given with their scores. mAP already measures the quality
  of the score ranking over all thresholds. A threshold can never increase mAP / AP50 / AP75.
- Coordinates are used as floats.

### 7.2 "Found / not found" metrics at a fixed threshold

The tracker "claims the target" when score ≥ `eval_score_thr`:

| | Tracker claims the target (score ≥ threshold) | Tracker does not (score < threshold) |
|---|---|---|
| **Target visible**, IoU ≥ `eval_iou_thr` | TP | FN |
| **Target visible**, IoU < `eval_iou_thr` | FP + FN (wrong location) | FN |
| **Target absent** | FP | TN |

- `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, `F1 = 2PR / (P + R)`
- `absent_reject_rate = TN / (number of frames without the target)`: the fraction of frames without the target on
  which the tracker correctly said "not found".

This counting uses the same matching rule as COCOeval (IoU ≥ `eval_iou_thr`, one GT per frame); it was verified to
give exactly the same P/R as counting from COCOeval's own matches.

### 7.3 F-max threshold (VOT-LT method)

`eval_score_thr` is a fixed threshold. The main comparison criterion is, as in the VOT-LT protocol, F computed at
**the threshold that maximises F**. The method is the same as `vot/analysis/tpr.py` of vot-toolkit 0.5.3; the only
difference is how P and R are counted:

1. **Candidate thresholds:** the scores of all sequences of the experiment (init frames excluded) are pooled and
   sorted in descending order. `eval_thr_resolution − 2` evenly spaced values are taken from this order, and +∞ and
   −∞ are added at the ends (an exact copy of vot-toolkit's `determine_thresholds`).
2. **P and R per sequence at every threshold:** vot-toolkit uses IoU-weighted P/R. Here the "found / not found"
   counting of §7.2 is used (COCOeval matching: IoU ≥ `eval_iou_thr`). If no prediction passes the threshold,
   P = 1 and R = 0, as in vot-toolkit.
3. **Aggregation:** P and R are averaged over sequences. F = 2PR / (P + R) is computed from these averages
   (it is not the mean of per-sequence F values).
4. **Selection:** the threshold with the largest F is selected (the highest threshold on ties). This threshold is
   **the same for all sequences**.

Outputs: `optimal_threshold` (threshold, P, R, F), per-sequence `precision_opt / recall_opt / F_opt`
(each sequence's value at the shared threshold), `metrics/f_curve.csv` and `f_curve.png`.

> The threshold is selected using the GT. This is the standard reporting protocol of VOT-LT (every tracker is
> compared at its own best threshold); the tracker does not know this threshold while running.

Verification: the candidate threshold list is identical to vot-toolkit's own function; the vectorised P/R matches
counting threshold by threshold; the selected threshold matches a brute-force search (tested on 50 sequences).

### 7.4 Other

- `mean_iou_visible`: mean IoU over frames where the target is visible (no threshold).
- `legacy_mAP / legacy_AP50 / legacy_AP75`: **exactly** the computation of the earlier
  `testler/detailed_analysis/coco_eval.py` script (frames without the target excluded, predictions below the
  threshold dropped, coordinates rounded to integers). Only for comparison with old results; it does not penalise
  false detections on frames without the target. Verified to match the old script on 5 sequences.
- **Mean over sequences:** the mean of each sequence's metric (same idea as the old `MEAN` row). Exception: `F1` and
  `F_opt` are computed from the mean P and R, following the VOT definition.
- **Pooled:** the frames of all sequences as a single dataset; longer sequences get more weight.

## 8. Training STARK-ST (base training)

`notebooks/03_train.ipynb` trains the STARK-ST model itself, with the **original STARK training procedure**
(`lib/train/`, ported from the official repository), on **any combination of training datasets**: one dataset
alone, or any two, three or all four of GOT-10k, COCO, LaSOT and TrackingNet. GOT-10k and COCO are downloaded /
extracted automatically. A trained model is then evaluated like the official one, with or without inference-time
fine-tuning, via `checkpoint="train:<run name>"` (§5).

### 8.1 Quick start

1. Open `notebooks/03_train.ipynb` (Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/03_train.ipynb); on Colab choose an **A100** or **L4** GPU runtime).
2. Set the parameters (`TRAIN = dict(...)`, §8.4) and, for GOT-10k, the download links (§8.3).
3. Run *Prepare*: on Colab the session is set up without the VOT dataset; then the datasets are downloaded /
   extracted (only what is missing) and a dry run shows the run folder, the epochs, the steps and the initial weights.
4. Run *Train*. Progress lines show samples/s, the epoch ETA and the training ETA. Stopping the cell stops the
   run; running it again **resumes** from the last finished epoch.
5. `nb.show_training(TRAIN)` shows the progress and `history.png` at any time.

### 8.2 What is trained

| Stage | Trained parameters | Epochs (LR ÷10 at) | Loss | Starts from |
|---|---|---|---|---|
| 1 | backbone + transformer + box head (the classification head is not used) | 500 (400) | GIoU × 2 + L1 × 5 | ImageNet ResNet backbone |
| 2 | classification head only (everything else frozen) | 50 (40) | BCE | stage-1 weights (`init`) |

Both stages (from the model YAMLs in `model_configs/stark_st1/`, `model_configs/stark_st2/`): 60 000 training
samples per epoch, AdamW with lr 1e-4 (backbone × 0.1) and weight decay 1e-4, gradient clipping at norm 0.1,
validation on 10 000 samples every 20 (stage 1) / 10 (stage 2) epochs, seed 42, cuDNN deterministic. The
sampling of template / search frames, the augmentations and the losses are those of the original code.

Stage 2 initialised with `init=None` / `"official"` starts from the **official STARK-ST checkpoint without its
classification head**. Its backbone, transformer and box head are the official stage-1 weights (stage 2 freezes
them), so this is the same as starting stage 2 from the official stage-1 model, which is not published separately.
`init="<stage-1 run name>"` starts from your own stage-1 run.

### 8.3 Training datasets

| `datasets` key | STARK dataset | Videos / images | Size (extracted) | Preparation |
|---|---|---|---|---|
| `got10k` | `GOT10K_vottrain`: GOT-10k train **without the 1000 videos that overlap with VOT** | 7 086 videos | 73.9 GB (whole GOT-10k train folder) | from the official archives (registration required, see below) |
| `got10k_full` | `GOT10K_train_full`: all GOT-10k train videos (GOT-10k protocol; used by the `*_got10k_only` configs) | 9 335 videos | (same folder) | (same) |
| `coco` | `COCO17`: COCO 2017 train, every object is a "video" of one frame | 118 287 images | 19.3 GB (+ 0.8 GB annotations) | **automatic** from images.cocodataset.org (19.6 GB of zip files) |
| `lasot` | `LASOT` (train split) | 1 120 videos | very large | manual: `<train_data>/lasot/<class>/<class>-<n>/` |
| `trackingnet` | `TRACKINGNET` | ≈ 30 000 videos | ≈ 1 TB | manual: `<train_data>/trackingnet/TRAIN_0 ... TRAIN_11/` |

Validation (`val_datasets=["got10k"]`) uses `GOT10K_votval`: 1 249 GOT-10k train videos that are **disjoint**
from `GOT10K_vottrain` (as in STARK). It needs the GOT-10k folder even when training only on COCO; use
`val_datasets=[]` to train without validation.

**Folder layout** (`train_data`, default `data/train/`; on Colab the local disk `/content/train_data`):

```
<train_data>/
├── got10k/train/list.txt, GOT-10k_Train_000001/ ... GOT-10k_Train_009335/
├── coco/annotations/instances_train2017.json
├── coco/images/train2017/*.jpg
├── lasot/ ...                      (manual)
├── trackingnet/TRAIN_0 ... TRAIN_11 (manual)
└── _archives/{coco,got10k}/        downloaded archives (on Colab: MyDrive/LOKAP/train_archives/)
```

**GOT-10k** can only be downloaded after a free registration at
[got-10k.aitestunion.com/downloads](http://got-10k.aitestunion.com/downloads); the download links are sent by
e-mail. They may point to Google Drive, e.g. `full_data.zip` (70.7 GB; contains `train/`, `val/` and `test/`).
The archives can be given in three ways:

1. **On Colab, for a file on Google Drive (recommended):** open the link in the browser with the Google account you
   use in Colab, choose *Add shortcut to Drive* and put the shortcut into
   `MyDrive/LOKAP/train_archives/got10k/`; leave `GOT10K_URLS` empty. Nothing is downloaded or copied:
   the archive is read from Drive and extracted to the local disk in every new session. If reading it fails because
   the file's download quota is exceeded, make a copy of it into the same folder instead (right-click the shortcut →
   *Make a copy*; needs 70.7 GB of Drive space).
2. **Links** in `GOT10K_URLS` (CLI: `--got10k-url`): downloaded once into the archive folder, resumable. Google Drive
   share links (`https://drive.google.com/file/d/<id>/view...`) are converted to direct downloads. On Colab the archive
   folder is on Drive, so a 70 GB download is written to Drive through the mount; option 1 avoids that.
3. **Files** that are already somewhere: put them into `<archives>/got10k/`, or give their paths (or a folder) in
   `GOT10K_URLS`; paths are used in place and never deleted.

Only the train videos and `list.txt` files are extracted (`val/` and `test/` are skipped), also from archives inside
archives (e.g. train split zips). The preparation checks that all 9 335 train videos are there and takes the official
`list.txt` (or creates an identical one: sorted names, the order the `data_specs` split files index into).

**COCO** is downloaded automatically (`train2017.zip`, `annotations_trainval2017.zip`), with resuming of an
interrupted download, and checked for 118 287 images.

**How the preparation behaves:** archives are kept (`delete_archives=False`), so a new Colab session or a new
machine only extracts them; a finished download is remembered by its URL (e-mailed links may expire); extraction
goes into a temporary folder that is renamed only when it is complete (an interrupted preparation leaves nothing
half-done); free disk space is checked before extracting; a dataset folder that already has the expected layout
is only read, never modified. An existing extracted GOT-10k copy can be used directly by setting `train_data` so that
`<train_data>/got10k/train/list.txt` exists.

**Space on Colab:** Google Drive holds the archives (COCO 19.6 GB, GOT-10k ≈ the size of its train archives) and
the training runs (§8.5); the local disk of the runtime must hold the extracted datasets (GOT-10k ≈ 74 GB, COCO ≈ 20 GB)
plus the environment (≈ 8 GB). If the runtime's disk is too small, the preparation stops with a message saying how
much is needed.

### 8.4 Parameters (`TrainConfig`, `stark_ft/training.py`)

| Parameter | Default | Description |
|---|---|---|
| `name` | `None` | Run folder name (`<train_outputs>/<name>/`). If `None`, generated from the parameters: model, stage, datasets, `from-<init>`, `e<epochs>`, `r<ratios>`, seed; e.g. `st101_stage2_got10k+coco_s42`. |
| `model_config` | `"baseline_R101"` | `baseline_R101` (STARK-ST101) or `baseline` (STARK-ST50); the YAML in `model_configs/stark_st1/` (stage 1) or `stark_st2/` (stage 2). The `*_got10k_only` YAMLs only differ in the data (use `datasets=["got10k_full"]` with them); for stage 2 they select the official GOT-10k-only checkpoint as `init`. |
| `stage` | `2` | `1` or `2` (§8.2). |
| `init` | `None` | Stage 2: `None` / `"official"` (official checkpoint without the classification head, §8.2), the name of a finished stage-1 run, or a checkpoint path. Stage 1: must be `None` (ImageNet backbone). |
| `datasets` | `["got10k"]` | Any non-empty combination of `got10k` (or `got10k_full`), `coco`, `lasot`, `trackingnet`. The order does not matter (it is normalised to STARK's order). |
| `dataset_ratios` | `None` | Sampling weight per dataset (same order as `datasets`). `None` = equal weights, as in STARK: each training sample first picks a dataset with these weights, then a video in it. |
| `val_datasets` | `["got10k"]` | `["got10k"]` (GOT10K_votval) or `[]` (no validation). |
| `epochs` | `None` | `None` = original (stage 1: 500, stage 2: 50). |
| `lr_drop_epoch` | `None` | Epoch after which the learning rate is divided by 10. `None` = original (400 / 40). |
| `samples_per_epoch` | `None` | Training samples per epoch. `None` = 60 000. |
| `val_samples_per_epoch` | `None` | Validation samples. `None` = 10 000. |
| `val_interval` | `None` | Validate every N epochs. `None` = original (20 / 10). |
| `effective_batch` | `128` | Samples per optimizer step. **128 = the original** (8 GPUs × 16). Changing it changes the training. |
| `micro_batch` | `16` | Samples per forward/backward pass; `effective_batch / micro_batch` passes are accumulated per step (§8.6). Only memory and speed depend on it. |
| `num_workers` | `8` | Data loading processes. |
| `seed` | `42` | Seed (STARK default). |
| `keep_every` | `None` | Keep the weights every N epochs (`None`: stage 1: 50, stage 2: 10). |

### 8.5 Run folder, resuming, using the result

Every run is written to `<train_outputs>/<run name>/` (default `outputs/training/`; on Colab
`MyDrive/LOKAP/training/`):

| File | Content |
|---|---|
| `train_config.json` | All parameters, dataset folders, initial weights, GPU, TF32 setting, code hash, date |
| `history.csv` | One row per epoch: duration, learning rate, training (and validation) losses / IoU |
| `history.png` | Plot of `history.csv` (written by `nb.show_training` / `train-report`; dashed lines: LR drops) |
| `logs/train.log` | The progress output |
| `checkpoints/latest.pth.tar` | Complete state after the last finished epoch (network, optimizer, LR scheduler, random number generators); written atomically every epoch. ST101: ≈ 0.56 GB (stage 1), ≈ 0.19 GB (stage 2). |
| `checkpoints/STARKST_epXXXX.pth.tar` | Network weights every `keep_every` epochs and at the end (≈ 0.19 GB each) |
| `final.pth.tar` | Final weights, written when the run finishes |

- **Resuming:** the same parameters → the run continues from `latest.pth.tar`. The resumed run is **bit-identical**
  to an uninterrupted one (verified by killing a run during training and resuming it). Different parameters for the
  same `name` → an error (use another `name`, or `overwrite=True`). A run started with a different version of the
  training code (`lib/train`, `lib/models`, `lib/config`, `lib/utils`, `model_configs/stark_st1|2`, `stark_ft/training.py`) is not resumed either.
- **Evaluating a stage-2 run:** `checkpoint="train:<run name>"` with the same `model_config` in
  `01_run_experiment.ipynb` (stage-1 runs have an untrained classification head; use them as `init` of stage 2).
- **Stage 1 → stage 2:** `TrainConfig(stage=2, init="<stage-1 run name>", ...)`.
- Nothing is written to the `checkpoints` folder.

### 8.6 One GPU instead of eight: effective batch and time

The original was trained on 8 GPUs with 16 samples each; the gradients of the 8 GPUs are averaged, so every
optimizer step uses the mean gradient of **128** samples. Here the loader produces micro-batches of `micro_batch`
samples; the loss of each is divided by `effective_batch / micro_batch`, the gradients are accumulated, and only
then is the gradient clipped and the optimizer step taken. Because STARK freezes all BatchNorm layers of the backbone,
the forward pass of a sample does not depend on the other samples in its batch, so this is the **same update** as
the original. Checked numerically: the accumulated gradient of 8 × 16 samples and the gradient of one batch of 128
samples differ by 3·10⁻¹⁵ (relative) in float64; in float32 by ≈ 10⁻⁴, only through the order of summation. The
number of optimizer steps per epoch is also the same (60 000 / 128 = 468; leftover samples are not used).

What is *not* reproduced exactly is the random sample stream: the original had 8 processes with their own data
workers. Runs with the same parameters (including `micro_batch` and `num_workers`) on the same GPU type are
bit-identical; changing `micro_batch` / `num_workers` changes which random samples are drawn, not the method.

**Time.** Stage 1 processes 500 × 60 000 = 30 M samples, stage 2 50 × 60 000 = 3 M. Measured on an RTX 3060 laptop
GPU (`micro_batch=16`): stage 2 ≈ 35 samples/s (≈ 24 h), stage 1 ≈ 20 samples/s (≈ 17 days). Datacenter GPUs are
several times faster; read the training ETA in the progress output after the first steps. Colab sessions end after
some hours, but every finished epoch is saved and the run resumes in the next session. Colab compute units are
consumed per hour, so check the cost of a stage-1 run before starting it. **TF32:** A100 / L4 (and the RTX
3000 / 4000 series) use TF32 for convolutions by default, the T4 does not (see [Results across GPUs](#results-across-gpus));
use the same GPU type for runs you compare.

### 8.7 Differences from the original STARK training code

| Original STARK | Here |
|---|---|
| 8 GPUs (DistributedDataParallel), 16 samples per GPU | 1 GPU, gradient accumulation to the same effective batch (§8.6) |
| Dataset paths in a machine-specific `local.py` | `train_data` (§2) and the `datasets` parameter; any combination |
| A resumed stage-2 run loaded the stage-1 weights again **after** resuming, overwriting the already trained classification head | Initial weights are loaded only when a run starts from scratch |
| Checkpoints only in the last 10 epochs and every 100 epochs, without the LR-scheduler and random-number-generator states (an interruption could lose up to 99 epochs) | `latest.pth.tar` with the complete state after every epoch, written atomically, plus the weights every `keep_every` epochs; resuming is bit-identical |
| Pickled settings objects in the checkpoints | Plain dictionaries |
| Did not run with newer PyTorch / pandas (`torch._six`, `storage()._new_shared`, `read_csv(squeeze=True)`) | Fixed, without changing the behaviour |

Kept as in the original: in stage 1, if the network predicts an invalid box (x2 < x1 or y2 < y1) for any sample of a
(micro-)batch, the GIoU computation fails and STARK counts the GIoU loss of that whole batch as 0 (only the L1 loss
is used). This happens at the very beginning of training from scratch (≈ 1 of 8 samples is invalid in the first steps)
and disappears quickly. Here it applies per micro-batch (16 samples by default, the same as on each of the 8 GPUs
in the original).

## 9. Command line (CLI)

The notebooks call these commands; they can also be used directly (in the `vot1` environment, from the repository root):

```bash
python -m stark_ft check                                   # environment + paths
python -m stark_ft show --config configs/experiments/example.yaml
python -m stark_ft smoke --set ft_mode=online --frames 50  # quick test without VOT
python -m stark_ft run --config configs/experiments/example.yaml --set 'sequences=[bull]' --set ft_samples=posneg
python -m stark_ft analyze outputs/<experiment> --score-thr 0.5 --thr-resolution 100
python -m stark_ft list
python -m stark_ft compare <exp1> <exp2> --out comparison.xlsx --plot comparison.png
python -m stark_ft download-checkpoints --model stark_st --model-config baseline_R101 baseline
python -m stark_ft download-dataset
python -m stark_ft prepare-train-data --datasets got10k coco [--got10k-url URL ...] [--archives DIR]
python -m stark_ft train --set stage=2 --set 'datasets=[got10k, coco]' --dry-run   # check, then run without --dry-run
python -m stark_ft train-report <run name>                 # progress + history.png
python -m stark_ft train-list
python -m tests.test_sampling                              # tests of the negative-sample geometry
python -m tests.test_config                                # tests of reading / validating parameters
python -m tests.test_training                              # tests of the training parameters
python -m tests.test_train_data                            # tests of the training data preparation
```

`--set` values are parsed as YAML: `1e-4` → number, `true` → bool, `[a, b]` → list.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Conda environment 'vot1' not found` | Step 1 of `00_setup.ipynb`. If the environment is elsewhere, set `VOT1_PYTHON=/path/envs/vot1/bin/python`. |
| `Checkpoint not found` | Step 3 of `00_setup.ipynb`, or put the file at the location shown in the error message. |
| Google Drive "quota exceeded" | Download the file in a browser from the link in the error message and put it in the folder shown there. |
| `No VOT sequences found in the dataset folder` | Step 4 of `00_setup.ipynb`, or set `dataset` in `configs/paths.local.yaml`. |
| Missing sequences, e.g. `Completed sequences: 47/50` | A failing sequence is skipped and the others keep running. The tracker's error output is in `outputs/<experiment>/vot_workspace/logs/`. Running the same experiment again skips completed sequences and retries the missing ones. |
| `no kernel image is available` / sm_120 warning | The GPU is not supported by the PyTorch in the environment (RTX 5000 series). Use an RTX 3000/4000 series GPU. |
| "A newer version of the VOT toolkit is available" | Ignore it; the repository is tested with vot-toolkit **0.5.3**. Do not upgrade. |
| Colab: `WARNING: no GPU in this session` | *Runtime → Change runtime type → GPU*, then run the first cell again. |
| Colab: caching to Drive fails | Not enough Google Drive space (≈ 26 GB needed). Free space and run the first cell again; completed steps are not repeated. |
| Training: `CUDA out of memory` | Lower `micro_batch` (e.g. 8) and keep `effective_batch=128`: the update stays the same (§8.6). |
| Training: the runtime / machine crashes or runs out of RAM | Lower `num_workers` (COCO's annotation file is large and every data worker uses memory). On Colab, use a High-RAM runtime. |
| Training: `Not enough disk space for ...` | The extracted datasets do not fit on the local disk (GOT-10k ≈ 74 GB, COCO ≈ 20 GB). Use a machine / runtime with a larger disk, or fewer datasets. |
| Training: `No GOT-10k archives in ...` | Register on the GOT-10k site and put the e-mailed links into `GOT10K_URLS`, or put the archives into the folder named in the message (§8.3). |
| Training: `... already exists with different parameters` / `different version of the training code` | As for experiments: use another `name`, or `overwrite=True` for a fresh run. |

## 11. Known limitations and open issues

- **`pos` mode inflates the score in general.** With a single label "1", the easiest solution for BCE is to increase
  the output for every input. Since AdamW normalises the gradient, every step moves by ≈ `ft_lr` even when the loss
  is tiny. As a result, the score also rises on frames without the target (`absent_reject_rate` drops) and template
  updates are triggered more often.
- **The `posneg` negatives are provisional.** A negative region taken from the same frame may contain other objects
  that look like the target (e.g. other dancers). High `neg_prob` values in `finetune_loss.txt` show this. The code
  is organised so that, once a proper negative-sampling method is found, only the `posneg` branch in
  `lib/test/tracker/ft_sampling.py` and `stark_st_ft.py` has to change.
- **Online mode treats the tracker's own prediction as a positive.** Since the template update decision is also made
  by the score of the head being fine-tuned, there is a risk of a self-reinforcing loop.
- **There is no principled way to choose `ft_lr` / `ft_epochs_*`.** The values are empirical. Keep in mind that
  choosing hyperparameters on the test sequences makes the results look optimistic.
- **vot-toolkit restarts the tracker process for every sequence** (the model is reloaded per sequence, ~3–4 s).
- **Training: LaSOT and TrackingNet are not prepared automatically.** The training code supports them
  (`datasets=["lasot", ...]`), but they must be downloaded and extracted manually into `<train_data>/lasot/` and
  `<train_data>/trackingnet/` (they are too large for a Colab runtime).
- **Training: a full stage-1 run is long** (30 M samples; ≈ 17 days on an RTX 3060 laptop GPU), see §8.6.

### Results across GPUs

The same environment gives **bit-identical** results only on the **same GPU type**. Different GPUs use different
numerical kernels; in addition, on Ampere and newer GPUs (RTX 3000/4000, A100, ...) PyTorch computes convolutions in
TF32 by default, while older GPUs such as the T4 (Colab) compute in full FP32. Tracking feeds every frame into the
next, so tiny numerical differences can grow.

Example (`bull`, online, pos, 15+15 steps, lr 1e-5, interval 100; legacy mAP):

| Run | legacy mAP |
|---|---|
| RTX 3060 (TF32 on, default), 8 seeds | 0.521 ± 0.001 |
| RTX 3060, TF32 off (`NVIDIA_TF32_OVERRIDE=0`) | 0.458 |
| Tesla T4 (Colab) | 0.432 |
| Baseline STARK-ST without fine-tuning, RTX 3060, TF32 on / off | 0.476 / 0.479 |

The baseline is insensitive to this, but online fine-tuning is not: in this sequence both runs are identical up to
frame 1599; at frame 1600 the tracker is on the wrong object and, depending on tiny numerical differences, its score is
either 0.19 (no update) or 1.00 (template update **and** 15 fine-tuning steps on the wrong object, after which the
tracker stays on the distractor). **Therefore: run all experiments that are compared with each other on the same GPU
type, and report the GPU.** Seeds alone do not capture this variability.

## 12. Changes from the earlier research code

This repository is a cleaned-up version of earlier, unpublished research code. The main bugs that were fixed:

| # | Earlier behaviour | In this repository |
|---|---|---|
| 1 | `STARK_FT_MODE=all` was undefined; since the tracker only recognised `online`, **online fine-tuning never ran** in experiments named "online". | Modes are `none / init / online`; an invalid value raises an error. |
| 2 | The negative region was shifted by `2·max(w,h)` from the GT; since the half side of the search crop is `2.5·sqrt(w·h)`, **the negative crop contained the target** for targets with aspect ratio below ≈ 1.56; clamping to the image border made it worse. | The shift is computed from the crop size; tested on 20,000 random cases that the target lies completely outside the crop. |
| 3 | The evaluation wrapper passed the full GT of the test sequence to the tracker (future frames without the target were picked for negatives): **test label leakage**. | Removed. The tracker only sees the first-frame box. |
| 4 | No fixed seed; the jitter differed between runs. | `seed` parameter; results are exactly reproducible with the same seed (verified). |
| 5 | `coco_eval.py` excluded frames without the target and applied the score threshold before computing mAP. | Standard COCO usage + P/R/F1 at a threshold + F-max threshold; the old computation is kept as `legacy_*`. |
| 6 | `Preprocessor` converted images to tensors via `tolist()` (very slow). | The original STARK version (`torch.tensor(ndarray)`). |
| 7 | `vot evaluate` also ran the `redetection` experiment of the VOT-LT2020 stack (~2× time). | Only `longterm` by default; can be enabled with `run_redetection=True`. |
| 8 | Paths were hard-coded absolute paths of the development machine; the Vast.ai setup script moved the sequences into the wrong workspace. | No hard-coded paths; every experiment creates its own VOT workspace. |
| 9 | The model downloaded the ImageNet ResNet weights at every start (then overwrote them with the checkpoint). | The download is skipped. |
| 10 | A separate entry file for every parameter combination (~40 files) and dozens of `run_*.py` scripts. | A single entry point: `ExperimentConfig` + `vot_entry.py`. |

**Verification:** with `ft_mode="none"`, the tracker output is bit-identical to the original `STARK_ST` over 150
frames (same checkpoint). STARK-S also gives the same results as the earlier code.

The differences from the original STARK **training** code are listed in [§8.7](#87-differences-from-the-original-stark-training-code).

## 13. License and citation

STARK (ICCV2021):

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```

### Previous work

This repository continues our previous work:

> K. Bal, A. Uslu, A. C. Kılcı and B. Günsel, "Cross-Domain Video Object Detection," *2026 34th Signal Processing and Communications Applications Conference (SIU)*, 2026, pp. 1–4, doi: [10.1109/SIU71813.2026.11636768](https://doi.org/10.1109/SIU71813.2026.11636768).

```bibtex
@inproceedings{bal2026crossdomain,
  author={Bal, Kerem and Uslu, Alper and K{\i}lc{\i}, A. Caner and G{\"u}nsel, Bilge},
  title={Cross-Domain Video Object Detection},
  booktitle={2026 34th Signal Processing and Communications Applications Conference (SIU)},
  year={2026},
  pages={1--4},
  doi={10.1109/SIU71813.2026.11636768}
}
```
