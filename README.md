# stark-cls-finetune — inference-time fine-tuning of the STARK-ST classification head

**English** | [Türkçe](README.tr.md)

This repository adds **video-specific fine-tuning of the classification head (cls_head)** at test (inference)
time to the **STARK-ST** variant of the [STARK](https://github.com/researchmm/Stark) tracker (ICCV 2021), and
evaluates it on **VOT-LT2020** with **detection-style (COCO)** metrics.

The whole workflow runs from Jupyter notebooks. The code contains no hard-coded file paths; the same repository
runs on a Vast.ai server and on a personal computer.

---

## Contents
1. [Quick start (Vast.ai)](#1-quick-start-vastai)
2. [On your own computer](#2-on-your-own-computer)
3. [Folder layout](#3-folder-layout)
4. [Method](#4-method)
5. [Parameters](#5-parameters)
6. [Outputs and file formats](#6-outputs-and-file-formats)
7. [Metrics](#7-metrics)
8. [Command line (CLI)](#8-command-line-cli)
9. [Troubleshooting](#9-troubleshooting)
10. [Known limitations and open issues](#10-known-limitations-and-open-issues)
11. [Changes from the earlier research code](#11-changes-from-the-earlier-research-code)
12. [License and citation](#12-license-and-citation)

---

## 1. Quick start (Vast.ai)

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

> **Notebook kernel:** any Python 3 kernel works. The notebooks do the actual work in the background with the
> Python of the `vot1` environment (`notebooks/nbhelper.py`). You do not need to change the kernel.

## 2. On your own computer

The same notebooks are used. If the checkpoints or the dataset already exist elsewhere, instead of downloading them,
copy `configs/paths.local.example.yaml` to `configs/paths.local.yaml` and set the paths:

```yaml
checkpoints: /home/user/stark/checkpoints/train   # <checkpoints>/stark_st2/baseline_R101/STARKST_ep0050.pth.tar
dataset: /home/user/vot/votlt2020/sequences        # <dataset>/<sequence>/{color/, groundtruth.txt, sequence}
outputs: /home/user/stark_outputs
```

This file is ignored by git. The same settings can be given with the environment variables
`STARK_CLEAN_CHECKPOINTS`, `STARK_CLEAN_DATASET` and `STARK_CLEAN_OUTPUTS`
(priority: environment variable > `paths.local.yaml` > `paths.yaml`).

**Nothing is written into the dataset folder.** The `list.txt` that vot-toolkit needs is created in each
experiment's own `vot_workspace/` folder, with absolute paths to the sequences.

## 3. Folder layout

```
stark-cls-finetune/
├── notebooks/
│   ├── 00_setup.ipynb            Setup (environment, checkpoint, dataset, smoke test)
│   ├── 01_run_experiment.ipynb   Parameters → run → results
│   ├── 02_compare.ipynb          Comparing experiments
│   └── nbhelper.py               Notebook helpers (standard library only)
├── stark_ft/                     Experiment framework
│   ├── config.py                 ExperimentConfig: ALL parameters and their validation
│   ├── paths.py                  Path resolution
│   ├── runner.py                 VOT workspace setup + `vot evaluate` + collecting results
│   ├── vot_entry.py              Tracker process that vot-toolkit starts for every sequence
│   ├── tracker_factory.py        Builds the tracker object from the parameters
│   ├── evaluation.py             COCO AP, P/R/F1, F-max threshold, legacy COCO
│   ├── analysis.py               Metric tables + plot generation
│   ├── plots.py                  Plots
│   ├── compare.py                Comparing experiments
│   ├── setup_utils.py            Environment check, downloads, smoke test
│   └── __main__.py               CLI (python -m stark_ft ...)
├── lib/                          STARK core (only what inference needs)
│   ├── models/stark/             Network architecture (original STARK)
│   ├── config/                   Model config defaults (original STARK)
│   ├── test/tracker/
│   │   ├── stark_st.py           STARK-ST tracker (original + update bookkeeping)
│   │   ├── stark_st_ft.py        ★ Tracker with fine-tuning
│   │   ├── ft_sampling.py        ★ Positive jitter and negative region generation
│   │   └── stark_s.py            STARK-S tracker
│   └── utils/
├── model_configs/                STARK model YAMLs (stark_st2/, stark_s/)
├── configs/
│   ├── paths.yaml                Default paths
│   ├── paths.local.example.yaml  Template for machine-specific paths
│   └── experiments/example.yaml  Example experiment file for the CLI
├── environment/vot1_environment.yml
├── tests/test_sampling.py
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
| `pos` | Positive sample only. Training with label "1" only pushes the head's score up in general; the score also rises on frames without the target (see §10). |
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
| `checkpoint` | `None` | If `None`, the official file name is used (`STARKST_ep0050.pth.tar` / `STARKS_ep0500.pth.tar`). If only a file name is given, it is looked up in `<checkpoints>/<stark_st2\|stark_s>/<model_config>/`, e.g. `"STARKSTcoco_ep0050.pth.tar"`. A value containing `/` is used as a path (relative paths are resolved against the repository root). |

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
| `eval_score_thr` | `0.35` | **Fixed** threshold: score threshold for the "tracker found the target" decision. Only affects the `precision / recall / F1` columns; does not affect AP or the F-max threshold result. |
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
| `metrics/summary.txt` | Readable summary (first lines: F-max threshold and fixed threshold results) + per-sequence table |
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

### 7.1 COCO AP / AP50 / AP75 (standard usage)

- **All** frames except the first one are evaluated.
- Frames where the target is not visible (GT = NaN) are added as images without annotations. Every prediction on
  these frames counts as a **false positive**.
- **No score threshold is applied**: all predictions are given with their scores. AP already measures the quality
  of the score ranking over all thresholds. A threshold can never increase AP.
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
- `legacy_AP / legacy_AP50 / legacy_AP75`: **exactly** the computation of the earlier
  `testler/detailed_analysis/coco_eval.py` script (frames without the target excluded, predictions below the
  threshold dropped, coordinates rounded to integers). Only for comparison with old results; it does not penalise
  false detections on frames without the target. Verified to match the old script on 5 sequences.
- **Mean over sequences:** the mean of each sequence's metric (same idea as the old `MEAN` row). Exception: `F1` and
  `F_opt` are computed from the mean P and R, following the VOT definition.
- **Pooled:** the frames of all sequences as a single dataset; longer sequences get more weight.

## 8. Command line (CLI)

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
python -m tests.test_sampling                              # tests of the negative-sample geometry
```

`--set` values are parsed as YAML: `1e-4` → number, `true` → bool, `[a, b]` → list.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Conda environment 'vot1' not found` | Step 1 of `00_setup.ipynb`. If the environment is elsewhere, set `VOT1_PYTHON=/path/envs/vot1/bin/python`. |
| `Checkpoint not found` | Step 3 of `00_setup.ipynb`, or put the file at the location shown in the error message. |
| Google Drive "quota exceeded" | Download the file in a browser from the link in the error message and put it in the folder shown there. |
| `No VOT sequences found in the dataset folder` | Step 4 of `00_setup.ipynb`, or set `dataset` in `configs/paths.local.yaml`. |
| Missing sequences, e.g. `Completed sequences: 47/50` | A failing sequence is skipped and the others keep running. The tracker's error output is in `outputs/<experiment>/vot_workspace/logs/`. Running the same experiment again skips completed sequences and retries the missing ones. |
| `no kernel image is available` / sm_120 warning | The GPU is not supported by the PyTorch in the environment (RTX 5000 series). Use an RTX 3000/4000 series GPU. |
| "A newer version of the VOT toolkit is available" | Ignore it; the repository is tested with vot-toolkit **0.5.3**. Do not upgrade. |

## 10. Known limitations and open issues

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

## 11. Changes from the earlier research code

This repository is a cleaned-up version of earlier, unpublished research code. The main bugs that were fixed:

| # | Earlier behaviour | In this repository |
|---|---|---|
| 1 | `STARK_FT_MODE=all` was undefined; since the tracker only recognised `online`, **online fine-tuning never ran** in experiments named "online". | Modes are `none / init / online`; an invalid value raises an error. |
| 2 | The negative region was shifted by `2·max(w,h)` from the GT; since the half side of the search crop is `2.5·sqrt(w·h)`, **the negative crop contained the target** for targets with aspect ratio below ≈ 1.56; clamping to the image border made it worse. | The shift is computed from the crop size; tested on 20,000 random cases that the target lies completely outside the crop. |
| 3 | The evaluation wrapper passed the full GT of the test sequence to the tracker (future frames without the target were picked for negatives): **test label leakage**. | Removed. The tracker only sees the first-frame box. |
| 4 | A parameter file silently loaded `STARKSTcoco_ep0050.pth.tar` if it existed, so "baseline" results may have used the COCO checkpoint. | The checkpoint is chosen only via the `checkpoint` parameter and recorded in `experiment.json`. |
| 5 | No fixed seed; the jitter differed between runs. | `seed` parameter; results are exactly reproducible with the same seed (verified). |
| 6 | `coco_eval.py` excluded frames without the target and applied the score threshold before computing AP. | Standard COCO usage + P/R/F1 at a threshold + F-max threshold; the old computation is kept as `legacy_*`. |
| 7 | `Preprocessor` converted images to tensors via `tolist()` (very slow). | The original STARK version (`torch.tensor(ndarray)`). |
| 8 | `vot evaluate` also ran the `redetection` experiment of the VOT-LT2020 stack (~2× time). | Only `longterm` by default; can be enabled with `run_redetection=True`. |
| 9 | Paths were hard-coded absolute paths of the development machine; the Vast.ai setup script moved the sequences into the wrong workspace. | No hard-coded paths; every experiment creates its own VOT workspace. |
| 10 | The model downloaded the ImageNet ResNet weights at every start (then overwrote them with the checkpoint). | The download is skipped. |
| 11 | A separate entry file for every parameter combination (~40 files) and dozens of `run_*.py` scripts. | A single entry point: `ExperimentConfig` + `vot_entry.py`. |

**Verification:** with `ft_mode="none"`, the tracker output is bit-identical to the original `STARK_ST` over 150
frames (same checkpoint). STARK-S also gives the same results as the earlier code.

## 12. License and citation

The STARK code is distributed under the MIT license (`LICENSE`). If you use STARK, please cite the original paper:

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```
