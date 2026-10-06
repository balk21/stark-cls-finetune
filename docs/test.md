# Test: test-time fine-tuning on VOT-LT2020

STARK-ST is run on VOT-LT2020 with vot-toolkit (0.5.3), with or without test-time fine-tuning of its classification
head, and evaluated with COCO-style detection metrics.

**English** | [Türkçe](test.tr.md)

- [Method](#method)
- [Running](#running)
- [Common settings](#common-settings)
- [VOT-LT2020 sequences](#vot-lt2020-sequences)
- [Parameters](#parameters)
- [Outputs](#outputs)
- [Metrics](#metrics)
- [Limitations](#limitations)
- [Results across GPUs](#results-across-gpus)
- [Changes from the earlier research code](#changes-from-the-earlier-research-code)
- [Troubleshooting](#troubleshooting)

## Method

STARK-ST outputs a **confidence score** in every frame: the sigmoid of a 3-layer MLP (**cls_head**) on the
transformer decoder output. It decides the **template update** (every `update_interval` frames, if the score is
above `update_conf_thr`) and, in this evaluation, whether the tracker reports the target as **found**.

In base training cls_head learns a generic "target present / absent" decision. Here it is reset to its base weights at
the start of every video and briefly trained for that video's target. Backbone, transformer and box head are frozen.

```
frame + target box
  ├─ positive box = the box (+ the stage-2 training jitter if ft_pos_jitter)        -> label 1
  └─ negative box = a region of the same frame that does not contain the target    -> label 0  (posneg only)
        │  each box: search crop -> backbone -> with the templates -> transformer -> cls_head
        ▼
  BCEWithLogitsLoss -> AdamW (ft_lr, ft_weight_decay) -> gradient clipping (ft_grad_clip_norm)
  (1 epoch = 1 optimisation step; a new optimiser for every fine-tuning session)
```

| `ft_mode` | When | Box |
|---|---|---|
| `none` | never (plain STARK-ST) | — |
| `init` | first frame, `ft_epochs_init` steps | ground truth of the first frame |
| `online` | first frame + `ft_epochs_online` steps at every template update | first frame: ground truth; later: the tracker's own prediction |

| `ft_samples` | Samples |
|---|---|
| `pos` | positive only |
| `posneg` | positive + a negative region of the same frame, shifted so that it lies completely outside the target's search crop (8 directions tried; the one most inside the image is used). Provisional until a better negative sampling is found. |

## Running

```bash
python -m stark_ft download-checkpoints --model-config baseline_R101   # official STARK-ST101 (also: baseline = ST50)
python -m stark_ft smoke --frames 50                                   # quick check without vot-toolkit (fetches ballet)

python -m stark_ft test --set ft_mode=none                             # plain STARK-ST
python -m stark_ft test --config configs/test_example.yaml --set ft_samples=posneg --set 'sequences=[bull]'
python -m stark_ft test --set checkpoint=train:<run name>              # a model trained with stark_ft train
python -m stark_ft analyze outputs/<experiment> --score-thr 0.5        # metrics again, without tracking
python -m stark_ft compare <experiment 1> <experiment 2> --out comparison.xlsx --plot comparison.png
```

Notebooks: `notebooks/test.ipynb`, `notebooks/compare.ipynb`. All 50 sequences (≈ 215 k frames) take ≈ 2 h on an
RTX 3060 laptop GPU.

- `show` (notebook: `nb.describe`) prints the output folder, the checkpoint and which sequences will be downloaded.
- Running the same `name` with the same parameters again **resumes**: completed sequences are skipped (that is why
  `vot evaluate` runs without `-f`; every experiment has its own VOT workspace). `vot analysis` is not used; metrics
  are always computed from the raw results.
- The same `name` with different parameters, or with a different version of the tracking code (`lib/`,
  `model_configs/`, the tracker entry point), stops with an error; use another `name` or `--overwrite`.
  The `eval_*` parameters can be changed freely (`analyze`).

## Common settings

| You want | Set |
|---|---|
| Plain STARK-ST (baseline) | `ft_mode="none"` |
| Fine-tuning on the first frame only | `ft_mode="init"` |
| Fine-tuning on the first frame and at every template update | `ft_mode="online"` |
| Positive samples only / positive + negative | `ft_samples="pos"` / `"posneg"` |
| Strength of the fine-tuning | `ft_lr`, `ft_epochs_init` (steps on frame 1), `ft_epochs_online` (steps per update) |
| Sequences | `sequences=["bull", "ballet"]` or `"all"` |
| STARK-ST50 instead of ST101 | `model_config="baseline"` |
| A model trained with this repository | `checkpoint="train:<stage-2 run name>"` (same `model_config`) |

## VOT-LT2020 sequences

Nothing has to be downloaded in advance. When a test starts, the sequences it uses that are not in the `dataset`
folder yet are downloaded one by one from the official VOT server (the VOT-LT2019 sequences, which the VOT-LT2020
stack uses), checked against the official SHA-1 checksums and written exactly as vot-toolkit writes them
(`stark_ft/test/vot_data.py`). `sequences=["bull"]` downloads only bull (58 MB); `"all"` all 50 (17.6 GB, 20–60 min).
In advance: `python -m stark_ft download-dataset [--sequences bull ballet]` (without `--sequences`: all 50).

If `dataset_cache` is set (`configs/paths.local.yaml`), a copy of every downloaded sequence (`<sequence>.tar`) is kept
there and used instead of the server next time (on Colab: the Drive folder `cache/votlt2019_sequences/`). Sequences
already in the dataset folder are only read; the `list.txt` vot-toolkit needs is written into each experiment's own
`vot_workspace/`.

## Parameters

`ExperimentConfig` in `stark_ft/test/config.py`. Notebook: `PARAMS = dict(...)`; CLI: `--config file.yaml` and/or
`--set key=value`.

| Parameter | Default | Description |
|---|---|---|
| `name` | `None` | Output folder `<outputs>/<name>/`. `None`: generated, e.g. `st101_online_pos_lr0.0001_i15_o1_int100_s0`. |
| `model` | `"stark_st"` | `"stark_st"` or `"stark_s"` (no score; reports 1.0; needs `ft_mode="none"`). |
| `model_config` | `"baseline_R101"` | YAML in `model_configs/stark_st2/` (`baseline_R101`, `baseline`, `*_got10k_only`) or `model_configs/stark_s/`. |
| `checkpoint` | `None` | `None`: official file. A file name is looked up in `<checkpoints>/stark_st2/<model_config>/`; a value with `/` is a path; `"train:<run name>"` is a training run's `final.pth.tar`. |
| `sequences` | `"all"` | `"all"` (50 sequences) or a list, e.g. `["bull", "ballet"]`; missing ones are downloaded. |
| `update_interval` | `100` | Template update attempt every N frames; `99999` = never. (`TEST.UPDATE_INTERVALS` of the YAMLs is not used.) |
| `update_conf_thr` | `0.5` | Update only if the score is greater than this (STARK: 0.5). |
| `max_template_updates` | `-1` | Per sequence; `-1` = unlimited. |
| `ft_mode` | `"online"` | `"none"`, `"init"`, `"online"` (see [Method](#method)). |
| `ft_samples` | `"pos"` | `"pos"` or `"posneg"`. |
| `ft_lr` | `1e-4` | AdamW learning rate (stage-2 training: 1e-4). |
| `ft_epochs_init` | `15` | Steps on the first frame; `0` = none. |
| `ft_epochs_online` | `1` | Steps per online fine-tuning. |
| `ft_weight_decay` | `1e-4` | As in training. |
| `ft_grad_clip_norm` | `0.1` | As in training. |
| `ft_pos_jitter` | `True` | Jitter the positive box as in stage-2 training; `False`: exact box. |
| `ft_center_jitter` | `4.5` | Centre shift within a window of `sqrt(w·h)·4.5` (stage 2: 4.5). |
| `ft_scale_jitter` | `0.5` | Size × `exp(N(0,1)·0.5)` (stage 2: 0.5). |
| `max_ft_updates` | `-1` | Online fine-tunings per sequence; `-1` = unlimited. |
| `seed` | `0` | Same seed and parameters give exactly the same result (on the same GPU type). |
| `eval_score_thr` | `0.35` | Fixed threshold for P / R / F1 (does not affect mAP or the F-max threshold). |
| `eval_iou_thr` | `0.5` | IoU of a correct detection (fixed threshold and F-max search). |
| `eval_thr_resolution` | `100` | Candidate thresholds in the F-max search (vot-toolkit: 100). |
| `run_redetection` | `False` | Also run the VOT-LT2020 `redetection` experiment (~2× time; not used in the metrics). |
| `tracker_timeout` | `300` | Seconds per tracker response, including model loading. |

## Outputs

`<outputs>/<experiment name>/`:

| File | Content |
|---|---|
| `experiment.json` | Parameters, checkpoint, dataset, sequences, git commit, code hash, date |
| `run.log`, `run_status.json` | vot-toolkit output; completed / missing sequences (tracker errors: `vot_workspace/logs/`) |
| `predictions/<seq>/<seq>_001.txt` | Box per frame `x,y,w,h` (first line `1`: init frame, VOT format) |
| `predictions/<seq>/<seq>_001_confidence.value`, `_time.value` | Score and time per frame |
| `predictions/<seq>/frames.csv` | `frame, x, y, w, h, conf, time, gt_visible, gt_x, gt_y, gt_w, gt_h, iou` |
| `tracker_logs/<seq>/finetune_loss.txt` | `frame, session, epoch, loss, pos_prob, neg_prob, n_pos, n_neg, neg_coverage` (one row per step; probabilities before the update) |
| `tracker_logs/<seq>/events.txt` | `frame, event, conf_score` (`template_update`, `ft_update`) |
| `plots/<seq>/iou_conf.png` | IoU and score vs. frame (grey: target absent; red: template update; green: fine-tuning; black: fixed threshold; purple: F-max threshold) |
| `plots/<seq>/finetune_loss.png` | Loss and positive / negative probabilities per step |
| `metrics/summary.txt`, `metrics.xlsx`, `metrics.json` | Summary and per-sequence metrics |
| `metrics/f_curve.csv`, `f_curve.png` | P / R / F vs. threshold, PR curve |
| `vot_workspace/` | The experiment's own VOT workspace (raw `results/`) |

Frame 0 is the first (init) frame (= line number − 1 in the VOT files); it is not evaluated.

## Metrics

Detection-style: every frame is an image with (at most) one object; computed with `pycocotools`
(`stark_ft/test/evaluation.py`). **mAP, AP50 and AP75 are the primary metrics.**

**mAP / AP50 / AP75** (standard COCO): mAP = COCO's AP (`stats[0]`, IoU 0.50:0.95), AP50 / AP75 at one IoU threshold.
All frames except the first are evaluated; frames without the target are images without annotations, so every
prediction there is a false positive. No score threshold is applied (mAP already ranks by score). "Mean over sequences"
averages the per-sequence values; "pooled" treats all frames as one dataset.

**Fixed threshold** (`eval_score_thr`): the tracker "claims the target" if score ≥ threshold.

| | claims (score ≥ thr) | does not |
|---|---|---|
| target visible, IoU ≥ `eval_iou_thr` | TP | FN |
| target visible, IoU < `eval_iou_thr` | FP + FN | FN |
| target absent | FP | TN |

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`, `absent_reject_rate` = TN / frames without the
target. Same matching rule as COCOeval (verified).

**F-max threshold** (VOT-LT protocol, as `vot/analysis/tpr.py` of vot-toolkit 0.5.3): candidate thresholds are taken
from the pooled scores exactly as vot-toolkit's `determine_thresholds` (`eval_thr_resolution` values); at every
threshold P and R are computed per sequence with the counting above (vot-toolkit uses IoU-weighted P/R), averaged
over sequences, and F = 2PR/(P+R) of the averages; the threshold with the largest F is used for all sequences
(`precision_opt / recall_opt / F_opt`). The threshold is chosen with the ground truth, as in VOT-LT reporting; the
tracker does not know it.

**Other:** `mean_iou_visible` (mean IoU where the target is visible); `legacy_mAP / AP50 / AP75` reproduce the earlier
`coco_eval.py` exactly (frames without the target excluded, predictions below the threshold dropped, integer
coordinates) for comparison with old results only.

## Limitations

- **`pos` inflates the score in general.** With only label 1, BCE is easiest to minimise by raising the output for
  every input; AdamW moves ≈ `ft_lr` per step even when the loss is tiny. The score also rises where the target is
  absent (`absent_reject_rate` drops) and template updates happen more often.
- **`posneg` negatives are provisional.** A region of the same frame may contain similar objects (high `neg_prob` in
  `finetune_loss.txt`). Only the `posneg` branch in `lib/test/tracker/ft_sampling.py` / `stark_st_ft.py` has to change.
- **`online` trains on the tracker's own prediction**, and the update decision comes from the head being fine-tuned:
  a possible self-reinforcing loop.
- **`ft_lr` / `ft_epochs_*` are empirical.** Choosing them on the test sequences makes results optimistic.
- vot-toolkit restarts the tracker process (and reloads the model, ~3–4 s) for every sequence.

## Results across GPUs

Results are bit-identical only on the **same GPU type**. GPUs use different numerical kernels, and Ampere and newer
GPUs (RTX 3000/4000, A100, L4) compute convolutions in TF32 by default while e.g. a T4 uses FP32. Tracking feeds
every frame into the next, so tiny differences can grow. Example (`bull`, online, pos, 15+15 steps, lr 1e-5,
interval 100; legacy mAP):

| Run | legacy mAP |
|---|---|
| RTX 3060 (TF32, default), 8 seeds | 0.521 ± 0.001 |
| RTX 3060, TF32 off (`NVIDIA_TF32_OVERRIDE=0`) | 0.458 |
| Tesla T4 | 0.432 |
| STARK-ST without fine-tuning, RTX 3060, TF32 on / off | 0.476 / 0.479 |

The runs are identical up to frame 1599; at frame 1600 the tracker is on a distractor and its score is either 0.19
(no update) or 1.00 (template update and 15 fine-tuning steps on the distractor). **Run all compared experiments on
the same GPU type and report it.**

## Changes from the earlier research code

| # | Earlier behaviour | Here |
|---|---|---|
| 1 | `STARK_FT_MODE=all` was undefined, so online fine-tuning never ran in the "online" experiments | `ft_mode` = `none / init / online`; invalid values raise an error |
| 2 | The negative region was shifted by `2·max(w,h)`; for aspect ratios below ≈ 1.56 the negative crop contained the target | Shift computed from the crop size; tested on 20 000 random cases |
| 3 | The full ground truth of the test sequence was passed to the tracker (test label leakage) | Removed; the tracker only sees the first-frame box |
| 4 | No fixed seed | `seed`; exactly reproducible |
| 5 | `coco_eval.py` excluded frames without the target and thresholded before mAP | Standard COCO + fixed threshold + F-max threshold; old computation kept as `legacy_*` |
| 6 | Images were converted to tensors via `tolist()` (slow) | Original STARK version |
| 7 | The `redetection` experiment always ran (~2× time) | Only `longterm` by default |
| 8 | Hard-coded paths of the development machine | No hard-coded paths; one VOT workspace per experiment |
| 9 | ImageNet weights downloaded at every start | Skipped |
| 10 | One entry file per parameter combination (~40) | One entry point (`ExperimentConfig` + `vot_entry.py`) |

With `ft_mode="none"` the output is bit-identical to the original `STARK_ST` (150 frames, same checkpoint).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Checkpoint not found` | `download-checkpoints`, or put the file where the message says. |
| Google Drive "quota exceeded" for a checkpoint | Download it in a browser from the link in the message and put it where the message says. |
| `Unknown sequence(s): [...]` | A typo in `sequences`; the message lists the 50 VOT-LT2020 names. |
| VOT sequence download fails / `Checksum mismatch` | Network or server problem; run again (finished sequences are kept, an interrupted file continues). |
| `Not enough disk space for N VOT sequence(s)` | Free disk space, or test on fewer sequences. |
| `Completed sequences: 47/50` | A failing sequence is skipped; see `vot_workspace/logs/`. Running again retries only the missing ones. |
| `no kernel image is available` / sm_120 | GPU not supported by PyTorch 2.4.1 (RTX 5000 series); use an RTX 3000/4000-class GPU. |
| "A newer version of the VOT toolkit is available" | Ignore; tested with vot-toolkit 0.5.3. |
