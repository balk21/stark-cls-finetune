# Test: test-time fine-tuning on VOT-LT2020 and GOT-10k

STARK-ST is run on VOT-LT2020 (with vot-toolkit 0.5.3) or on a GOT-10k split, with or without test-time fine-tuning
of its classification head, and evaluated with COCO-style detection metrics (GOT-10k: also AO / SR).

**English** | [Türkçe](test.tr.md)

- [Method](#method)
- [Running](#running)
- [Datasets](#datasets)
- [Common settings](#common-settings)
- [Weights](#weights)
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
transformer decoder output. It decides the **template update** and, in this evaluation, whether the tracker reports
the target as **found**.

| `update_mode` | Template update (N = `update_interval`) |
|---|---|
| `stark` (STARK) | At frames N, 2N, 3N, ...: the current frame, if its score is above `update_conf_thr`. |
| `max` | At frames 2N, 3N, ...: the frame with the highest score of the last N frames, among the candidates: score above `update_conf_thr` and IoU ≥ `update_iou_thr` with the box of the previous candidate (for the first frame of the N: the box of the frame before). No candidate: no update. Frames 1 ... N keep the first frame's template. |

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
| `online` | first frame + `ft_epochs_online` steps at every template update, on the frame the template is taken from | first frame: ground truth; later: the tracker's own prediction |

| `ft_samples` | Samples |
|---|---|
| `pos` | positive only |
| `posneg` | positive + a negative region of the same frame, shifted so that it lies completely outside the target's search crop (8 directions tried; the one most inside the image is used). Provisional until a better negative sampling is found. |

## Running

```bash
python -m stark_ft weights                                             # official weights and runs trained here
python -m stark_ft smoke --frames 50                                   # optional quick check (fetches ballet)

python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=none    # plain STARK-ST, official weights
python -m stark_ft test --config configs/test_example.yaml --set ft_samples=posneg --set 'sequences=[bull]'
python -m stark_ft test --set 'sequences=[bull]' --set weights=<run name>   # a model trained with stark_ft train
# STARK's GOT-10k-only weights on all GOT-10k val sequences / on the GOT-10k test split / on VOT-LT2020
python -m stark_ft test --set model_config=baseline_R101_got10k_only --set dataset=got10k_val --set sequences=all \
    --set ft_mode=none --set update_interval=200
python -m stark_ft test --set model_config=baseline_R101_got10k_only --set dataset=got10k_test --set sequences=all \
    --set ft_mode=none --set update_interval=200
python -m stark_ft test --set model_config=baseline_R101_got10k_only --set 'sequences=[bull]' --set ft_mode=none
python -m stark_ft analyze outputs/<experiment> --score-thr 0.5        # metrics again, without tracking
python -m stark_ft compare <experiment 1> <experiment 2> --out comparison.xlsx --plot comparison.png
```

Notebooks: `notebooks/test.ipynb`, `notebooks/compare.ipynb`. All 50 sequences (≈ 215 k frames) take ≈ 2 h on an
RTX 3060 laptop GPU.

- `show` (notebook: `nb.describe`) prints the output folder, the weights and which sequences will be downloaded.
- Running the same `name` with the same parameters again **resumes**: completed sequences are skipped (that is why
  `vot evaluate` runs without `-f`; every experiment has its own VOT workspace). `vot analysis` is not used; metrics
  are always computed from the raw results.
- The same `name` with different parameters, or with a different version of the tracking code (`lib/`,
  `model_configs/`, the tracker entry point), stops with an error; use another `name` or `--overwrite`.
  The `eval_*` parameters can be changed freely (`analyze`).

## Datasets

| `dataset` | Sequences | How it is run | Metrics |
|---|---|---|---|
| `"votlt2020"` (default) | the 50 VOT-LT2020 sequences, downloaded on demand ([below](#vot-lt2020-sequences)) | vot-toolkit, `longterm` experiment | mAP / AP50 / AP75, F-max threshold, ... ([Metrics](#metrics)) |
| `"got10k_val"` | the 180 GOT-10k validation sequences | GOT-10k protocol: one pass per sequence, no restarts (without vot-toolkit; the model is loaded once) | **AO, SR0.50, SR0.75** as the GOT-10k toolkit, and the metrics above |
| `"got10k_test"` | the 180 GOT-10k test sequences (only the first-frame box is public) | as above | none locally: `got10k_submission.zip` for the [GOT-10k server](http://got-10k.aitestunion.com/submit_instructions) |
| `"got10k_train"` | the 9 335 GOT-10k train sequences | as above | as val (the `*_got10k_only` weights were trained on these videos) |

GOT-10k sequence names are `GOT-10k_Val_000001` ... (`sequences="all"` or a list). A GOT-10k split is extracted on its
first use from the same GOT-10k archives as the training data (`<archives>/got10k/`, e.g. `full_data.zip`, which
contains train, val and test; see [train.md](train.md#datasets) and [colab.md](colab.md#got-10k-from-google-drive))
into `<train_data>/got10k/<split>/`. Every sequence's result is written when it is complete, so an interrupted run
continues with the remaining sequences.

STARK's weights trained on GOT-10k only are `model_config="baseline_R101_got10k_only"` (ST101) or
`"baseline_got10k_only"` (ST50) with `weights="official"`; they can be tested on GOT-10k and on VOT-LT2020 alike.
STARK used `update_interval=200` on GOT-10k (100 on VOT-LT).

## Common settings

| You want | Set |
|---|---|
| Plain STARK-ST (baseline) | `ft_mode="none"` |
| Fine-tuning on the first frame only | `ft_mode="init"` |
| Fine-tuning on the first frame and at every template update | `ft_mode="online"` |
| Positive samples only / positive + negative | `ft_samples="pos"` / `"posneg"` |
| Strength of the fine-tuning | `ft_lr`, `ft_epochs_init` (steps on frame 1), `ft_epochs_online` (steps per update) |
| Template update with the best frame of every N frames | `update_mode="max"` (`update_conf_thr`, `update_iou_thr`) |
| Sequences | `sequences=["bull", "ballet"]` or `"all"` |
| GOT-10k instead of VOT-LT2020 | `dataset="got10k_val"` / `"got10k_test"` / `"got10k_train"` |
| STARK's GOT-10k-only weights | `model_config="baseline_R101_got10k_only"` (ST50: `"baseline_got10k_only"`) |
| STARK-ST50 instead of ST101 | `model_config="baseline"` |
| A model trained with this repository | `weights="<stage-2 run name>"` (same `model_config`) |

## Weights

`weights` selects the network weights of a test; the experiment name and `experiment.json` record them.

| `weights` | Weights |
|---|---|
| `"official"` (default) | STARK's published weights for `model_config` (trained on LaSOT + GOT-10k + COCO + TrackingNet; `*_got10k_only`: GOT-10k). Downloaded to `<checkpoints>` automatically when first used. |
| `"<run name>"` | A finished **stage-2** run trained here, e.g. `"st101_coco_stage2"` ([train.md](train.md#our-runs-and-the-official-weights)). It must have the same `model_config`; stage-1 and unfinished runs are refused. |
| file name / path | Any other checkpoint: a file name is looked up in `<checkpoints>/stark_st2/<model_config>/`, a value with `/` is a path. |

`python -m stark_ft weights` (notebook: `nb.list_weights()`) lists them by origin: the official weights, runs trained
here from ImageNet, and runs trained here on top of the official weights. `show` / `nb.describe` prints which weights a
test will use. Experiment names start with the run name for trained weights (`st101_coco_stage2_online_...`) and
with the model for the official ones (`st101_online_...`), so the results of different weights never mix.

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
| `name` | `None` | Output folder `<outputs>/<name>/`. `None`: generated, e.g. `st101_online_pos_lr0.0001_i15_o1_int100_s0` (`max100` with `update_mode="max"`). |
| `model` | `"stark_st"` | `"stark_st"` or `"stark_s"` (no score; reports 1.0; needs `ft_mode="none"`). |
| `model_config` | `"baseline_R101"` | YAML in `model_configs/stark_st2/` (`baseline_R101`, `baseline`, `*_got10k_only`) or `model_configs/stark_s/`. |
| `weights` | `"official"` | `"official"`, a stage-2 run trained here, or a checkpoint file / path (see [Weights](#weights)). (Earlier name: `checkpoint`.) |
| `dataset` | `"votlt2020"` | `"votlt2020"`, `"got10k_val"`, `"got10k_test"` or `"got10k_train"` (see [Datasets](#datasets)). |
| `sequences` | `"all"` | `"all"` or a list, e.g. `["bull", "ballet"]` / `["GOT-10k_Val_000001"]`; missing data is downloaded / extracted. |
| `update_mode` | `"stark"` | `"stark"`: every N-th frame, `"max"`: the best frame of every N frames (see [Method](#method)). |
| `update_interval` | `100` | N; `99999` = no updates. (`TEST.UPDATE_INTERVALS` of the YAMLs is not used.) |
| `update_conf_thr` | `0.5` | Update only if the score is greater than this (STARK: 0.5). |
| `update_iou_thr` | `0.5` | `max` only: minimum IoU with the box of the previous candidate frame; `0` = no IoU check. |
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
| `run_redetection` | `False` | Also run the VOT-LT2020 `redetection` experiment (~2× time; not used in the metrics; VOT only). |
| `tracker_timeout` | `300` | Seconds per tracker response, including model loading. |

## Outputs

`<outputs>/<experiment name>/`:

| File | Content |
|---|---|
| `experiment.json` | Parameters (incl. `weights`), checkpoint file, dataset, sequences, git commit, code hash, date |
| `run.log`, `run_status.json` | vot-toolkit output; completed / missing sequences (tracker errors: `vot_workspace/logs/`) |
| `predictions/<seq>/<seq>_001.txt` | Box per frame `x,y,w,h` (first line `1`: init frame, VOT format) |
| `predictions/<seq>/<seq>_001_confidence.value`, `_time.value` | Score and time per frame |
| `predictions/<seq>/frames.csv` | `frame, x, y, w, h, conf, time, gt_visible, gt_x, gt_y, gt_w, gt_h, iou` |
| `tracker_logs/<seq>/finetune_loss.txt` | `frame, session, epoch, loss, pos_prob, neg_prob, n_pos, n_neg, neg_coverage` (one row per step; probabilities before the update) |
| `tracker_logs/<seq>/events.txt` | `frame, event, conf_score` (`template_update`, `ft_update`): the frame the template was taken from and its score |
| `plots/<seq>/iou_conf.png` | IoU and score vs. frame (grey: target absent; red: template update; green: fine-tuning; black: fixed threshold; purple: F-max threshold) |
| `plots/<seq>/finetune_loss.png` | Loss and positive / negative probabilities per step |
| `metrics/summary.txt`, `metrics.xlsx`, `metrics.json` | Summary and per-sequence metrics |
| `metrics/f_curve.csv`, `f_curve.png` | P / R / F vs. threshold, PR curve |
| `vot_workspace/` | The experiment's own VOT workspace (raw `results/`; VOT-LT2020 only) |
| `got10k_submission.zip` | GOT-10k test split: `<seq>/<seq>_001.txt` and `<seq>/<seq>_time.txt` of every sequence, for the GOT-10k server |

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

**GOT-10k** (`got10k_val`, `got10k_train`): **AO** (average overlap), **SR0.50** and **SR0.75** (fraction of frames
with IoU above 0.5 / 0.75), computed as the GOT-10k toolkit does: the frames after the first in which the target is
visible (`cover.label` > 0), boxes clipped to the image, all frames of all sequences pooled. They are the first line of
`summary.txt` and the `AO`, `SR50`, `SR75` columns (per sequence: that sequence's frames). On GOT-10k the frames without
a visible target are treated as "target absent" in the detection metrics above.

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
| `No training run '...'` / `is a stage-1 run` / `was trained with model_config=...` | Choose a finished stage-2 run from `python -m stark_ft weights`, with its `model_config`. |
| Google Drive "quota exceeded" for the official weights | Download the file in a browser from the link in the message and put it where the message says. |
| `Unknown sequence(s): [...]` | A typo in `sequences`; the message lists the 50 VOT-LT2020 names. |
| VOT sequence download fails / `Checksum mismatch` | Network or server problem; run again (finished sequences are kept, an interrupted file continues). |
| `Not enough disk space for N VOT sequence(s)` | Free disk space, or test on fewer sequences. |
| `Completed sequences: 47/50` | A failing sequence is skipped; see `vot_workspace/logs/`. Running again retries only the missing ones. |
| `no kernel image is available` / sm_120 | GPU not supported by PyTorch 2.4.1 (RTX 5000 series); use an RTX 3000/4000-class GPU. |
| "A newer version of the VOT toolkit is available" | Ignore; tested with vot-toolkit 0.5.3. |
