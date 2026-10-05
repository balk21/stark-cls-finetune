# stark-cls-finetune

**English** | [Türkçe](README.tr.md)

Test-time fine-tuning of the confidence head of the [STARK-ST](https://github.com/researchmm/Stark) tracker
(ICCV 2021), evaluated on **VOT-LT2020** with detection metrics (**mAP, AP50, AP75**).

**Why.** In every frame STARK-ST outputs a box and a confidence score. The score says whether the target is visible
and decides when the template is updated, so in long-term tracking a wrong score is as harmful as a wrong box. The
score comes from a small head (`cls_head`) that is trained once for all objects. This repo trains that head for a few
steps **on the target of each video** while tracking (everything else stays frozen) and measures the effect.

**What you can do with it**

| Notebook | Purpose |
|---|---|
| `00_setup_colab` / `00_setup` | Prepare the environment once (Colab / Vast.ai or your own machine) |
| `01_run_experiment` | Track VOT-LT2020 sequences with or without fine-tuning and compute the metrics |
| `02_compare` | Put experiments side by side |
| `03_train` | Retrain STARK-ST itself on the datasets you choose (GOT-10k, COCO or both) |

Everything runs from these notebooks; every notebook's first cell prepares what it needs.

## 1. Setup

**GPU:** NVIDIA with ≥ 8 GB, e.g. Colab **A100** / L4 or an RTX 3000 / 4000 card. Not RTX 5000 (not supported by the
PyTorch version) and preferably not T4: it has no TF32, so its results differ slightly. **Compare only runs from the
same GPU type** ([why](docs/details.md#results-across-gpus)).

**Google Colab** (recommended)
1. Open [`00_setup_colab.ipynb`](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/00_setup_colab.ipynb)
   and choose *Runtime → Change runtime type → A100*.
2. Run all cells (first time 10–15 min). This creates the conda environment `vot1` (the same packages on every
   machine, so results are reproducible), downloads the STARK-ST101 checkpoint and tracks 50 frames as a test
   (mean IoU > 0.5 means the setup works).

Colab deletes the machine after every session, so everything is kept in Google Drive under `MyDrive/LOKAP/`
(environment, checkpoints, results, training runs). Later sessions restore it in ~3 min. To use another folder,
change `DRIVE_ROOT` in the first cell of every notebook.

**Vast.ai or your own computer** (Linux with conda and Jupyter)
```bash
git clone https://github.com/balk21/stark-cls-finetune.git
```
Open `notebooks/00_setup.ipynb` and run all cells (the environment takes 10–20 min once).

## 2. Run an experiment: `01_run_experiment.ipynb`

One experiment = STARK-ST tracks the chosen sequences with one set of parameters (through the official
vot-toolkit), then the metrics and plots are computed.

**Change only the `PARAMS` cell (step 1):**

| You want | Set |
|---|---|
| Plain STARK-ST (baseline) | `ft_mode="none"` |
| Fine-tuning on the first frame only | `ft_mode="init"` |
| Fine-tuning on the first frame and at every template update | `ft_mode="online"` |
| Training samples: positive only / positive + negative | `ft_samples="pos"` / `"posneg"` |
| How strong the fine-tuning is | `ft_lr`, `ft_epochs_init` (steps on frame 1), `ft_epochs_online` (steps per update) |
| Sequences | `sequences=["bull", "ballet"]` or `"all"` (all 50) |
| STARK-ST50 instead of ST101 | `model_config="baseline"` |
| A model you trained ([§4](#4-train-stark-st-03_trainipynb)) | `checkpoint="train:<stage-2 run name>"` |

Leave the rest at the defaults ([all parameters](docs/details.md#4-parameters)). Then run the cells from top to
bottom: the check cell shows the output folder and what will be downloaded, step 2 runs, step 3 shows the results.

- A sequence is downloaded the first time it is used (`bull` 58 MB; all 50: 17.6 GB) and then kept (on Colab: on Drive).
- Time: ~30 ms per frame on an RTX 3060, i.e. `bull` ~1 min, all 50 sequences ~2 h.
- Interrupted? Run the cell again; finished sequences are skipped. The output folder is named after the parameters,
  so a different parameter set never overwrites another experiment.

**Results** are in `outputs/<experiment>/` (on Colab `MyDrive/LOKAP/outputs/`):

| File | Content |
|---|---|
| `metrics/summary.txt` | mAP / AP50 / AP75 first, then precision / recall / F |
| `metrics/metrics.xlsx` | all metrics, also per sequence |
| `plots/<seq>/iou_conf.png` | IoU and score per frame, with the template / fine-tuning updates |
| `predictions/<seq>/` | box and score of every frame |

## 3. Compare experiments: `02_compare.ipynb`

Put the experiment names into `NAMES = [...]` (the first one is the reference) and run the cells. You get the
metrics side by side, the parameter differences and the per-sequence differences, also as `comparison.xlsx`.

## 4. Train STARK-ST: `03_train.ipynb`

The official STARK-ST weights were trained on four datasets together (LaSOT, GOT-10k, COCO, TrackingNet). This
notebook trains STARK-ST with the original procedure on the datasets **you** choose, e.g. COCO only, so that the
effect of the training data can be studied. STARK-ST is trained in two stages:

| Stage | Trains | Length | Starts from |
|---|---|---|---|
| 1 | backbone, transformer, box head | 500 epochs | ImageNet backbone |
| 2 | classification head only | 50 epochs | a finished stage-1 run |

The original used 8 GPUs (128 samples per step). Here one GPU adds up 8 × 16 samples before each step (gradient
accumulation), which gives the same update, so the training is unchanged.

### 4.1 Data (once)

- **COCO:** nothing to do; it is downloaded automatically (19.6 GB, kept on Drive).
- **GOT-10k:** needs a free registration at [got-10k.aitestunion.com](http://got-10k.aitestunion.com/downloads);
  the download links arrive by e-mail. On Colab:
  1. Open the `full_data.zip` link → *Add shortcut to Drive*.
  2. Right-click the shortcut → *Make a copy* (uses 70.7 GB of Drive). Why: the shared file often hits Google's
     daily download limit; your own copy does not.
  3. Move the copy into `MyDrive/LOKAP/train_archives/got10k/` (step 2 of the notebook creates this folder) and
     delete the shortcut.

  On Vast.ai / your computer, put the archives into `data/train/_archives/got10k/`.

At the start of every Colab session the archives are extracted to the runtime's local disk (GOT-10k ≈ 35 min),
because reading the images from Drive during training would be far slower.

### 4.2 Train

**Change only these lines of the `TRAIN` cell (step 1):**

| Goal | `stage` | `datasets` | `init` | `val_datasets` |
|---|---|---|---|---|
| COCO, stage 1 | `1` | `["coco"]` | `None` | `[]` |
| COCO, stage 2 | `2` | `["coco"]` | `"st101_stage1_coco_s42"` | `[]` |
| GOT-10k, stage 1 | `1` | `["got10k"]` | `None` | `["got10k"]` |
| GOT-10k, stage 2 | `2` | `["got10k"]` | `"st101_stage1_got10k_s42"` | `["got10k"]` |
| GOT-10k + COCO, stage 1 | `1` | `["got10k", "coco"]` | `None` | `["got10k"]` |
| GOT-10k + COCO, stage 2 | `2` | `["got10k", "coco"]` | `"st101_stage1_got10k+coco_s42"` | `["got10k"]` |

- `init` (stage 2) is the name of the finished stage-1 run; it is printed during training and is the folder name in
  `MyDrive/LOKAP/training/`. `init="official"` would start from STARK's own weights, which saw all four datasets.
- `val_datasets=["got10k"]` checks the loss on GOT-10k videos that are not used for training; use `[]` when GOT-10k
  is not prepared.
- STARK-ST50: `model_config="baseline"` (run names then start with `st50_`).

Then run step 2 (prepares the session and the data, prints the plan: run name, epochs, initial weights) and
step 3 (trains). If the session ends, run steps 2 and 3 again: training continues from the last finished epoch,
with exactly the same result as without the interruption. Every run has its own folder `MyDrive/LOKAP/training/<run name>/`.

**Time:** stage 1 reads 30 M samples, stage 2 3 M. On an RTX 3060: stage 2 ≈ 1 day, stage 1 ≈ 17 days; an A100 is
several times faster. The progress line shows the remaining time after the first minutes; check it before you
commit to a stage-1 run.

**GOT-10k only, without stage 1:** STARK published weights trained on GOT-10k alone:
`model_config="baseline_R101_got10k_only", datasets=["got10k_full"], stage=2, init="official"`. They used all
9 335 GOT-10k videos, including the 1 000 that overlap with VOT and are left out by `got10k`.

### 4.3 Evaluate the trained model

In `01_run_experiment.ipynb`: `checkpoint="train:<stage-2 run name>"` and the same `model_config`.

## 5. Metrics

- **mAP, AP50, AP75** (primary): standard COCO detection metrics; every frame is an image. Frames without the
  target count too (a box there is a false positive). No score threshold is applied.
- **F at the best threshold** (`precision_opt`, `recall_opt`, `F_opt`): the VOT-LT protocol, i.e. the score threshold
  that maximises F over all sequences.
- `legacy_*`: the computation of the earlier `coco_eval.py`, only for comparing with old results.

Exact definitions: [docs/details.md](docs/details.md#6-metrics).

## 6. Good to know

- vot-toolkit prints "A newer version of the VOT toolkit is available": ignore it, the repo needs version 0.5.3.
- Training runs out of GPU memory → `micro_batch=8` (same training, only slower). The runtime crashes (RAM) →
  `num_workers=4`.
- More: [troubleshooting](docs/details.md#9-troubleshooting), [command line](docs/details.md#8-command-line-cli),
  [known limitations](docs/details.md#10-known-limitations-and-open-issues),
  [changes from the earlier code](docs/details.md#11-changes-from-the-earlier-research-code).

## 7. Citation

This repo builds on STARK:

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```

and continues our previous work:

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
