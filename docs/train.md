# Training STARK-ST

The original STARK training (`lib/train/`, ported from the official repository) on one GPU, with the original effective
batch size, on any combination of GOT-10k, COCO, LaSOT and TrackingNet.

**English** | [Türkçe](train.tr.md)

- [What is trained](#what-is-trained)
- [Our runs and the official weights](#our-runs-and-the-official-weights)
- [Datasets](#datasets)
- [Running](#running)
- [Typical runs](#typical-runs)
- [Parameters](#parameters)
- [Run folder and resuming](#run-folder-and-resuming)
- [One GPU instead of eight](#one-gpu-instead-of-eight)
- [Differences from the original STARK training code](#differences-from-the-original-stark-training-code)
- [Troubleshooting](#troubleshooting)

## What is trained

| Stage | Trained parameters | Epochs (LR ÷10 at) | Loss | Starts from |
|---|---|---|---|---|
| 1 | backbone + transformer + box head | 500 (400) | GIoU × 2 + L1 × 5 | ImageNet ResNet backbone |
| 2 | classification head only (everything else frozen) | 50 (40) | BCE | stage-1 weights (`init`) |

Both stages (model YAMLs in `model_configs/stark_st1/` and `model_configs/stark_st2/`): 60 000 training samples per
epoch, AdamW with lr 1e-4 (backbone × 0.1) and weight decay 1e-4, gradient clipping at norm 0.1, validation on 10 000
samples every 20 (stage 1) / 10 (stage 2) epochs, deterministic cuDNN. Frame sampling, augmentations and
losses are those of the original code.

## Our runs and the official weights

STARK publishes weights trained on all four datasets (LaSOT, GOT-10k, COCO, TrackingNet; the `*_got10k_only` ones on
GOT-10k only). They are used here only as they are: for testing (`weights="official"`, [test.md](test.md#weights)).
The runs of this repository are kept apart from that training:

| Origin | How | Run name | Listed under |
|---|---|---|---|
| from ImageNet | stage 1 from the ImageNet backbone, stage 2 on our own stage 1 (default) | `st101_coco_stage1`, `st101_coco_stage2` | "trained here from ImageNet" |
| on the official weights | stage 2 with `init="official"`: the official checkpoint without its classification head | `st101_coco_stage2_on-official` | "trained here on top of the official STARK weights" |
| on another file | stage 2 with `init="<checkpoint path>"` | `..._from-<file>_...` | "trained here on top of another checkpoint file" |

Runs are grouped by **family** (model + datasets + sampling ratios): `st101_coco`, `st101_got10k+coco`, ... A
family's stage 2 starts from its own stage-1 run (`init=None`, i.e. the run with the same parameters and `stage=1`),
so its weights have seen only the family's datasets. The origin is stored in every run's `train_config.json` and is
inherited along the chain; `python -m stark_ft weights` (notebook: `nb.list_weights()`) lists the official weights and
our runs by origin. With `init="official"` the backbone, transformer and box head are STARK's stage-1 weights (stage 2
freezes them), trained on all four datasets: such a run is not "trained on fewer datasets", hence the separate name.

## Datasets

| `datasets` key | STARK dataset | Size | Preparation |
|---|---|---|---|
| `got10k` | `GOT10K_vottrain`: GOT-10k train without the 1000 videos that overlap with VOT (7 086 videos) | 73.9 GB (GOT-10k train folder) | from the official archives (registration, see below) |
| `got10k_full` | `GOT10K_train_full`: all 9 335 GOT-10k train videos (GOT-10k protocol, `*_got10k_only` configs) | (same folder) | (same) |
| `coco` | `COCO17`: COCO 2017 train (118 287 images; every object is a one-frame "video") | 20.1 GB | automatic (images.cocodataset.org) |
| `lasot` | `LASOT`, train split (1 120 videos) | very large | manual |
| `trackingnet` | `TRACKINGNET` (≈ 30 000 videos) | ≈ 1 TB | manual |

Validation (`val_datasets=["got10k"]`) uses `GOT10K_votval`, 1 249 GOT-10k train videos disjoint from
`GOT10K_vottrain` (as in STARK). It needs GOT-10k even when training only on COCO; `val_datasets=[]` disables it.

```
<train_data>                      default data/train (configs/paths.yaml)
├── got10k/train/                 list.txt, GOT-10k_Train_000001/ ...
├── coco/                         annotations/instances_train2017.json, images/train2017/
├── lasot/                        <class>/<class>-<n>/
├── trackingnet/                  TRAIN_0 ... TRAIN_11/
└── _archives/{coco,got10k}/      downloaded archives
```

```bash
python -m stark_ft prepare-train-data --datasets got10k coco [--got10k-url URL_OR_PATH ...] [--archives DIR]
```

- **COCO** is downloaded (`train2017.zip`, `annotations_trainval2017.zip`; interrupted downloads resume) and checked
  for 118 287 images.
- **GOT-10k** needs a free registration at [got-10k.aitestunion.com/downloads](http://got-10k.aitestunion.com/downloads);
  the download links come by e-mail (e.g. a Google Drive link to `full_data.zip`, 70.7 GB). Give the links or the
  paths of downloaded archives with `--got10k-url` (notebook: `got10k_sources`), or put the archives into
  `<archives>/got10k/`. Google Drive share links are converted to direct downloads; shared Drive files often hit
  Google's daily download limit ("Quota exceeded"), in which case make a copy of the file in your own Drive and use
  that (see [colab.md](colab.md#got-10k-from-google-drive)). Only the train videos and `list.txt` are extracted, also
  from archives inside archives; all 9 335 videos must be present. The same archive twice (e.g. your copy and a
  shortcut to the shared file) is extracted only once.
- Archives are kept, so another machine / session only extracts them. Extraction goes into a temporary folder that is
  renamed when complete, free disk space is checked first, and a dataset folder that already has the expected layout
  is only read. An extracted GOT-10k copy can be used directly: set `train_data` so that
  `<train_data>/got10k/train/list.txt` exists.

## Running

```bash
# STARK-ST101 on GOT-10k + COCO: stage 1 from ImageNet, then stage 2 on top of it (STARK-ST50: model_config=baseline)
python -m stark_ft train --set 'datasets=[got10k, coco]'               # st101_got10k+coco_stage1
python -m stark_ft train --set 'datasets=[got10k, coco]' --set stage=2 # st101_got10k+coco_stage2
# only the classification head, on STARK's weights (marked on-official)
python -m stark_ft train --set 'datasets=[coco]' --set stage=2 --set init=official --set 'val_datasets=[]'
python -m stark_ft train --config configs/train_example.yaml
python -m stark_ft weights                      # official weights and our runs, by origin
python -m stark_ft train ... --dry-run          # only check the parameters and the data
python -m stark_ft train-report <run name>      # progress, last losses, history.png
python -m stark_ft train-list
```

Notebook: `notebooks/train.ipynb`. The progress output shows samples/s, the epoch ETA and the training ETA.

## Typical runs

| Goal | `stage` | `datasets` | `init` | `val_datasets` |
|---|---|---|---|---|
| COCO, stage 1 | `1` | `["coco"]` | `None` | `[]` |
| COCO, stage 2 | `2` | `["coco"]` | `None` (= `st101_coco_stage1`) | `[]` |
| GOT-10k, stage 1 | `1` | `["got10k"]` | `None` | `["got10k"]` |
| GOT-10k, stage 2 | `2` | `["got10k"]` | `None` (= `st101_got10k_stage1`) | `["got10k"]` |
| GOT-10k + COCO, stage 1 | `1` | `["got10k", "coco"]` | `None` | `["got10k"]` |
| GOT-10k + COCO, stage 2 | `2` | `["got10k", "coco"]` | `None` (= `st101_got10k+coco_stage1`) | `["got10k"]` |
| Only the classification head, on STARK's weights | `2` | e.g. `["coco"]` | `"official"` | `[]` |
| GOT-10k only, without stage 1 | `2` | `["got10k_full"]` | `"official"` with `model_config="baseline_R101_got10k_only"` | `["got10k"]` |

Stage 2 is the stage-1 cell with `stage=2`; another stage-1 run can be given by name (`init`, the folder name in
`<train_outputs>`). The rows with `"official"` build on STARK's weights and are marked `on-official`.
`val_datasets=["got10k"]` validates on GOT-10k videos that are not used for training; use `[]` without GOT-10k.
STARK's GOT-10k-only weights used all 9 335 GOT-10k videos, including the 1 000 that overlap with VOT and are left
out by `got10k`.

## Parameters

`TrainConfig` in `stark_ft/train/config.py`. Notebook: `TRAIN = dict(...)`; CLI: `--config file.yaml` and/or
`--set key=value`. `None` means the original STARK value.

| Parameter | Default | Description |
|---|---|---|
| `name` | `None` | Run folder `<train_outputs>/<name>/`. `None`: `<family>_stage<N>[_e<epochs>][_on-official / _from-<init>]`, e.g. `st101_got10k+coco_stage1`. |
| `model_config` | `"baseline_R101"` | `baseline_R101` (STARK-ST101) or `baseline` (STARK-ST50). The `*_got10k_only` configs only differ in the data (use them with `datasets=["got10k_full"]`); in stage 2 they select the official GOT-10k-only checkpoint as `init`. |
| `stage` | `1` | `1` or `2`. |
| `init` | `None` | Stage 1: `None` (ImageNet backbone). Stage 2: `None` = our stage-1 run with the same parameters, another stage-1 run name, `"official"` (STARK's weights; marked `on-official`) or a checkpoint path. |
| `datasets` | `["got10k"]` | Any non-empty combination of `got10k` (or `got10k_full`), `coco`, `lasot`, `trackingnet`. The order does not matter. |
| `dataset_ratios` | `None` | Sampling weight per dataset (same order as `datasets`); each sample first picks a dataset with these weights, then a video. `None`: equal. |
| `val_datasets` | `["got10k"]` | `["got10k"]` or `[]`. |
| `epochs` | `None` | 500 (stage 1) / 50 (stage 2). |
| `lr_drop_epoch` | `None` | Learning rate ÷ 10 after this epoch: 400 / 40. |
| `samples_per_epoch` | `None` | 60 000. |
| `val_samples_per_epoch` | `None` | 10 000. |
| `val_interval` | `None` | Validate every 20 / 10 epochs. |
| `effective_batch` | `128` | Samples per optimizer step. 128 = original (8 GPUs × 16); changing it changes the training. |
| `micro_batch` | `16` | Samples per GPU pass; `effective_batch / micro_batch` passes are accumulated. Only memory and speed depend on it. |
| `num_workers` | `8` | Data loading processes. |
| `keep_every` | `None` | Keep the weights every N epochs: 50 / 10. |

## Run folder and resuming

| File in `<train_outputs>/<run name>/` | Content |
|---|---|
| `train_config.json` | Parameters, origin, dataset folders, initial weights, GPU, TF32 setting, code hash, date |
| `history.csv`, `history.png` | One row per epoch: duration, learning rate, training / validation losses; plot (dashed: LR drops) |
| `logs/train.log` | Progress output |
| `checkpoints/latest.pth.tar` | Complete state after the last finished epoch (network, optimizer, scheduler, random generators), written atomically every epoch. ST101: 0.56 GB (stage 1), 0.19 GB (stage 2) |
| `checkpoints/STARKST_epXXXX.pth.tar` | Weights every `keep_every` epochs and at the end (0.19 GB each) |
| `final.pth.tar` | Final weights |

- Running the same parameters again **resumes** from `latest.pth.tar`; the result is bit-identical to an
  uninterrupted run (verified by killing a run during training). The same `name` with different parameters, or with a
  different version of the training code (`lib/train`, `lib/models`, `lib/config`, `lib/utils`,
  `model_configs/stark_st1|2`, `stark_ft/train/run.py`), stops with an error; use another `name` or `--overwrite`.
- A finished stage-2 run is tested with `weights="<run name>"` (same `model_config`), see [test.md](test.md#weights).
  Stage-1 runs have an untrained classification head and cannot be tested; they are the starting point of stage 2.
- Nothing is written to the `checkpoints` folder.

## One GPU instead of eight

The original was trained on 8 GPUs with 16 samples each; DistributedDataParallel averages their gradients, so every
optimizer step uses the mean gradient of 128 samples. Here each micro-batch loss is divided by
`effective_batch / micro_batch`, the gradients are accumulated, and then the gradient is clipped and the step is
taken. STARK freezes all BatchNorm layers of the backbone, so a sample's forward pass does not depend on the rest of
its batch: this is the same update. Numerically, the accumulated gradient of 8 × 16 samples and the gradient of one
batch of 128 differ by 3·10⁻¹⁵ (relative) in float64 and by ≈ 10⁻⁴ in float32 (summation order). The number of steps
per epoch is also the same (60 000 / 128 = 468). Only the random sample stream differs from the original (8 processes
with their own data workers); runs with the same parameters, including `micro_batch` and `num_workers`, on the same
GPU type are bit-identical.

**Time:** stage 1 processes 30 M samples, stage 2 3 M. Measured on an RTX 3060 laptop GPU: stage 2 ≈ 35 samples/s
(≈ 24 h), stage 1 ≈ 20 samples/s (≈ 17 days); datacenter GPUs are several times faster. Ampere and newer GPUs use
TF32 for convolutions by default, a T4 does not; compare runs on the same GPU type
(see [test.md](test.md#results-across-gpus)).

## Differences from the original STARK training code

| Original STARK | Here |
|---|---|
| 8 GPUs (DistributedDataParallel), 16 samples per GPU | 1 GPU, gradient accumulation to the same effective batch |
| Dataset paths in a machine-specific `local.py` | `train_data` (`configs/paths.yaml`) and the `datasets` parameter |
| A resumed stage-2 run loaded the stage-1 weights again after resuming, overwriting the trained classification head | Initial weights are loaded only when a run starts from scratch |
| Checkpoints only in the last 10 epochs and every 100 epochs, without scheduler and random-generator states | Complete state after every epoch (atomic), weights every `keep_every` epochs; bit-identical resuming |
| Pickled settings objects in the checkpoints | Plain dictionaries |
| Did not run with newer PyTorch / pandas (`torch._six`, `storage()._new_shared`, `read_csv(squeeze=True)`) | Fixed without changing the behaviour |

Kept as in the original: in stage 1, if the network predicts an invalid box (x2 < x1 or y2 < y1) for any sample of a
batch, the GIoU computation fails and the GIoU loss of that batch counts as 0 (only L1 is used). This happens in the
first steps of training from scratch and disappears quickly; here it applies per micro-batch (16 samples by default,
as on each of the 8 GPUs of the original).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Lower `micro_batch` (e.g. 8) and keep `effective_batch=128`; the update stays the same. |
| The machine crashes or runs out of RAM | Lower `num_workers` (COCO's annotation file is large and every data worker uses memory). |
| `Not enough disk space for ...` | The extracted datasets do not fit (GOT-10k ≈ 74 GB, COCO ≈ 20 GB). Use a larger disk or fewer datasets. |
| `Stage 2 starts from the stage-1 run of the same family ...` | Train stage 1 first (the same parameters with `stage=1`), or give another finished stage-1 run / `"official"` as `init`. |
| `No GOT-10k archive in ...` | Put the GOT-10k archives into the folder named in the message (it is created), or give links / archive paths. |
| `... 'Google Drive - Quota exceeded'` | The shared file reached its daily download limit; use your own copy of it ([colab.md](colab.md#got-10k-from-google-drive)). |
| `... already exists with different parameters` / `different version of the training code` | Use another `name`, or `--overwrite` (notebook: `overwrite=True`) for a fresh run. |
