# STARK-ST with test-time fine-tuning of the classification head

**English** | [Türkçe](README.tr.md)

This repository builds on [STARK](https://github.com/researchmm/Stark) (ICCV 2021) and adds:

- **Test-time fine-tuning** of the STARK-ST classification head on every video (`init` / `online`, positive or
  positive + negative samples), evaluated on **VOT-LT2020** with COCO-style detection metrics (mAP, AP50, AP75) and
  the VOT-LT F-max threshold.
- **Training STARK-ST on one GPU** (stage 1 and stage 2) with the original effective batch size, on any combination
  of GOT-10k, COCO, LaSOT and TrackingNet.

## Install

```bash
git clone https://github.com/balk21/stark-cls-finetune.git
cd stark-cls-finetune
conda env create -n vot1 -f environment/vot1_environment.yml
conda activate vot1
```

Python 3.8, PyTorch 2.4.1, CUDA 12.1, vot-toolkit 0.5.3; tested on NVIDIA RTX 3000 / 4000 series GPUs.
Everything can also be run from the notebooks in `notebooks/` (they create the environment if it is missing), on a
local machine, a server or [Google Colab](docs/colab.md).

## Data preparation

Put the data in `./data` and the official checkpoints in `./checkpoints` (or set other locations in
`configs/paths.local.yaml`, see `configs/paths.local.example.yaml`):

```
${ROOT}
├── checkpoints                      # official STARK weights (downloaded when used)
│   └── stark_st2/baseline_R101/STARKST_ep0050.pth.tar
└── data
    ├── votlt2020/sequences          # test (downloaded on demand)
    └── train                        # training
        ├── got10k/train
        ├── coco/{annotations, images}
        ├── lasot
        └── trackingnet
```

```bash
python -m stark_ft prepare-train-data --datasets got10k coco           # COCO is downloaded; GOT-10k: docs/train.md
```

The official STARK weights and the VOT-LT2020 sequences are downloaded automatically when a test uses them
(in advance: `download-checkpoints`, `download-dataset`).

## Train

```bash
# STARK-ST101 on GOT-10k + COCO: stage 1 (backbone, transformer, box head) from ImageNet -> st101_got10k+coco_stage1_s42
python -m stark_ft train --set 'datasets=[got10k, coco]'
# stage 2 (classification head) on top of our stage 1                               -> st101_got10k+coco_stage2_s42
python -m stark_ft train --set 'datasets=[got10k, coco]' --set stage=2
# progress / all weights (official and trained here)
python -m stark_ft train-report st101_got10k+coco_stage1_s42
python -m stark_ft weights
```

Our runs are kept apart from STARK's published training: they are trained only on the given datasets (any subset of
`got10k, coco, lasot, trackingnet`) and grouped by name (model + datasets + stage). Stage 2 on STARK's own weights
(`--set init=official`) is possible, but such runs are named `..._on-official_...` and listed separately.
`--set model_config=baseline` trains STARK-ST50. Notebook: [`notebooks/train.ipynb`](notebooks/train.ipynb).
Details: [docs/train.md](docs/train.md).

## Test

```bash
# STARK-ST101 (official weights) on bull: without fine-tuning / with online fine-tuning of the classification head
python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=none
python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=online --set ft_samples=pos
# the same with weights trained here
python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=none --set weights=st101_got10k+coco_stage2_s42
# compare experiments
python -m stark_ft compare st101_base_int100_bull st101_got10k+coco_stage2_s42_base_int100_bull
```

`weights`: `official` (STARK's, default), a stage-2 run trained here (`python -m stark_ft weights` lists them) or a
checkpoint file. The sequences are downloaded automatically (e.g. `bull` 58 MB; `sequences=all`: 50, 17.6 GB).
Notebooks: [`notebooks/test.ipynb`](notebooks/test.ipynb), [`notebooks/compare.ipynb`](notebooks/compare.ipynb).
Method, parameters, outputs and metrics: [docs/test.md](docs/test.md).

## Citation

STARK:

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```

Previous work:

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

## Acknowledgments

The tracker and its training code are from the official [STARK](https://github.com/researchmm/Stark) repository,
which builds on [PyTracking](https://github.com/visionml/pytracking) and [DETR](https://github.com/facebookresearch/detr).
Evaluation uses [vot-toolkit](https://github.com/votchallenge/toolkit) and [pycocotools](https://github.com/cocodataset/cocoapi).
