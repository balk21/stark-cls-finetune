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
├── checkpoints
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
python -m stark_ft download-checkpoints --model-config baseline_R101   # official STARK-ST101 (baseline: ST50)
python -m stark_ft prepare-train-data --datasets got10k coco           # COCO is downloaded; GOT-10k: docs/train.md
python -m stark_ft download-dataset                                    # optional: all 50 VOT-LT2020 sequences (17.6 GB)
```

## Train

```bash
# STARK-ST101, stage 1 (backbone + transformer + box head)
python -m stark_ft train --set stage=1 --set 'datasets=[got10k, coco]' --set name=st101_s1
# stage 2 (classification head) on top of it
python -m stark_ft train --set stage=2 --set 'datasets=[got10k, coco]' --set init=st101_s1 --set name=st101_s2
# progress
python -m stark_ft train-report st101_s1
```

Any subset of `got10k, coco, lasot, trackingnet` can be used; `--set model_config=baseline` trains STARK-ST50.
Stage 2 always needs `init`: a stage-1 run, or `init=official` (STARK's weights, trained on all four datasets).
Notebook: [`notebooks/train.ipynb`](notebooks/train.ipynb). Details: [docs/train.md](docs/train.md).

## Test

```bash
# STARK-ST101 on VOT-LT2020: without fine-tuning / with online fine-tuning of the classification head
python -m stark_ft test --set ft_mode=none
python -m stark_ft test --set ft_mode=online --set ft_samples=pos
# a model trained with this repository
python -m stark_ft test --set checkpoint=train:st101_s2
# compare experiments
python -m stark_ft compare st101_base_int100 st101_online_pos_lr0.0001_i15_o1_int100_s0
```

A test downloads the VOT-LT2020 sequences it uses that are not on disk yet (e.g. `bull` 58 MB; all 50: 17.6 GB).
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
