# STARK-ST için sınıflandırma başlığının test sırasında fine-tune edilmesi

[English](README.md) | **Türkçe**

Bu depo [STARK](https://github.com/researchmm/Stark) (ICCV 2021) üzerine kuruludur ve şunları ekler:

- STARK-ST sınıflandırma başlığının her videoda **test sırasında fine-tune edilmesi** (`init` / `online`; pozitif ya da
  pozitif + negatif örneklerle). **VOT-LT2020** üzerinde COCO tarzı detection metrikleri (mAP, AP50, AP75) ve VOT-LT
  F-maksimum eşiğiyle değerlendirilir.
- **STARK-ST'nin tek GPU'da eğitimi** (aşama 1 ve aşama 2): orijinal effective batch boyutuyla, GOT-10k, COCO, LaSOT
  ve TrackingNet'in herhangi bir kombinasyonuyla.

## Kurulum

```bash
git clone https://github.com/balk21/stark-cls-finetune.git
cd stark-cls-finetune
conda env create -n vot1 -f environment/vot1_environment.yml
conda activate vot1
```

Python 3.8, PyTorch 2.4.1, CUDA 12.1, vot-toolkit 0.5.3; NVIDIA RTX 3000 / 4000 serisi GPU'larda test edildi.
Her şey `notebooks/` altındaki notebook'larla da çalıştırılabilir (ortam yoksa kendileri kurar): yerel makinede,
sunucuda veya [Google Colab](docs/colab.tr.md)'de.

## Verinin hazırlanması

Veriyi `./data`, resmi checkpoint'leri `./checkpoints` altına koyun (veya başka konumları
`configs/paths.local.yaml` içinde belirtin, bkz. `configs/paths.local.example.yaml`):

```
${ROOT}
├── checkpoints
│   └── stark_st2/baseline_R101/STARKST_ep0050.pth.tar
└── data
    ├── votlt2020/sequences          # test (gerektiğinde indirilir)
    └── train                        # eğitim
        ├── got10k/train
        ├── coco/{annotations, images}
        ├── lasot
        └── trackingnet
```

```bash
python -m stark_ft download-checkpoints --model-config baseline_R101   # resmi STARK-ST101 (baseline: ST50)
python -m stark_ft prepare-train-data --datasets got10k coco           # COCO indirilir; GOT-10k: docs/train.tr.md
python -m stark_ft download-dataset                                    # isteğe bağlı: 50 VOT-LT2020 dizisinin tamamı (17.6 GB)
```

## Eğitim

```bash
# STARK-ST101, aşama 1 (backbone + transformer + kutu başlığı)
python -m stark_ft train --set stage=1 --set 'datasets=[got10k, coco]' --set name=st101_s1
# aşama 2 (sınıflandırma başlığı), aşama 1'in üzerine
python -m stark_ft train --set stage=2 --set 'datasets=[got10k, coco]' --set init=st101_s1 --set name=st101_s2
# ilerleme
python -m stark_ft train-report st101_s1
```

`got10k, coco, lasot, trackingnet` veri setlerinin herhangi bir alt kümesi kullanılabilir; `--set model_config=baseline`
STARK-ST50'yi eğitir. Aşama 2 her zaman `init` ister: bir aşama-1 koşusu ya da `init=official` (STARK'ın dört veri
setinin tamamıyla eğitilmiş ağırlıkları). Notebook: [`notebooks/train.ipynb`](notebooks/train.ipynb). Ayrıntılar:
[docs/train.tr.md](docs/train.tr.md).

## Test

```bash
# VOT-LT2020'de STARK-ST101: fine-tune olmadan / sınıflandırma başlığının online fine-tune'u ile
python -m stark_ft test --set ft_mode=none
python -m stark_ft test --set ft_mode=online --set ft_samples=pos
# bu depoyla eğitilmiş bir model
python -m stark_ft test --set checkpoint=train:st101_s2
# deneyleri karşılaştırma
python -m stark_ft compare st101_base_int100 st101_online_pos_lr0.0001_i15_o1_int100_s0
```

Test, kullandığı VOT-LT2020 dizilerinden diskte olmayanları indirir (örn. `bull` 58 MB; 50 dizinin tamamı: 17.6 GB).
Notebook'lar: [`notebooks/test.ipynb`](notebooks/test.ipynb), [`notebooks/compare.ipynb`](notebooks/compare.ipynb).
Yöntem, parametreler, çıktılar ve metrikler: [docs/test.tr.md](docs/test.tr.md).

## Atıf

STARK:

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```

Önceki çalışma:

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

## Teşekkür

Tracker ve eğitim kodu resmi [STARK](https://github.com/researchmm/Stark) deposundandır; o da
[PyTracking](https://github.com/visionml/pytracking) ve [DETR](https://github.com/facebookresearch/detr) üzerine
kuruludur. Değerlendirmede [vot-toolkit](https://github.com/votchallenge/toolkit) ve
[pycocotools](https://github.com/cocodataset/cocoapi) kullanılır.
