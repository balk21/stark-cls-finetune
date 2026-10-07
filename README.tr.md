# STARK-ST için sınıflandırma başlığının test sırasında fine-tune edilmesi

[English](README.md) | **Türkçe**

Bu repo [STARK](https://github.com/researchmm/Stark) (ICCV 2021) üzerine kuruludur ve şunları ekler:

- STARK-ST sınıflandırma başlığının her videoda **test sırasında fine-tune edilmesi** (`init` / `online`; pozitif ya da
  pozitif + negatif örneklerle). **VOT-LT2020** üzerinde COCO tarzı detection metrikleri (mAP, AP50, AP75) ve VOT-LT
  F-maksimum eşiğiyle, **GOT-10k**'da (val / test / train; AO, SR) değerlendirilir.
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
├── checkpoints                      # resmi STARK ağırlıkları (kullanıldığında indirilir)
│   └── stark_st2/baseline_R101/STARKST_ep0050.pth.tar
└── data
    ├── votlt2020/sequences          # test (gerektiğinde indirilir)
    └── train                        # eğitim
        ├── got10k/{train, val, test}   # val / test: GOT-10k testleri
        ├── coco/{annotations, images}
        ├── lasot
        └── trackingnet
```

```bash
python -m stark_ft prepare-train-data --datasets got10k coco           # COCO indirilir; GOT-10k: docs/train.tr.md
```

Resmi STARK ağırlıkları ve VOT-LT2020 dizileri, bir test onları kullandığında otomatik indirilir
(önceden indirmek için: `download-checkpoints`, `download-dataset`).

## Eğitim

```bash
# GOT-10k + COCO ile STARK-ST101: ImageNet'ten aşama 1 (backbone, transformer, kutu başlığı) -> st101_got10k+coco_stage1
python -m stark_ft train --set 'datasets=[got10k, coco]'
# kendi aşama 1'imizin üzerine aşama 2 (sınıflandırma başlığı)                           -> st101_got10k+coco_stage2
python -m stark_ft train --set 'datasets=[got10k, coco]' --set stage=2
# ilerleme / tüm ağırlıklar (resmi ve burada eğitilenler)
python -m stark_ft train-report st101_got10k+coco_stage1
python -m stark_ft weights
```

Bizim koşularımız STARK'ın yayımladığı eğitimden ayrı tutulur: yalnızca verilen veri setleriyle eğitilir
(`got10k, coco, lasot, trackingnet`'in herhangi bir alt kümesi) ve adlarıyla gruplanır (model + veri setleri + aşama).
Aşama 2'yi STARK'ın kendi ağırlıkları üzerine eğitmek (`--set init=official`) mümkündür, ama bu koşular
`..._on-official_...` olarak adlandırılır ve ayrı listelenir. `--set model_config=baseline` STARK-ST50'yi eğitir.
Notebook: [`notebooks/train.ipynb`](notebooks/train.ipynb). Ayrıntılar: [docs/train.tr.md](docs/train.tr.md).

## Test

```bash
# bull'da STARK-ST101 (resmi ağırlıklar): fine-tune olmadan / sınıflandırma başlığının online fine-tune'u ile
python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=none
python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=online --set ft_samples=pos
# aynısı, burada eğitilmiş ağırlıklarla
python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=none --set weights=st101_got10k+coco_stage2
# STARK'ın yalnızca GOT-10k ağırlıkları GOT-10k doğrulama setinde (AO / SR); dataset=got10k_test: GOT-10k sunucusu için zip
python -m stark_ft test --set model_config=baseline_R101_got10k_only --set dataset=got10k_val --set sequences=all \
    --set ft_mode=none --set update_interval=200
# deneyleri karşılaştırma
python -m stark_ft compare st101_base_int100_bull st101_got10k+coco_stage2_base_int100_bull
```

`weights`: `official` (STARK'ın ağırlıkları, varsayılan), burada eğitilmiş bir aşama-2 koşusu (`python -m stark_ft weights`
listeler) ya da bir checkpoint dosyası. `dataset`: `votlt2020` (varsayılan; vot-toolkit), `got10k_val`, `got10k_test`,
`got10k_train`. VOT dizileri otomatik indirilir (örn. `bull` 58 MB; `sequences=all`: 50 dizi, 17.6 GB); GOT-10k
bölümleri GOT-10k arşivlerinden açılır (docs/train.tr.md).
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

Tracker ve eğitim kodu resmi [STARK](https://github.com/researchmm/Stark) reposundandır; o da
[PyTracking](https://github.com/visionml/pytracking) ve [DETR](https://github.com/facebookresearch/detr) üzerine
kuruludur. Değerlendirmede [vot-toolkit](https://github.com/votchallenge/toolkit) ve
[pycocotools](https://github.com/cocodataset/cocoapi) kullanılır.
