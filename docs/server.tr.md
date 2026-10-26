# Kiralık GPU makinesinde eğitim

[English](server.md) | **Türkçe**

Betikler [`scripts/`](../scripts) klasöründe; repo klasöründe çalıştırın. Her betik yarıda kalırsa tekrar
çalıştırılabilir: kaldığı yerden devam eder.

## 1. Hazırlık

```bash
git clone https://github.com/balk21/stark-cls-finetune.git && cd stark-cls-finetune
scripts/setup.sh /workspace/data    # araçlar, vot1 ortamı (~10 dk), yollar, kontrol
```

`/workspace/data`: büyük diskteki bir klasör (`df -h`). Verilmezse veriler repo klasörüne gider.

Yalnızca eğitimde kullanılacak veri setleri; aynı anda çalışabilirler (ayrı terminal ya da tmux pencereleri):

| Veri seti | Komut | İndirme → diskte |
|---|---|---|
| COCO | `scripts/get_coco.sh` | 19 GB → 20 GB |
| GOT-10k | `scripts/get_got10k.sh <Google Drive linki \| full_data.zip yolu>` | 70.7 GB → 74 GB |
| LaSOT | `scripts/get_lasot.sh` | 248 GB → 248 GB |

- Arşivler açıldıktan sonra silinir; `KEEP_ARCHIVES=1 scripts/get_coco.sh` saklar.
- GOT-10k linki: `full_data.zip`'in kendi Google Drive'ınızdaki kopyası, "bağlantıya sahip herkes" ile paylaşılmış
  ([colab.tr.md](colab.tr.md#google-driveda-got-10k)).
- Klasörler: `<data>/train/{coco, got10k/train, lasot}`, eğitim koşuları `<data>/training/` altında.

## 2. Eğitim

```bash
scripts/train.sh --set 'datasets=[got10k, coco, lasot]'                # stage 1, tmux oturumu "train" içinde
scripts/train.sh --set 'datasets=[got10k, coco, lasot]' --set stage=2  # stage 2, stage 1 bittikten sonra
tmux attach -t train                                                   # izlemek için (çalışır bırakmak: Ctrl-b d)
scripts/stark.sh train-report st101_lasot+got10k+coco_stage1           # ilerleme
```

- Aynı komut tekrar çalıştırılırsa koşu son biten epoch'tan devam eder.
- `train.sh`, `python -m stark_ft train` ile aynı argümanları alır ([train.tr.md](train.tr.md)), ör.
  `--set num_workers=12`, `--set model_config=baseline`.

Yedekleme (kiralık makinenin diski makineyle birlikte gider):

```bash
rclone config                                                              # bir kez, ör. Google Drive'ı "gdrive" adıyla
scripts/backup.sh st101_lasot+got10k+coco_stage1 gdrive:LOKAP/training     # 30 dk'da bir, tmux oturumu backup-<koşu>
scripts/restore.sh st101_lasot+got10k+coco_stage1 gdrive:LOKAP/training    # yeni makinede; sonra aynı train.sh
```

Başka makinede devam etmek için parametreler ve eğitim kodu aynı olmalı ([train.tr.md](train.tr.md)).
