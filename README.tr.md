# stark-cls-finetune

[English](README.md) | **Türkçe**

[STARK-ST](https://github.com/researchmm/Stark) tracker'ının (ICCV 2021) güven skoru başlığının test sırasında
fine-tune edilmesi; değerlendirme **VOT-LT2020** üzerinde, tespit metrikleriyle (**mAP, AP50, AP75**).

**Neden.** STARK-ST her karede bir kutu ve bir güven skoru verir. Skor, hedefin görünüp görünmediğini söyler ve
template'in ne zaman güncelleneceğine karar verir; bu yüzden uzun süreli takipte yanlış skor, yanlış kutu kadar
zararlıdır. Skor, tüm nesneler için bir kez eğitilmiş küçük bir başlıktan (`cls_head`) gelir. Bu repo, takip
sırasında bu başlığı **her videonun hedefine özel** birkaç adım eğitir (geri kalan her şey dondurulur) ve etkisini
ölçer.

**Neler yapılabilir**

| Notebook | Amaç |
|---|---|
| `00_setup_colab` / `00_setup` | Ortamı bir kez hazırlamak (Colab / Vast.ai veya kendi bilgisayarınız) |
| `01_run_experiment` | VOT-LT2020 dizilerini fine-tune ile veya fine-tune olmadan takip edip metrikleri hesaplamak |
| `02_compare` | Deneyleri yan yana koymak |
| `03_train` | STARK-ST'nin kendisini seçtiğiniz veri setleriyle (GOT-10k, COCO veya ikisi) yeniden eğitmek |

Her şey bu notebook'lardan çalışır; her notebook'un ilk hücresi ihtiyaç duyduğu şeyleri hazırlar.

## 1. Kurulum

**GPU:** en az 8 GB'lık bir NVIDIA GPU, örn. Colab **A100** / L4 veya RTX 3000 / 4000 serisi. RTX 5000 serisi
çalışmaz (PyTorch sürümü desteklemiyor). T4'ten de kaçının: T4'te TF32 yok, bu yüzden sonuçları biraz farklı çıkar.
**Yalnızca aynı GPU tipinde alınmış sonuçları karşılaştırın** ([neden](docs/details.tr.md#gpular-arası-sonuçlar)).

**Google Colab** (önerilen)
1. [`00_setup_colab.ipynb`](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/00_setup_colab.ipynb)
   dosyasını açın ve *Runtime → Change runtime type → A100* seçin.
2. Tüm hücreleri çalıştırın (ilk seferde 10–15 dk). Bu adım `vot1` conda ortamını kurar (her makinede aynı
   paketler, böylece sonuçlar tekrarlanabilir), STARK-ST101 checkpoint'ini indirir ve test olarak 50 kare takip
   eder (ortalama IoU > 0.5 ise kurulum çalışıyordur).

Colab her oturumdan sonra makineyi siler; bu yüzden her şey Google Drive'da `MyDrive/LOKAP/` altında tutulur (ortam,
checkpoint'ler, sonuçlar, eğitim koşuları). Sonraki oturumlar bunları ~3 dk'da geri yükler. Başka bir klasör
kullanmak için her notebook'un ilk hücresindeki `DRIVE_ROOT` değerini değiştirin.

**Vast.ai veya kendi bilgisayarınız** (conda ve Jupyter kurulu Linux)
```bash
git clone https://github.com/balk21/stark-cls-finetune.git
```
`notebooks/00_setup.ipynb` dosyasını açıp tüm hücreleri çalıştırın (ortamın kurulumu bir kez, 10–20 dk).

## 2. Deney çalıştırma: `01_run_experiment.ipynb`

Bir deney = STARK-ST'nin seçilen dizileri tek bir parametre setiyle (resmi vot-toolkit üzerinden) takip etmesi,
ardından metriklerin ve grafiklerin hesaplanması.

**Yalnızca `PARAMS` hücresini (1. adım) değiştirin:**

| İstediğiniz | Ayar |
|---|---|
| Düz STARK-ST (baseline) | `ft_mode="none"` |
| Yalnızca ilk karede fine-tune | `ft_mode="init"` |
| İlk karede ve her template update'te fine-tune | `ft_mode="online"` |
| Eğitim örnekleri: yalnızca pozitif / pozitif + negatif | `ft_samples="pos"` / `"posneg"` |
| Fine-tune'un şiddeti | `ft_lr`, `ft_epochs_init` (1. karedeki adım sayısı), `ft_epochs_online` (her güncellemedeki adım sayısı) |
| Diziler | `sequences=["bull", "ballet"]` veya `"all"` (50 dizinin tamamı) |
| ST101 yerine STARK-ST50 | `model_config="baseline"` |
| Kendi eğittiğiniz model ([§4](#4-stark-st-eğitimi-03_trainipynb)) | `checkpoint="train:<aşama-2 koşu adı>"` |

Gerisini varsayılan değerlerde bırakın ([tüm parametreler](docs/details.tr.md#4-parametreler)). Sonra hücreleri
yukarıdan aşağı çalıştırın: kontrol hücresi çıktı klasörünü ve neyin indirileceğini gösterir, 2. adım çalıştırır,
3. adım sonuçları gösterir.

- Bir dizi ilk kullanıldığında indirilir (`bull` 58 MB; 50 dizinin tamamı 17,6 GB) ve sonra saklanır (Colab'de
  Drive'da).
- Süre: RTX 3060'ta kare başına ~30 ms; `bull` ~1 dk, 50 dizinin tamamı ~2 saat.
- Yarıda mı kaldı? Hücreyi tekrar çalıştırın; biten diziler atlanır. Çıktı klasörünün adı parametrelerden
  oluşturulur; böylece farklı bir parametre seti başka bir deneyin üzerine yazmaz.

**Sonuçlar** `outputs/<deney>/` altındadır (Colab'de `MyDrive/LOKAP/outputs/`):

| Dosya | İçerik |
|---|---|
| `metrics/summary.txt` | önce mAP / AP50 / AP75, sonra precision / recall / F |
| `metrics/metrics.xlsx` | tüm metrikler, dizi bazında da |
| `plots/<dizi>/iou_conf.png` | kare bazında IoU ve skor, template / fine-tune güncellemeleriyle |
| `predictions/<dizi>/` | her karenin kutusu ve skoru |

## 3. Deneyleri karşılaştırma: `02_compare.ipynb`

Deney adlarını `NAMES = [...]` içine yazın (ilki referanstır) ve hücreleri çalıştırın. Metrikleri yan yana,
parametre farklarını ve dizi bazındaki farkları görürsünüz; hepsi `comparison.xlsx` olarak da kaydedilir.

## 4. STARK-ST eğitimi: `03_train.ipynb`

Resmi STARK-ST ağırlıkları dört veri setiyle birlikte eğitildi (LaSOT, GOT-10k, COCO, TrackingNet). Bu notebook,
STARK-ST'yi orijinal yöntemle **sizin** seçtiğiniz veri setleriyle (örn. yalnızca COCO) eğitir; böylece eğitim
verisinin etkisi incelenebilir. STARK-ST iki aşamada eğitilir:

| Aşama | Eğitilen | Uzunluk | Başlangıç |
|---|---|---|---|
| 1 | backbone, transformer, kutu başlığı | 500 epoch | ImageNet backbone |
| 2 | yalnızca sınıflandırma başlığı | 50 epoch | tamamlanmış bir aşama-1 koşusu |

Orijinal eğitim 8 GPU kullanıyordu (adım başına 128 örnek). Burada tek GPU, her adımdan önce 8 × 16 örneği toplar
(gradient accumulation); bu aynı güncellemeyi verir, yani eğitim değişmez.

### 4.1 Veri (bir kez)

- **COCO:** bir şey yapmanız gerekmez; otomatik indirilir (19,6 GB, Drive'da saklanır).
- **GOT-10k:** [got-10k.aitestunion.com](http://got-10k.aitestunion.com/downloads) üzerinde ücretsiz kayıt gerekir;
  indirme linkleri e-postayla gelir. Colab'de:
  1. `full_data.zip` linkini açın → *Drive'a kısayol ekle*.
  2. Kısayola sağ tık → *Kopyasını oluştur* (Drive'da 70,7 GB yer kaplar). Neden: paylaşılan dosya sık sık
     Google'ın günlük indirme sınırına takılır; sizin kopyanız takılmaz.
  3. Kopyayı `MyDrive/LOKAP/train_archives/got10k/` içine taşıyın (bu klasörü notebook'un 2. adımı oluşturur) ve
     kısayolu silin.

  Vast.ai'da / kendi bilgisayarınızda arşivleri `data/train/_archives/got10k/` içine koyun.

Her Colab oturumunun başında arşivler oturumun yerel diskine açılır (GOT-10k ≈ 35 dk); çünkü eğitim sırasında
görüntüleri Drive'dan okumak çok daha yavaş olurdu.

### 4.2 Eğitim

**`TRAIN` hücresinde (1. adım) yalnızca şu satırları değiştirin:**

| Amaç | `stage` | `datasets` | `init` | `val_datasets` |
|---|---|---|---|---|
| COCO, aşama 1 | `1` | `["coco"]` | `None` | `[]` |
| COCO, aşama 2 | `2` | `["coco"]` | `"st101_stage1_coco_s42"` | `[]` |
| GOT-10k, aşama 1 | `1` | `["got10k"]` | `None` | `["got10k"]` |
| GOT-10k, aşama 2 | `2` | `["got10k"]` | `"st101_stage1_got10k_s42"` | `["got10k"]` |
| GOT-10k + COCO, aşama 1 | `1` | `["got10k", "coco"]` | `None` | `["got10k"]` |
| GOT-10k + COCO, aşama 2 | `2` | `["got10k", "coco"]` | `"st101_stage1_got10k+coco_s42"` | `["got10k"]` |

- `init` (aşama 2), tamamlanmış aşama-1 koşusunun adıdır; eğitim sırasında yazdırılır ve
  `MyDrive/LOKAP/training/` altındaki klasörün adıdır. `init="official"` ise dört veri setini de görmüş resmi STARK
  ağırlıklarından başlatır.
- `val_datasets=["got10k"]`, eğitimde kullanılmayan GOT-10k videolarında kaybı ölçer; GOT-10k hazırlanmadıysa `[]`
  verin.
- STARK-ST50 için: `model_config="baseline"` (koşu adları bu durumda `st50_` ile başlar).

Ardından 2. adımı (oturumu ve veriyi hazırlar, planı yazdırır: koşu adı, epoch'lar, başlangıç ağırlıkları) ve
3. adımı (eğitir) çalıştırın. Oturum kapanırsa 2. ve 3. adımı tekrar çalıştırın: eğitim son tamamlanan epoch'tan,
kesinti hiç olmamış gibi aynı sonuçla devam eder. Her koşunun kendi klasörü vardır: `MyDrive/LOKAP/training/<koşu adı>/`.

**Süre:** aşama 1 30 milyon, aşama 2 3 milyon örnek okur. RTX 3060'ta aşama 2 ≈ 1 gün, aşama 1 ≈ 17 gün sürer;
A100 birkaç kat daha hızlıdır. İlerleme satırı ilk dakikalardan sonra kalan süreyi gösterir; uzun bir aşama-1
koşusuna girmeden önce buna bakın.

**Yalnızca GOT-10k, aşama 1 olmadan:** STARK, yalnızca GOT-10k ile eğitilmiş ağırlıklar da yayımladı:
`model_config="baseline_R101_got10k_only", datasets=["got10k_full"], stage=2, init="official"`. Bu ağırlıklar
GOT-10k'nın 9 335 videosunun tamamıyla eğitildi; `got10k` seçeneğinin dışarıda bıraktığı, VOT ile örtüşen 1 000
video da buna dahil.

### 4.3 Eğitilen modeli değerlendirme

`01_run_experiment.ipynb` içinde: `checkpoint="train:<aşama-2 koşu adı>"` ve aynı `model_config`.

## 5. Metrikler

- **mAP, AP50, AP75** (birincil): standart COCO tespit metrikleri; her kare bir görüntüdür. Hedefin olmadığı kareler
  de sayılır (oradaki bir kutu yanlış pozitiftir). Skor eşiği uygulanmaz.
- **En iyi eşikteki F** (`precision_opt`, `recall_opt`, `F_opt`): VOT-LT protokolü, yani tüm dizilerde F'yi en büyük
  yapan skor eşiği.
- `legacy_*`: eski `coco_eval.py` hesaplaması; yalnızca eski sonuçlarla karşılaştırmak için.

Tam tanımlar: [docs/details.tr.md](docs/details.tr.md#6-metrikler).

## 6. Bilmekte fayda var

- vot-toolkit "A newer version of the VOT toolkit is available" yazar: önemsemeyin, repo 0.5.3 sürümüyle çalışır.
- Eğitimde GPU belleği yetmezse → `micro_batch=8` (aynı eğitim, yalnızca daha yavaş). Oturum çökerse (RAM) →
  `num_workers=4`.
- Daha fazlası: [sık karşılaşılan sorunlar](docs/details.tr.md#9-sık-karşılaşılan-sorunlar),
  [komut satırı](docs/details.tr.md#8-komut-satırı-cli),
  [bilinen sınırlamalar](docs/details.tr.md#10-bilinen-sınırlamalar-ve-açık-konular),
  [eski koddan farklar](docs/details.tr.md#11-eski-koddan-farklar-ve-düzeltilen-hatalar).

## 7. Atıf

Bu repo STARK üzerine kuruludur:

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```

ve önceki çalışmamızın devamıdır:

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
