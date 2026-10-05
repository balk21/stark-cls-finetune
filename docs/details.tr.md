# stark-cls-finetune — ayrıntılar

[English](details.md) | **Türkçe**

[README](../README.tr.md)'nin dışarıda bıraktığı her şeyin başvuru belgesi: tüm parametreler, çıktı dosyaları, metriklerin tam tanımları, eğitim ayrıntıları, komut satırı, sık karşılaşılan sorunlar ve eski koddan farklar.

## İçindekiler
1. [Kendi bilgisayarınızda yollar](#1-kendi-bilgisayarınızda)
2. [Klasör yapısı](#2-klasör-yapısı)
3. [Yöntem](#3-yöntem)
4. [Parametreler](#4-parametreler)
5. [Çıktılar ve dosya formatları](#5-çıktılar-ve-dosya-formatları)
6. [Metrikler](#6-metrikler)
7. [STARK-ST eğitimi](#7-stark-st-eğitimi-base-training)
8. [Komut satırı (CLI)](#8-komut-satırı-cli)
9. [Sık karşılaşılan sorunlar](#9-sık-karşılaşılan-sorunlar)
10. [Bilinen sınırlamalar, GPU'lar arası sonuçlar](#10-bilinen-sınırlamalar-ve-açık-konular)
11. [Eski koddan farklar](#11-eski-koddan-farklar-ve-düzeltilen-hatalar)

## 1. Kendi bilgisayarınızda

Aynı notebook'lar kullanılır. Checkpoint veya veri seti zaten başka bir yerdeyse indirmek yerine
`configs/paths.local.example.yaml` dosyasını `configs/paths.local.yaml` adıyla kopyalayıp yolları yazın:

```yaml
checkpoints: /home/kullanici/stark/checkpoints/train   # <checkpoints>/stark_st2/baseline_R101/STARKST_ep0050.pth.tar
dataset: /home/kullanici/vot/votlt2020/sequences        # <dataset>/<dizi>/{color/, groundtruth.txt, sequence}
dataset_cache: /veri/vot_dizi_arsivleri                 # isteğe bağlı: indirilen her dizinin bir kopyası (<dizi>.tar)
outputs: /home/kullanici/stark_outputs
train_data: /veri/tracking_train                        # yalnızca eğitim: <train_data>/{got10k/train, coco, ...} (§7.3)
train_outputs: /home/kullanici/stark_training           # yalnızca eğitim: her eğitim koşusu için bir klasör (§7.5)
```

Bu dosya git'e girmez. Aynı ayarlar `STARK_CLEAN_CHECKPOINTS`, `STARK_CLEAN_DATASET`, `STARK_CLEAN_DATASET_CACHE`,
`STARK_CLEAN_OUTPUTS`, `STARK_CLEAN_TRAIN_DATA` ve `STARK_CLEAN_TRAIN_OUTPUTS` ortam değişkenleriyle de verilebilir
(öncelik: ortam değişkeni > `paths.local.yaml` > `paths.yaml`).

**VOT dizileri gerektiğinde indirilir.** Önceden bir şey indirmek gerekmez. Bir deney başladığında, kullandığı ve
`dataset` klasöründe henüz olmayan diziler resmi VOT sunucusundan tek tek indirilir (VOT-LT2020 stack'inin kullandığı
VOT-LT2019 dizileri), resmi SHA-1 sağlama toplamlarıyla kontrol edilir ve vot-toolkit'in yazdığıyla birebir aynı
dosyalar olarak yazılır. `sequences=["bull"]` yalnızca bull'u indirir (58 MB); `"all"` 50 dizinin tamamını indirir
(17,6 GB, 20–60 dk). `show` / `nb.describe` neyin indirileceğini önceden gösterir. Önceden indirmek için:
`python -m stark_ft download-dataset [--sequences bull ballet]` (`--sequences` olmadan: 50 dizinin tamamı).
`dataset_cache` tanımlıysa indirilen her dizinin bir kopyası (`<dizi>.tar`) orada tutulur ve başka bir makinede
sunucu yerine oradan alınır; Colab'de bu, Drive'daki `cache/votlt2019_sequences/` klasörüdür.

**Mevcut diziler yalnızca okunur.** Veri seti klasörüne yalnızca eksik olup indirilen diziler yazılır.
VOT-toolkit'in bir deney için ihtiyaç duyduğu `list.txt`, deneyin kendi `vot_workspace/` klasöründe, dizilerin mutlak
yollarıyla oluşturulur.

## 2. Klasör yapısı

```
stark-cls-finetune/
├── notebooks/
│   ├── 00_setup.ipynb            Conda'lı makinede kurulum (ortam, checkpoint, duman testi)
│   ├── 00_setup_colab.ipynb      Google Colab'de kurulum (ortam / checkpoint'ler / diziler Google Drive'da önbelleklenir)
│   ├── 01_run_experiment.ipynb   Parametreler → çalıştırma → sonuçlar
│   ├── 02_compare.ipynb          Deney karşılaştırma
│   ├── 03_train.ipynb            STARK-ST eğitimi (aşama 1 / 2), veri seti kombinasyonlarıyla
│   ├── nbhelper.py               Notebook yardımcıları (yalnızca standart kütüphane)
│   └── colab_setup.py            Google Colab oturum hazırlığı (yalnızca standart kütüphane)
├── stark_ft/                     Deney altyapısı
│   ├── config.py                 ExperimentConfig: TÜM parametreler ve doğrulama
│   ├── paths.py                  Yol çözümleme
│   ├── runner.py                 VOT workspace hazırlığı + `vot evaluate` + sonuç toplama
│   ├── vot_entry.py              VOT-toolkit'in her dizi için başlattığı tracker süreci
│   ├── tracker_factory.py        Parametrelerden tracker nesnesi
│   ├── evaluation.py             COCO mAP/AP50/AP75, P/R/F1, F-maksimum eşik, legacy COCO
│   ├── analysis.py               Metrik tabloları + grafik üretimi
│   ├── plots.py                  Grafikler
│   ├── compare.py                Deney karşılaştırma
│   ├── setup_utils.py            Ortam kontrolü, indirme, duman testi
│   ├── vot_data.py               VOT-LT2020 dizileri: gerektiğinde indirme (yalnızca kullanılanlar) + önbellek
│   ├── training.py               TrainConfig: eğitim parametreleri, koşu klasörleri, devam etme, dışa aktarma
│   ├── train_data.py             Eğitim veri setlerini indirme / açma
│   └── __main__.py               CLI (python -m stark_ft ...)
├── lib/                          STARK çekirdeği
│   ├── models/stark/             Ağ mimarisi (orijinal STARK)
│   ├── config/                   Model config varsayılanları (orijinal STARK; stark_st1/ aşama-1 eğitiminde kullanılır)
│   ├── train/                    STARK eğitim kodu (orijinalden: veri yükleme, actor'lar, trainer)
│   ├── test/tracker/
│   │   ├── stark_st.py           STARK-ST tracker (orijinal + update kayıtları)
│   │   ├── stark_st_ft.py        ★ Fine-tune'lu tracker
│   │   ├── ft_sampling.py        ★ Pozitif jitter ve negatif bölge üretimi
│   │   └── stark_s.py            STARK-S tracker
│   └── utils/
├── model_configs/                STARK model YAML'ları (stark_st1/, stark_st2/, stark_s/)
├── configs/
│   ├── paths.yaml                Varsayılan yollar
│   ├── paths.local.example.yaml  Kişisel yol ayarı şablonu
│   └── experiments/example.yaml  CLI için örnek deney dosyası
├── environment/vot1_environment.yml
├── tests/                        python -m tests.<ad> (GPU gerekmez)
├── checkpoints/   data/   outputs/   (git'e girmez)
```

## 3. Yöntem

### 3.1 STARK-ST'de güven skoru

STARK-ST, her karede hedefin kutusunun yanında bir **güven skoru** üretir. Bu skor, transformer decoder
çıktısı `hs` (256 boyutlu) üzerinde çalışan 3 katmanlı bir MLP'nin (**cls_head**) sigmoid çıktısıdır.
Skor iki yerde kullanılır:

1. **Template update kararı:** Her `update_interval` karede bir, skor `update_conf_thr` (0.5) değerinden
   büyükse dinamik template o karedeki tahminle güncellenir.
2. **Long-term değerlendirme:** Hedefin görünmediği karelerde tracker'ın düşük skor vermesi beklenir.
   Bu repodaki metriklerde skor, "hedefi buldu / bulamadı" kararının temelidir.

### 3.2 Fine-tune

cls_head, base eğitimde genel bir "hedef var / yok" ayrımını öğrenir. Bu yöntemde cls_head her video
başında base ağırlıklarına döndürülür ve o videonun hedefine özel olarak kısa bir süre eğitilir.
Backbone, transformer ve box head **dondurulmuştur**; gradyan yalnızca cls_head'den akar.

Bir fine-tune **oturumu** şöyle işler:

```
kare + hedef kutusu
  ├─ pozitif kutu  = kutu (+ ft_pos_jitter ise ST2 eğitimindeki jitter)        → etiket 1
  └─ negatif kutu  = aynı karede hedefi HİÇ içermeyen bölge (yalnızca posneg)   → etiket 0
        │
        ▼  her kutu için: search bölgesi kırp → backbone → template'lerle birleştir → transformer → hs
        ▼
  cls_head(hs) → BCEWithLogitsLoss → AdamW (ft_lr, ft_weight_decay) → grad clip (ft_grad_clip_norm)
  (1–2 örnek olduğundan 1 epoch = 1 optimizasyon adımı; her oturumda yeni optimizer)
```

**Modlar (`ft_mode`)**

| Mod | Ne zaman fine-tune yapılır | Kutu kaynağı |
|---|---|---|
| `none` | Hiç (saf STARK-ST; baseline) | — |
| `init` | Yalnızca ilk karede, `ft_epochs_init` adım | Ground truth (VOT'un verdiği ilk kutu) |
| `online` | İlk karede `ft_epochs_init` adım **+** her template update yapılan karede `ft_epochs_online` adım | İlk kare: GT. Sonrası: **tracker'ın kendi tahmini** |

`online` modunda fine-tune yalnızca template update yapılan karelerde tetiklenir. Yani `update_interval=99999`
ile `online` fiilen `init` ile aynıdır.

**Örnek türleri (`ft_samples`)**

| Değer | Açıklama |
|---|---|
| `pos` | Yalnızca pozitif örnek. Yalnızca "1" etiketiyle eğitim, başlığın skorunu genel olarak yukarı iter; hedefin olmadığı karelerde de skor yükselir (bkz. §10). |
| `posneg` | Pozitif + negatif. Negatif, aynı karede, hedef kutusunun search kırpımının **tamamen dışında** kalacak şekilde kaydırılmış bir bölgedir (8 yön denenir, kırpımın görüntü içinde kalan oranı en yüksek olan seçilir). Bu, doğru negatif yöntemi bulunana kadar **geçici** bir çözümdür. |

## 4. Parametreler

Tüm parametreler `stark_ft/config.py` içindeki `ExperimentConfig` sınıfında tanımlıdır. Notebook'ta
`PARAMS = dict(...)` olarak, CLI'da YAML dosyası veya `--set anahtar=değer` olarak verilir.
Verilmeyen parametre varsayılan değerini alır.

### Kimlik ve model

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `name` | `None` | Çıktı klasörünün adı (`outputs/<name>/`). `None` ise parametrelerden üretilir, örn. `st101_online_pos_lr0.0001_i15_o1_int100_s0`. Boşluk ve `/\:*?"<>\|` içeremez. |
| `model` | `"stark_st"` | `"stark_st"` (güven skorlu, fine-tune edilebilir) veya `"stark_s"` (skor yok; sabit 1.0 raporlanır, `ft_mode="none"` olmalı). |
| `model_config` | `"baseline_R101"` | `model_configs/<stark_st2\|stark_s>/` altındaki YAML. `stark_st`: `baseline_R101` (ST101), `baseline` (ST50), `baseline_R101_got10k_only`, `baseline_got10k_only`. `stark_s`: `baseline`, `baseline_got10k_only`. |
| `checkpoint` | `None` | `None` ise resmi dosya adı kullanılır (`STARKST_ep0050.pth.tar` / `STARKS_ep0500.pth.tar`). Yalnızca dosya adı verilirse `<checkpoints>/<stark_st2\|stark_s>/<model_config>/` içinde aranır, örn. `"STARKSTcoco_ep0050.pth.tar"`. `/` içeren değer yol olarak kullanılır (göreli ise repo köküne göre). `"train:<koşu adı>"` bir eğitim koşusunun sonucunu kullanır (`<train_outputs>/<koşu adı>/final.pth.tar`, §7.5). |

### Veri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `sequences` | `"all"` | `"all"` (50 VOT-LT2020 dizisi) veya dizi adları listesi, örn. `["bull", "ballet"]`. Dataset klasöründe henüz olmayan diziler koşu başlarken indirilir (§1). Dataset klasöründeki kendi klasörleriniz de verilebilir. |

### Template update (yalnızca `stark_st`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `update_interval` | `100` | Kaç karede bir template update denenir. `99999` = hiç update yok. (Model YAML'larındaki `TEST.UPDATE_INTERVALS` bu repoda **kullanılmaz**; değer her zaman bu parametreden gelir.) |
| `update_conf_thr` | `0.5` | Update için skorun **büyük** olması gereken eşik (orijinal STARK: 0.5). |
| `max_template_updates` | `-1` | Bir dizide yapılabilecek en fazla template update sayısı. `-1` = sınırsız. |

### Fine-tune (yalnızca `stark_st`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `ft_mode` | `"online"` | `"none"`, `"init"`, `"online"` (bkz. §3.2). |
| `ft_samples` | `"pos"` | `"pos"` veya `"posneg"` (bkz. §3.2). |
| `ft_lr` | `1e-4` | AdamW öğrenme oranı (base Stage-2 eğitimi: 1e-4). |
| `ft_epochs_init` | `15` | İlk karedeki optimizasyon adımı sayısı. `0` = ilk karede fine-tune yok. |
| `ft_epochs_online` | `1` | Her online oturumdaki adım sayısı (`ft_mode="online"`). |
| `ft_weight_decay` | `1e-4` | AdamW weight decay (base eğitimle aynı). |
| `ft_grad_clip_norm` | `0.1` | Gradyan normu kırpma (base eğitimle aynı). |
| `ft_pos_jitter` | `True` | Pozitif kutuya ST2 eğitimindeki search jitter'ı uygulanır mı? `False` ise kutu olduğu gibi kullanılır. |
| `ft_center_jitter` | `4.5` | Jitter merkez kayma katsayısı (ST2: 4.5). Merkez, `sqrt(w·h)·4.5` genişliğinde düzgün dağılımla kayar. |
| `ft_scale_jitter` | `0.5` | Jitter ölçek katsayısı (ST2: 0.5). Boyutlar `exp(N(0,1)·0.5)` ile ölçeklenir. |
| `max_ft_updates` | `-1` | Bir dizide en fazla kaç online fine-tune oturumu yapılır. `-1` = sınırsız. |
| `seed` | `0` | Rastgelelik tohumu (jitter, PyTorch, cuDNN deterministik). Aynı seed ve aynı parametreler **birebir aynı** sonucu verir. Varyansı ölçmek için farklı seed'lerle tekrarlayın. |

### Değerlendirme

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `eval_score_thr` | `0.35` | **Sabit** eşik: "tracker hedefi buldu" kararı için skor eşiği. Yalnızca `precision / recall / F1` kolonlarını etkiler; mAP / AP50 / AP75'i ve F-maksimum eşik sonucunu etkilemez. |
| `eval_iou_thr` | `0.5` | Bulunan kutunun doğru sayılması için en düşük IoU (hem sabit eşikte hem F-maksimum aramasında). |
| `eval_thr_resolution` | `100` | F-maksimum eşik aramasındaki aday eşik sayısı (vot-toolkit varsayılanı: 100). |

Bu üç değer tracking'i tekrar çalıştırmadan değiştirilebilir: `nb.analyze(OUT, score_thr=0.5, thr_resolution=200)`.
Yalnızca bunları değiştirmek, deneyi "farklı parametreli" yapmaz (devam etme kuralını etkilemez).

### VOT-toolkit

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `run_redetection` | `False` | VOT-LT2020 stack'indeki ikinci deney (`redetection`) de koşulsun mu? Süreyi yaklaşık iki katına çıkarır; bu repodaki metrikler yalnızca `longterm` sonuçlarını kullanır. |
| `tracker_timeout` | `300` | Tracker'ın tek bir yanıtı için zaman aşımı (saniye). Model yükleme dahil. |

### Deney adı, devam etme ve üzerine yazma

- Aynı `name` + **aynı** parametrelerle tekrar çalıştırmak **kaldığı yerden devam eder**: VOT-toolkit
  tamamlanmış dizileri atlar. Kesilen bir koşuyu sürdürmek için hücreyi yeniden çalıştırmanız yeterlidir.
  (`eval_*` parametreleri bu karşılaştırmaya dahil değildir.)
- Aynı `name` + **farklı** parametreler hata verir ve eski sonuçlar korunur. Üzerine yazmak için
  `nb.run(PARAMS, overwrite=True)` (CLI: `--overwrite`).
- Deney başlatıldıktan sonra **tracking kodu değiştiyse** (örneğin bir `git pull` sonrası) devam etme de reddedilir:
  `lib/`, `model_configs/` ve tracker giriş noktasının bir özeti `experiment.json`'a yazılır; böylece iki kod
  sürümünün sonuçları asla karışmaz. Temiz bir koşu için `overwrite=True` kullanın.
- Bu yüzden `vot evaluate` **`-f` olmadan** çağrılır: her deneyin kendi workspace'i olduğu için başka deneylerden kalma
  sonuç yoktur, tamamlanan dizilerin atlanması da devam etmeyi mümkün kılar. `vot analysis` (ve onun `--nocache`
  seçeneği) kullanılmaz; metrikler her seferinde ham sonuçlardan yeniden hesaplanır.

## 5. Çıktılar ve dosya formatları

Her deney `outputs/<deney adı>/` altına yazılır.

| Dosya | İçerik |
|---|---|
| `experiment.json` | Tüm parametreler, checkpoint ve veri seti yolu, dizi listesi, git commit, tarih |
| `run.log` | vot-toolkit'in tam çıktısı. Bir dizi başarısız olursa tracker'ın hata çıktısı `vot_workspace/logs/` altına yazılır. |
| `run_status.json` | Tamamlanan / eksik diziler |
| `predictions/<dizi>/<dizi>_001.txt` | Kare başına kutu `x,y,w,h` (sol-üst köşe + boyut, piksel). **İlk satır `1`** (init karesi işareti, VOT formatı). |
| `predictions/<dizi>/<dizi>_001_confidence.value` | Kare başına güven skoru (ilk satır boş). |
| `predictions/<dizi>/<dizi>_001_time.value` | Kare başına süre (saniye). |
| `predictions/<dizi>/frames.csv` | Hepsi bir arada: `frame, x, y, w, h, conf, time, gt_visible, gt_x, gt_y, gt_w, gt_h, iou` |
| `tracker_logs/<dizi>/finetune_loss.txt` | CSV: `frame, session, epoch, loss, pos_prob, neg_prob, n_pos, n_neg, neg_coverage`. Her satır bir optimizasyon adımı. `pos_prob`/`neg_prob`: o adımdaki (güncellemeden önceki) sigmoid çıktısı. `neg_coverage`: negatif kırpımın görüntü içinde kalan oranı. |
| `tracker_logs/<dizi>/events.txt` | CSV: `frame, event, conf_score`. `event` = `template_update` veya `ft_update`. |
| `plots/<dizi>/iou_conf.png` | IoU ve güven skoru – kare numarası. Gri taralı alan: hedef yok. Kırmızı kesikli: template update. Yeşil noktalı: fine-tune update. Siyah kesik-noktalı: sabit eşik (`eval_score_thr`). Mor: F-maksimum eşik. |
| `plots/<dizi>/finetune_loss.png` | Üstte loss (log ölçek), altta pozitif/negatif örnek olasılıkları. Dikey çizgiler oturum başlangıçları (`init`, `f<kare>`). |
| `metrics/summary.txt` | Okunabilir özet (ilk satırlar: mAP / AP50 / AP75, ardından F-maksimum eşik ve sabit eşik sonuçları) + dizi bazında tablo |
| `metrics/metrics.xlsx` | `summary` (kolonlar: `mean over sequences`, `pooled`), `optimal_threshold`, `per_sequence`, `f_curve`, `parameters` sayfaları |
| `metrics/f_curve.csv` | Her aday eşikte dizi ortalaması `precision`, `recall`, `F` |
| `metrics/f_curve.png` | Solda P / R / F – eşik (mor: seçilen eşik, siyah: sabit eşik), sağda precision–recall eğrisi |
| `metrics/metrics.json` | Aynı veriler, makine tarafından okunabilir |
| `vot_workspace/` | Deneye özel VOT workspace'i (`config.yaml`, `trackers.ini`, ham `results/`) |

**Kare numaralandırması:** Her yerde `frame = 0` ilk (init) karedir. VOT sonuç dosyalarında bu, satır numarasının
1 eksiğidir. İlk kare değerlendirmeye girmez (orada GT tracker'a verilir).

## 6. Metrikler

Değerlendirme **detection** tarzındadır: her kare bir "görüntü", hedef tek nesnedir. Hesap `pycocotools`
kütüphanesiyle yapılır (`stark_ft/evaluation.py`).

**Birincil metrikler: mAP, AP50 ve AP75** (§6.1). Her özette, tabloda ve karşılaştırmada ilk sırada yer alırlar.

### 6.1 mAP / AP50 / AP75 (standart COCO kullanımı)

- **mAP**, COCO'nun ana "AP" metriğidir (`COCOeval`'in `stats[0]` değeri): precision, recall seviyeleri ve
  0.50, 0.55, …, 0.95 IoU eşikleri üzerinden ortalanır. **AP50** ve **AP75** tek bir IoU eşiği (0.50 / 0.75) kullanır.
- Özetlerde "mean over sequences" dizi değerlerinin ortalamasıdır (eski `coco_eval.py`'deki MEAN satırı gibi);
  "pooled" tüm dizilerin karelerini tek bir veri seti gibi değerlendirir.
- İlk kare hariç **tüm** kareler değerlendirmeye girer.
- Hedefin görünmediği (GT = NaN) kareler, annotation'ı olmayan görüntü olarak eklenir. Bu karelerde verilen
  her tahmin **false positive** sayılır.
- **Skor eşiği uygulanmaz**: tüm tahminler skorlarıyla verilir. mAP, skora göre sıralamanın kalitesini
  zaten tüm eşikler üzerinden ölçer. Eşik koymak mAP / AP50 / AP75'i hiçbir zaman artıramaz.
- Koordinatlar float olarak kullanılır.

### 6.2 Eşikte "buldu / bulmadı" metrikleri

Skor ≥ `eval_score_thr` ise tracker "buldum" demiş sayılır:

| | Tracker "buldum" (skor ≥ eşik) | Tracker "bulmadım" (skor < eşik) |
|---|---|---|
| **Hedef görünür**, IoU ≥ `eval_iou_thr` | TP | FN |
| **Hedef görünür**, IoU < `eval_iou_thr` | FP + FN (yanlış yer) | FN |
| **Hedef yok** | FP | TN |

- `precision = TP / (TP + FP)`, `recall = TP / (TP + FN)`, `F1 = 2PR / (P + R)`
- `absent_reject_rate = TN / (hedefin olmadığı kare sayısı)`: hedef yokken tracker'ın doğru şekilde
  "bulmadım" dediği karelerin oranı.

Bu sayım COCOeval ile aynı eşleştirme kuralını kullanır (IoU ≥ `eval_iou_thr`, kare başına tek GT); COCOeval'in
kendi eşleştirmelerinden sayılan P/R ile birebir aynı sonucu verdiği doğrulanmıştır.

### 6.3 F-maksimum eşik (VOT-LT yöntemi)

`eval_score_thr` sabit bir eşiktir. Asıl karşılaştırma ölçütü, VOT-LT protokolündeki gibi **F'yi maksimize eden
eşikte** hesaplanan F'dir. Yöntem vot-toolkit 0.5.3'ün `vot/analysis/tpr.py` dosyasıyla aynıdır; tek fark P ve R'nin
nasıl sayıldığıdır:

1. **Aday eşikler:** Deneydeki tüm dizilerin skorları (init karesi hariç) havuzlanıp büyükten küçüğe sıralanır.
   Bu sıradan eşit aralıklı `eval_thr_resolution − 2` değer seçilir, başa +∞ ve sona −∞ eklenir
   (vot-toolkit'in `determine_thresholds` fonksiyonunun birebir kopyası).
2. **Her eşikte, her dizi için P ve R:** vot-toolkit IoU ağırlıklı P/R kullanır. Burada §6.2'deki
   "buldu / bulmadı" sayımı kullanılır (COCOeval'in eşleştirmesiyle aynı: IoU ≥ `eval_iou_thr`).
   Eşiği geçen tahmin yoksa vot-toolkit'teki gibi P = 1, R = 0.
3. **Birleştirme:** P ve R diziler üzerinden ortalanır. F = 2PR / (P + R), bu ortalamalardan hesaplanır
   (dizi F'lerinin ortalaması değildir).
4. **Seçim:** F'nin en büyük olduğu eşik seçilir (eşitlikte en yüksek eşik). Bu eşik **tüm diziler için tektir**.

Çıktıda: `optimal_threshold` (eşik, P, R, F), dizi bazında `precision_opt / recall_opt / F_opt`
(her dizinin ortak eşikteki değeri), `metrics/f_curve.csv` ve `f_curve.png`.

> Eşik GT kullanılarak seçilir. Bu, VOT-LT'nin standart raporlama protokolüdür (her tracker kendi en iyi eşiğinde
> karşılaştırılır); tracker çalışırken bu eşiği bilmez.

Doğrulama: aday eşik listesi vot-toolkit'in kendi fonksiyonuyla aynıdır; vektörel P/R hesabı tek tek sayımla
aynıdır; seçilen eşik kaba kuvvetle bulunanla aynıdır (50 dizide test edildi).

### 6.4 Diğer

- `mean_iou_visible`: hedefin görünür olduğu karelerdeki ortalama IoU (eşik yok).
- `legacy_mAP / legacy_AP50 / legacy_AP75`: eski `testler/detailed_analysis/coco_eval.py` ile **birebir aynı**
  hesap (hedefsiz kareler dışarıda, skoru eşiğin altındaki tahminler atılır, koordinatlar tam sayıya yuvarlanır).
  Yalnızca eski sonuçlarla karşılaştırma içindir; hedefsiz karelerdeki yanlış tespitleri cezalandırmaz.
  5 dizide eski betikle aynı sonucu verdiği doğrulanmıştır.
- **Dizi ortalaması:** her dizinin metriğinin ortalaması (eski `MEAN` satırıyla aynı mantık). İstisna: `F1` ve
  `F_opt`, VOT tanımıyla ortalama P ve R'den hesaplanır.
- **Havuzlanmış (pooled):** tüm dizilerin kareleri tek veri seti gibi; uzun diziler daha çok ağırlık alır.

## 7. STARK-ST eğitimi (base training)

`notebooks/03_train.ipynb`, STARK-ST modelinin kendisini **orijinal STARK eğitim prosedürüyle** (`lib/train/`,
resmi repodan taşındı) ve **eğitim veri setlerinin herhangi bir kombinasyonuyla** eğitir: GOT-10k, COCO, LaSOT ve
TrackingNet'ten tek biri, herhangi ikisi, üçü ya da dördü. GOT-10k ve COCO otomatik olarak indirilir / açılır.
Eğitilen model, resmi model gibi, inference sırasında fine-tune ile veya fine-tune olmadan
`checkpoint="train:<koşu adı>"` ile değerlendirilir (§4).

### 7.1 Hızlı başlangıç

1. `notebooks/03_train.ipynb` dosyasını açın (Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/03_train.ipynb); Colab'de **A100** veya **L4** GPU seçin).
2. Parametreleri (`TRAIN = dict(...)`, §7.4) ve GOT-10k için indirme linklerini (§7.3) yazın.
3. *Hazırlık* hücresini çalıştırın: Colab'de oturum hazırlanır (ortam, checkpoint); ardından veri setleri indirilir /
   açılır (yalnızca eksik olanlar) ve bir deneme çalıştırması (dry run) koşu klasörünü, epoch ve adım sayılarını ve
   başlangıç ağırlıklarını gösterir.
4. *Eğitim* hücresini çalıştırın. İlerleme satırları saniyedeki örnek sayısını, epoch'un ve eğitimin kalan süresini
   gösterir. Hücreyi durdurmak koşuyu durdurur; tekrar çalıştırmak son tamamlanan epoch'tan **devam eder**.
5. `nb.show_training(TRAIN)` ilerlemeyi ve `history.png` grafiğini istediğiniz zaman gösterir.

### 7.2 Ne eğitilir

| Aşama | Eğitilen parametreler | Epoch (LR ÷10) | Loss | Başlangıç |
|---|---|---|---|---|
| 1 | backbone + transformer + kutu başlığı (sınıflandırma başlığı kullanılmaz) | 500 (400'de) | GIoU × 2 + L1 × 5 | ImageNet ResNet backbone |
| 2 | yalnızca sınıflandırma başlığı (geri kalan her şey dondurulur) | 50 (40'ta) | BCE | aşama-1 ağırlıkları (`init`) |

İki aşamada da (`model_configs/stark_st1/` ve `model_configs/stark_st2/` altındaki YAML'lardan): epoch başına 60 000
eğitim örneği, AdamW (lr 1e-4, backbone için × 0.1, weight decay 1e-4), norm 0.1'de gradyan kırpma, 20 (aşama 1) /
10 (aşama 2) epoch'ta bir 10 000 örnekle doğrulama, seed 42, deterministik cuDNN. Template / arama karelerinin
örneklenmesi, augmentation'lar ve loss'lar orijinal koddakilerle aynıdır.

**Aşama 2 her zaman `init` ister**; böylece dondurulan kısmın hangi veriyle eğitildiği her zaman açıktır:
- `init="<aşama-1 koşu adı>"`: kendi aşama-1 koşunuz. Yalnızca belirli veri setleriyle eğitilmiş bir model için önce
  aşama 1'i o veri setleriyle, sonra aşama 2'yi aynı veri setleriyle bu `init` ile çalıştırın
  (örn. `st101_stage1_coco_s42` → `st101_stage2_coco_from-st101_stage1_coco_s42_s42`).
- `init="official"`: **sınıflandırma başlığı çıkarılmış resmi STARK-ST checkpoint'i**. Bu checkpoint'in backbone,
  transformer ve kutu başlığı resmi aşama-1 ağırlıklarıdır (aşama 2 bunları dondurur). Yani bu, aşama 2'yi ayrıca
  yayımlanmamış resmi aşama-1 modelinden başlatmakla aynıdır. Bu ağırlıklar **dört veri setinin tamamıyla** (LaSOT,
  GOT-10k, COCO, TrackingNet) eğitilmiştir; koşu adında `from-official` yazar. İstisna:
  `model_config="baseline_R101_got10k_only"` (veya `baseline_got10k_only`) ile resmi ağırlıklar **yalnızca GOT-10k**
  ile eğitilmiştir (9 335 train videosunun tamamı = `got10k_full`); bu, GOT-10k için aşama-1 koşusunu gereksiz kılar.

### 7.3 Eğitim veri setleri

| `datasets` anahtarı | STARK veri seti | Video / görüntü | Boyut (açılmış) | Hazırlık |
|---|---|---|---|---|
| `got10k` | `GOT10K_vottrain`: **VOT ile örtüşen 1000 video çıkarılmış** GOT-10k train | 7 086 video | 73.9 GB (GOT-10k train klasörünün tamamı) | resmi arşivlerden (kayıt gerekir, aşağıya bakın) |
| `got10k_full` | `GOT10K_train_full`: GOT-10k train'in tamamı (GOT-10k protokolü; `*_got10k_only` config'leri) | 9 335 video | (aynı klasör) | (aynı) |
| `coco` | `COCO17`: COCO 2017 train; her nesne tek karelik bir "video" | 118 287 görüntü | 19.3 GB (+ 0.8 GB anotasyon) | images.cocodataset.org'dan **otomatik** (19.6 GB zip) |
| `lasot` | `LASOT` (train bölümü) | 1 120 video | çok büyük | elle: `<train_data>/lasot/<sınıf>/<sınıf>-<n>/` |
| `trackingnet` | `TRACKINGNET` | ≈ 30 000 video | ≈ 1 TB | elle: `<train_data>/trackingnet/TRAIN_0 ... TRAIN_11/` |

Doğrulama (`val_datasets=["got10k"]`), `GOT10K_vottrain` ile **kesişmeyen** 1 249 GOT-10k train videosundan oluşan
`GOT10K_votval` bölümünü kullanır (STARK'taki gibi). Bu yüzden yalnızca COCO ile eğitimde bile GOT-10k klasörü
gerekir; doğrulamasız eğitim için `val_datasets=[]` verin.

**Klasör yapısı** (`train_data`, varsayılan `data/train/`; Colab'de yerel disk `/content/train_data`):

```
<train_data>/
├── got10k/train/list.txt, GOT-10k_Train_000001/ ... GOT-10k_Train_009335/
├── coco/annotations/instances_train2017.json
├── coco/images/train2017/*.jpg
├── lasot/ ...                      (elle)
├── trackingnet/TRAIN_0 ... TRAIN_11 (elle)
└── _archives/{coco,got10k}/        indirilen arşivler (Colab'de: MyDrive/LOKAP/train_archives/)
```

**GOT-10k** ancak [got-10k.aitestunion.com/downloads](http://got-10k.aitestunion.com/downloads) adresinde ücretsiz
kayıttan sonra indirilebilir; indirme linkleri e-postayla gelir. Linkler Google Drive'a da işaret edebilir, örn.
`full_data.zip` (70.7 GB; içinde `train/`, `val/` ve `test/` vardır). Arşivler üç şekilde verilebilir:

1. **Colab'de, Google Drive'daki bir dosya için (önerilen):** dosyanın **kendi kopyanızı**
   `MyDrive/LOKAP/train_archives/got10k/` klasörüne koyun ve `GOT10K_URLS` listesini boş bırakın: linki tarayıcıda
   Colab'de kullandığınız Google hesabıyla açın, *Drive'a kısayol ekle*, kısayola sağ tık → *Kopyasını oluştur*,
   oluşan kopyayı (`Copy of full_data.zip`; adı önemli değil) o klasöre taşıyın. Kopya Google Drive'ın içinde yapılır
   (hiçbir şey indirilmez) ve Drive'da 70.7 GB yer kaplar; arşiv sonra her yeni oturumda Drive'dan okunup yerel diske
   açılır. Neden kopya: bu dosya gibi popüler paylaşılan dosyalar Google Drive'ın günlük indirme sınırına sık sık
   takılır ("Quota exceeded"); o durumda ne indirme ne kısayol çalışır, ama kendi kopyanız çalışır. Yalnızca kısayol
   (Drive'da yer gerektirmez) ancak paylaşılan dosya sınırın altındayken çalışır.
2. `GOT10K_URLS` içinde **linkler** (CLI: `--got10k-url`): arşiv klasörüne bir kez indirilir, kaldığı yerden devam
   edebilir. Google Drive paylaşım linkleri (`https://drive.google.com/file/d/<id>/view...`) doğrudan indirme linkine
   çevrilir. Colab'de arşiv klasörü Drive'da olduğu için 70 GB'lık indirme mount üzerinden Drive'a yazılır; 1. yol
   bunu önler.
3. Zaten bir yerde duran **dosyalar**: `<archives>/got10k/` klasörüne koyun ya da yollarını (veya bir klasörü)
   `GOT10K_URLS` içinde verin; yollar yerinde kullanılır ve asla silinmez.

Yalnızca train videoları ve `list.txt` dosyaları açılır (`val/` ve `test/` atlanır); arşiv içindeki arşivlerden
(örn. train split zip'leri) de. Hazırlık 9 335 train videosunun tamamının bulunduğunu kontrol eder ve resmi
`list.txt` dosyasını alır (yoksa birebir aynısını oluşturur: sıralı isimler; `data_specs` bölüm dosyaları bu sıraya
göre indeksler).

**COCO** otomatik indirilir (`train2017.zip`, `annotations_trainval2017.zip`); yarıda kalan indirme kaldığı yerden
devam eder ve 118 287 görüntünün tamamı kontrol edilir.

**Hazırlığın davranışı:** arşivler saklanır (`delete_archives=False`), böylece yeni bir Colab oturumu ya da yeni bir
makine yalnızca açma işlemini yapar; tamamlanan bir indirme URL'siyle hatırlanır (e-postadaki linklerin süresi
dolabilir); açma işlemi geçici bir klasöre yapılır ve yalnızca tamamlandığında yeniden adlandırılır (yarıda kalan
bir hazırlık yarım iş bırakmaz); açmadan önce boş disk alanı kontrol edilir; beklenen yapıya sahip bir veri seti
klasörü yalnızca okunur, asla değiştirilmez. Zaten açılmış bir GOT-10k kopyası, `<train_data>/got10k/train/list.txt`
var olacak şekilde `train_data` ayarlanarak doğrudan kullanılabilir.

**Colab'de alan:** Google Drive arşivleri (COCO 19.6 GB, GOT-10k ≈ train arşivlerinin boyutu) ve eğitim koşularını
(§7.5) tutar; oturumun yerel diski açılmış veri setlerini (GOT-10k ≈ 74 GB, COCO ≈ 20 GB) ve ortamı (≈ 8 GB) tutabilmelidir.
Disk yetmezse hazırlık, ne kadar alan gerektiğini söyleyen bir mesajla durur.

### 7.4 Parametreler (`TrainConfig`, `stark_ft/training.py`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `name` | `None` | Koşu klasörünün adı (`<train_outputs>/<name>/`). `None` ise parametrelerden üretilir: model, aşama, veri setleri, `from-<init>`, `e<epochs>`, `r<ratios>`, seed; örn. `st101_stage1_coco_s42`, `st101_stage2_coco_from-st101_stage1_coco_s42_s42`. |
| `model_config` | `"baseline_R101"` | `baseline_R101` (STARK-ST101) veya `baseline` (STARK-ST50); `model_configs/stark_st1/` (aşama 1) veya `stark_st2/` (aşama 2) altındaki YAML. `*_got10k_only` YAML'ları yalnızca veride farklıdır (onlarla `datasets=["got10k_full"]` kullanın); aşama 2'de `init` olarak resmi GOT-10k-only checkpoint'ini seçerler. |
| `stage` | `1` | `1` veya `2` (§7.2). |
| `init` | `None` | Aşama 1: `None` olmalı (ImageNet backbone). Aşama 2 (**zorunlu**): tamamlanmış bir aşama-1 koşusunun adı, bir checkpoint yolu veya `"official"` (sınıflandırma başlığı çıkarılmış, dört veri setiyle eğitilmiş resmi checkpoint; §7.2). |
| `datasets` | `["got10k"]` | `got10k` (veya `got10k_full`), `coco`, `lasot`, `trackingnet` değerlerinin boş olmayan herhangi bir kombinasyonu. Sıra önemli değildir (STARK'ın sırasına getirilir). |
| `dataset_ratios` | `None` | Veri seti başına örnekleme ağırlığı (`datasets` ile aynı sırada). `None` = eşit ağırlık (STARK'taki gibi): her eğitim örneği önce bu ağırlıklarla bir veri seti, sonra o veri setinden bir video seçer. |
| `val_datasets` | `["got10k"]` | `["got10k"]` (GOT10K_votval) veya `[]` (doğrulama yok). |
| `epochs` | `None` | `None` = orijinal (aşama 1: 500, aşama 2: 50). |
| `lr_drop_epoch` | `None` | Öğrenme oranının 10'a bölündüğü epoch. `None` = orijinal (400 / 40). |
| `samples_per_epoch` | `None` | Epoch başına eğitim örneği. `None` = 60 000. |
| `val_samples_per_epoch` | `None` | Doğrulama örneği sayısı. `None` = 10 000. |
| `val_interval` | `None` | Kaç epoch'ta bir doğrulama yapılacağı. `None` = orijinal (20 / 10). |
| `effective_batch` | `128` | Optimizer adımı başına örnek sayısı. **128 = orijinal** (8 GPU × 16). Değiştirmek eğitimi değiştirir. |
| `micro_batch` | `16` | Bir ileri/geri geçişteki örnek sayısı; her adımda `effective_batch / micro_batch` geçişin gradyanı biriktirilir (§7.6). Yalnızca bellek ve hız buna bağlıdır. |
| `num_workers` | `8` | Veri yükleme süreçleri. |
| `seed` | `42` | Seed (STARK varsayılanı). |
| `keep_every` | `None` | Kaç epoch'ta bir ağırlıkların saklanacağı (`None`: aşama 1: 50, aşama 2: 10). |

### 7.5 Koşu klasörü, devam etme, sonucu kullanma

Her koşu `<train_outputs>/<koşu adı>/` klasörüne yazılır (varsayılan `outputs/training/`; Colab'de
`MyDrive/LOKAP/training/`):

| Dosya | İçerik |
|---|---|
| `train_config.json` | Tüm parametreler, veri seti klasörleri, başlangıç ağırlıkları, GPU, TF32 ayarı, kod hash'i, tarih |
| `history.csv` | Epoch başına bir satır: süre, öğrenme oranı, eğitim (ve doğrulama) loss'ları / IoU |
| `history.png` | `history.csv` grafiği (`nb.show_training` / `train-report` yazar; kesikli çizgiler: LR düşüşleri) |
| `logs/train.log` | İlerleme çıktısı |
| `checkpoints/latest.pth.tar` | Son tamamlanan epoch'tan sonraki eksiksiz durum (ağ, optimizer, LR scheduler, rastgele sayı üreteçleri); her epoch atomik olarak yazılır. ST101: ≈ 0.56 GB (aşama 1), ≈ 0.19 GB (aşama 2). |
| `checkpoints/STARKST_epXXXX.pth.tar` | Her `keep_every` epoch'ta ve sonda ağ ağırlıkları (her biri ≈ 0.19 GB) |
| `final.pth.tar` | Koşu bittiğinde yazılan son ağırlıklar |

- **Devam etme:** aynı parametreler → koşu `latest.pth.tar` dosyasından devam eder. Devam eden koşu, hiç kesilmemiş
  bir koşuyla **bit düzeyinde aynıdır** (eğitim sırasında öldürülüp devam ettirilen bir koşuyla doğrulandı). Aynı
  `name` için farklı parametreler → hata (başka bir `name` veya `overwrite=True` kullanın). Eğitim kodunun
  (`lib/train`, `lib/models`, `lib/config`, `lib/utils`, `model_configs/stark_st1|2`, `stark_ft/training.py`) farklı bir sürümüyle başlatılmış bir koşu da devam ettirilmez.
- **Aşama-2 koşusunu değerlendirme:** `01_run_experiment.ipynb` içinde aynı `model_config` ile
  `checkpoint="train:<koşu adı>"` (aşama-1 koşularının sınıflandırma başlığı eğitilmemiştir; onları aşama 2'nin
  `init`'i olarak kullanın).
- **Aşama 1 → aşama 2:** `TrainConfig(stage=2, init="<aşama-1 koşu adı>", ...)`.
- `checkpoints` klasörüne hiçbir şey yazılmaz.

### 7.6 Sekiz yerine tek GPU: effective batch ve süre

Orijinal model, her birinde 16 örnek olan 8 GPU ile eğitildi; 8 GPU'nun gradyanlarının ortalaması alındığı için her
optimizer adımı **128** örneğin ortalama gradyanını kullanır. Burada loader `micro_batch` örneklik mikro-batch'ler
üretir; her birinin loss'u `effective_batch / micro_batch` sayısına bölünür, gradyanlar biriktirilir ve ancak ondan
sonra gradyan kırpılıp optimizer adımı atılır. STARK backbone'daki tüm BatchNorm katmanlarını dondurduğu için bir
örneğin ileri geçişi batch'teki diğer örneklere bağlı değildir; dolayısıyla bu, orijinalle **aynı güncellemedir**.
Sayısal olarak kontrol edildi: 8 × 16 örneğin biriktirilmiş gradyanı ile tek bir 128 örneklik batch'in gradyanı
float64'te 3·10⁻¹⁵ (bağıl) farklıdır; float32'de ≈ 10⁻⁴, yalnızca toplama sırasından. Epoch başına optimizer adımı
sayısı da aynıdır (60 000 / 128 = 468; artan örnekler kullanılmaz).

Birebir yeniden üretilemeyen şey rastgele örnek akışıdır: orijinalde her biri kendi veri işçilerine sahip 8 süreç
vardı. Aynı GPU tipinde aynı parametrelerle (`micro_batch` ve `num_workers` dahil) yapılan koşular bit düzeyinde
aynıdır; `micro_batch` / `num_workers` değişirse yöntem değil, çekilen rastgele örnekler değişir.

**Süre.** Aşama 1, 500 × 60 000 = 30 M örnek; aşama 2, 50 × 60 000 = 3 M örnek işler. RTX 3060 laptop GPU'da
ölçülen (`micro_batch=16`): aşama 2 ≈ 35 örnek/sn (≈ 24 saat), aşama 1 ≈ 20 örnek/sn (≈ 17 gün). Veri merkezi
GPU'ları birkaç kat hızlıdır; ilk adımlardan sonra ilerleme çıktısındaki eğitim ETA'sına bakın. Colab oturumları
birkaç saat sonra biter, ancak tamamlanan her epoch kaydedilir ve koşu sonraki oturumda devam eder. Colab compute
unit'leri saat başına harcanır; aşama-1 koşusuna başlamadan önce maliyetine bakın. **TF32:** A100 / L4 (ve RTX
3000 / 4000 serisi) konvolüsyonlarda varsayılan olarak TF32 kullanır, T4 kullanmaz (bkz. [GPU'lar arası sonuçlar](#gpular-arası-sonuçlar));
karşılaştırdığınız koşular için aynı GPU tipini kullanın.

### 7.7 Orijinal STARK eğitim kodundan farklar

| Orijinal STARK | Burada |
|---|---|
| 8 GPU (DistributedDataParallel), GPU başına 16 örnek | 1 GPU, aynı effective batch'e gradyan biriktirme ile (§7.6) |
| Veri seti yolları makineye özel bir `local.py` içinde | `train_data` (§1) ve `datasets` parametresi; herhangi bir kombinasyon |
| Devam ettirilen bir aşama-2 koşusu, devam ettikten **sonra** aşama-1 ağırlıklarını yeniden yüklüyor ve zaten eğitilmiş sınıflandırma başlığını eziyordu | Başlangıç ağırlıkları yalnızca koşu sıfırdan başlarken yüklenir |
| Checkpoint yalnızca son 10 epoch'ta ve 100 epoch'ta bir, LR scheduler ve rastgele sayı üreteci durumları olmadan yazılıyordu (bir kesinti 99 epoch'a kadar kayba yol açabilirdi) | Her epoch'tan sonra eksiksiz durumla atomik olarak yazılan `latest.pth.tar` + her `keep_every` epoch'ta ağırlıklar; devam etme bit düzeyinde aynı |
| Checkpoint'lerde pickle'lanmış settings nesneleri | Düz sözlükler |
| Yeni PyTorch / pandas ile çalışmıyordu (`torch._six`, `storage()._new_shared`, `read_csv(squeeze=True)`) | Davranış değiştirilmeden düzeltildi |

Orijinaldeki gibi bırakılan: aşama 1'de ağ, bir (mikro-)batch'teki herhangi bir örnek için geçersiz bir kutu
(x2 < x1 veya y2 < y1) tahmin ederse GIoU hesabı başarısız olur ve STARK o batch'in tamamının GIoU loss'unu 0 sayar
(yalnızca L1 loss kullanılır). Bu, sıfırdan eğitimin en başında olur (ilk adımlarda örneklerin ≈ 8'de 1'i geçersizdir)
ve hızla kaybolur. Burada mikro-batch başına uygulanır (varsayılan 16 örnek; orijinalde 8 GPU'nun her birindekiyle
aynı).

## 8. Komut satırı (CLI)

Notebook'lar bu komutları çağırır; doğrudan da kullanılabilir (`vot1` ortamında, repo kökünde):

```bash
python -m stark_ft check                                   # ortam + yollar
python -m stark_ft show --config configs/experiments/example.yaml
python -m stark_ft smoke --set ft_mode=online --frames 50  # VOT'suz hızlı test
python -m stark_ft run --config configs/experiments/example.yaml --set 'sequences=[bull]' --set ft_samples=posneg
python -m stark_ft analyze outputs/<deney> --score-thr 0.5 --thr-resolution 100
python -m stark_ft list
python -m stark_ft compare <deney1> <deney2> --out comparison.xlsx --plot comparison.png
python -m stark_ft download-checkpoints --model stark_st --model-config baseline_R101 baseline
python -m stark_ft download-dataset --sequences bull ballet  # isteğe bağlı: deneyler kullandıklarını indirir
python -m stark_ft prepare-train-data --datasets got10k coco [--got10k-url URL ...] [--archives KLASÖR]
python -m stark_ft train --set stage=2 --set 'datasets=[got10k, coco]' --dry-run   # kontrol; sonra --dry-run olmadan
python -m stark_ft train-report <koşu adı>                 # ilerleme + history.png
python -m stark_ft train-list
python -m tests.test_sampling                              # negatif örnek geometrisi testleri
python -m tests.test_config                                # parametre okuma / doğrulama testleri
python -m tests.test_training                              # eğitim parametreleri testleri
python -m tests.test_train_data                            # eğitim verisi hazırlığı testleri
python -m tests.test_vot_data                              # VOT dizisi indirme testleri
```

`--set` değerleri YAML olarak yorumlanır: `1e-4` → sayı, `true` → bool, `[a, b]` → liste.

## 9. Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| `Conda environment 'vot1' not found` | `00_setup.ipynb` 1. adım. Ortam başka yerdeyse `VOT1_PYTHON=/yol/envs/vot1/bin/python` ortam değişkenini tanımlayın. |
| `Checkpoint not found` | `00_setup.ipynb` 3. adım veya dosyayı hata mesajındaki konuma koyun. |
| Google Drive "kota aşıldı" | Hata mesajındaki linkten tarayıcıyla indirip belirtilen klasöre koyun. |
| `Unknown sequence(s): [...]` | `sequences` içinde yazım hatası; hata mesajı 50 VOT-LT2020 dizisinin adlarını listeler. |
| VOT dizisi indirilemiyor / `Checksum mismatch` | Ağ veya sunucu sorunu. Aynı hücreyi tekrar çalıştırın: biten diziler korunur, yarım kalan dosya kaldığı yerden devam eder. |
| `Completed sequences: 47/50` gibi eksik | Hata veren dizi atlanır, diğerleri koşmaya devam eder. Tracker'ın hata çıktısı `outputs/<deney>/vot_workspace/logs/` içindedir. Aynı deneyi tekrar çalıştırmak tamamlananları atlar, eksikleri yeniden dener. |
| `no kernel image is available` / sm_120 uyarısı | GPU, ortamdaki PyTorch tarafından desteklenmiyor (RTX 5000 serisi). RTX 3000/4000 serisi kullanın. |
| "A newer version of the VOT toolkit is available" | Görmezden gelin; repo vot-toolkit **0.5.3** ile test edilmiştir. Güncellemeyin. |
| Colab: `WARNING: no GPU in this session` | *Runtime → Change runtime type → GPU* seçip ilk hücreyi tekrar çalıştırın. |
| Colab: Drive'a arşivleme başarısız | Google Drive'da yeterli yer yok (≈ 8 GB + kullandığınız diziler gerekir). Yer açıp ilk hücreyi tekrar çalıştırın; tamamlanan adımlar tekrarlanmaz. |
| Eğitim: `CUDA out of memory` | `micro_batch` değerini düşürün (örn. 8) ve `effective_batch=128` olarak bırakın: güncelleme aynı kalır (§7.6). |
| Eğitim: oturum / makine çöküyor ya da RAM yetmiyor | `num_workers` değerini düşürün (COCO'nun anotasyon dosyası büyüktür ve her veri işçisi bellek kullanır). Colab'de High-RAM oturumu kullanın. |
| Eğitim: `Not enough disk space for ...` | Açılmış veri setleri yerel diske sığmıyor (GOT-10k ≈ 74 GB, COCO ≈ 20 GB). Daha büyük diskli bir makine / oturum ya da daha az veri seti kullanın. |
| Eğitim: `No GOT-10k archives in ...` | GOT-10k sitesine kaydolup e-postayla gelen linkleri `GOT10K_URLS` listesine yazın veya arşivleri mesajda belirtilen klasöre koyun (§7.3). |
| Eğitim: `... already exists with different parameters` / `different version of the training code` | Deneylerdeki gibi: başka bir `name` ya da sıfırdan koşu için `overwrite=True` kullanın. |

## 10. Bilinen sınırlamalar ve açık konular

- **`pos` modu skoru genel olarak şişirir.** Tek etiket "1" iken BCE'nin en kolay çözümü çıktıyı her girdi için
  büyütmektir. AdamW gradyanı normalize ettiği için loss çok küçükken bile her adım ≈ `ft_lr` kadar ilerler.
  Sonuçta hedefin olmadığı karelerde de skor yükselir (`absent_reject_rate` düşer) ve template update daha sık tetiklenir.
- **`posneg` negatifleri geçicidir.** Aynı kareden alınan negatif bölge, hedefe benzeyen başka nesneler
  (ör. başka dansçılar) içerebilir. `finetune_loss.txt` dosyasındaki yüksek `neg_prob` değerleri bunu gösterir.
  Doğru negatif üretimi bulunduğunda yalnızca `lib/test/tracker/ft_sampling.py` ve `stark_st_ft.py` içindeki
  `posneg` dalı değişecek şekilde tasarlanmıştır.
- **Online mod tracker'ın kendi tahminini pozitif kabul eder.** Template update kararını da fine-tune edilen başlığın
  skoru verdiği için kendini doğrulayan bir döngü riski vardır.
- **`ft_lr` / `ft_epochs_*` için teorik bir seçim yöntemi yok.** Değerler deneyseldir. Hiperparametreleri test
  dizileri üzerinde seçmenin sonuçları iyimser göstereceğini unutmayın.
- **vot-toolkit her dizi için tracker sürecini yeniden başlatır** (model her dizide yeniden yüklenir, ~3–4 sn).
- **Eğitim: LaSOT ve TrackingNet otomatik hazırlanmaz.** Eğitim kodu bunları destekler (`datasets=["lasot", ...]`),
  ancak `<train_data>/lasot/` ve `<train_data>/trackingnet/` klasörlerine elle indirilip açılmaları gerekir
  (Colab oturumu için çok büyüktürler).
- **Eğitim: tam bir aşama-1 koşusu uzundur** (30 M örnek; RTX 3060 laptop GPU'da ≈ 17 gün), bkz. §7.6.

### GPU'lar arası sonuçlar

Aynı ortam, yalnızca **aynı GPU tipinde** bit düzeyinde aynı sonucu verir. Farklı GPU'lar farklı sayısal çekirdekler
kullanır; ayrıca Ampere ve sonrası GPU'larda (RTX 3000/4000, A100, ...) PyTorch konvolüsyonları varsayılan olarak TF32
ile hesaplarken T4 (Colab) gibi eski GPU'lar tam FP32 hesaplar. Tracking her kareyi bir sonrakine aktardığı için çok
küçük sayısal farklar büyüyebilir.

Örnek (`bull`, online, pos, 15+15 adım, lr 1e-5, interval 100; legacy mAP):

| Koşu | legacy mAP |
|---|---|
| RTX 3060 (TF32 açık, varsayılan), 8 seed | 0.521 ± 0.001 |
| RTX 3060, TF32 kapalı (`NVIDIA_TF32_OVERRIDE=0`) | 0.458 |
| Tesla T4 (Colab) | 0.432 |
| Fine-tune'suz STARK-ST baseline, RTX 3060, TF32 açık / kapalı | 0.476 / 0.479 |

Baseline bundan etkilenmiyor, online fine-tune ise etkileniyor: bu dizide iki koşu kare 1599'a kadar aynı; kare
1600'de tracker yanlış nesnenin üzerinde ve küçük sayısal farklara bağlı olarak skoru ya 0.19 (update yok) ya da
1.00 oluyor (yanlış nesneyle template update **ve** 15 fine-tune adımı; sonrasında tracker benzer nesnede kalıyor).
**Bu yüzden birbiriyle karşılaştırılan tüm deneyleri aynı GPU tipinde koşun ve GPU'yu raporlayın.** Seed'ler tek
başına bu değişkenliği yakalamaz.

## 11. Eski koddan farklar ve düzeltilen hatalar

Bu repo, eski çalışma dizinindeki (`Stark/`) kodun düzenlenmiş halidir. Eski dizin değiştirilmemiştir.

| # | Eski davranış | Bu repoda |
|---|---|---|
| 1 | `STARK_FT_MODE=all` tanımsızdı; tracker yalnızca `online`'ı tanıdığı için "online" diye adlandırılan deneylerde **online fine-tune hiç çalışmıyordu** (`run_all_finetune.py`, `run_bull_posonly_update.py`). | Modlar `none / init / online`; geçersiz değer hata verir. |
| 2 | Negatif bölge GT'den `2·max(w,h)` kaydırılıyordu; arama kırpımının yarı kenarı `2.5·sqrt(w·h)` olduğundan en-boy oranı ≈ 1.56'dan küçük hedeflerde **negatif kırpım hedefi içeriyordu**; kenara kırpma (clamp) bunu kötüleştiriyordu. | Kaydırma kırpım boyutuna göre hesaplanır; hedefin kırpımın tamamen dışında kaldığı 20.000 rastgele durumda test edilmiştir. |
| 3 | `lib/test/evaluation/tracker.py` test dizisinin tüm GT'sini tracker'a veriyordu (negatif için gelecekteki görünmez kareler seçiliyordu): **test etiketi sızıntısı**. | Kaldırıldı. Tracker yalnızca ilk kare kutusunu görür. |
| 4 | Seed sabit değildi; jitter her koşuda farklıydı. | `seed` parametresi; aynı seed ile sonuçlar birebir aynı (doğrulandı). |
| 5 | `coco_eval.py` hedefsiz kareleri dışarıda bırakıyor ve skor eşiğini mAP'den önce uyguluyordu. | Standart COCO kullanımı + eşikte P/R/F1; eski hesap `legacy_*` olarak korunur. |
| 6 | `Preprocessor` görüntüyü `tolist()` ile tensöre çeviriyordu (çok yavaş). | Orijinal STARK hali (`torch.tensor(ndarray)`); `vot1`'de sorunsuz çalışıyor. |
| 7 | `vot evaluate` VOT-LT2020 stack'indeki `redetection` deneyini de koşuyordu (~2× süre). | Varsayılan olarak yalnızca `longterm`; `run_redetection=True` ile açılabilir. |
| 8 | Yollar geliştirme makinesine özel mutlak yollar olarak sabitti; Vast.ai kurulum betiği dizileri yanlış workspace'e taşıyordu. | Sabit yol yok; her deney kendi VOT workspace'ini oluşturur. |
| 9 | Model her başlatmada ImageNet ResNet ağırlıklarını indiriyordu (sonra checkpoint ile eziliyordu). | İndirme atlanır. |
| 10 | Her ayar kombinasyonu için ayrı bir `lib/test/vot20/stark_st101_ft_*.py` dosyası (~40 adet) ve onlarca `run_*.py` betiği vardı. | Tek giriş noktası: `ExperimentConfig` + `vot_entry.py`. |

**Doğrulamalar:** `ft_mode="none"` ile tracker çıktısı eski `STARK_ST` ile 150 karede bit düzeyinde aynıdır
(aynı checkpoint ile). STARK-S de eski kodla aynı sonucu verir.

Orijinal STARK **eğitim** kodundan farklar [§7.7](#77-orijinal-stark-eğitim-kodundan-farklar)'de listelenmiştir.
