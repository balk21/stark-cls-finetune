# stark-cls-finetune — STARK-ST için inference sırasında sınıflandırma başlığı fine-tune'u

[English](README.md) | **Türkçe**

Bu depo, [STARK](https://github.com/researchmm/Stark) (ICCV 2021) tracker'ının **STARK-ST** sürümüne
test (inference) sırasında, her videoya özel olarak **sınıflandırma başlığını (cls_head) fine-tune etme**
yöntemini ekler ve bunu **VOT-LT2020** üzerinde, **detection tarzı (COCO)** metriklerle değerlendirir.

Tüm iş akışı Jupyter notebook'ları üzerinden yürür. Kod hiçbir sabit dosya yolu içermez; aynı repo
Vast.ai sunucusunda da kişisel bilgisayarda da çalışır.

---

## İçindekiler
1. [Hızlı başlangıç (Vast.ai / Google Colab)](#1-hızlı-başlangıç)
2. [Kendi bilgisayarınızda](#2-kendi-bilgisayarınızda)
3. [Klasör yapısı](#3-klasör-yapısı)
4. [Yöntem](#4-yöntem)
5. [Parametreler](#5-parametreler)
6. [Çıktılar ve dosya formatları](#6-çıktılar-ve-dosya-formatları)
7. [Metrikler](#7-metrikler)
8. [Komut satırı (CLI)](#8-komut-satırı-cli)
9. [Sık karşılaşılan sorunlar](#9-sık-karşılaşılan-sorunlar)
10. [Bilinen sınırlamalar ve açık konular](#10-bilinen-sınırlamalar-ve-açık-konular)
11. [Eski koddan farklar ve düzeltilen hatalar](#11-eski-koddan-farklar-ve-düzeltilen-hatalar)
12. [Lisans ve atıf](#12-lisans-ve-atıf)

---

## 1. Hızlı başlangıç

### 1.1 Vast.ai

**Gereksinimler**

| | |
|---|---|
| GPU | NVIDIA **RTX 3000 / 4000 serisi** (test edilen). RTX 5000 serisi (Blackwell) ortamdaki PyTorch 2.4.1 ile **çalışmaz**. |
| GPU belleği | ≥ 8 GB |
| Disk | ≥ 40 GB (veri seti ≈ 17 GB + conda ortamı + çıktılar) |
| Yazılım | `conda` içeren bir imaj (örn. Vast.ai'ın conda'lı PyTorch şablonları) ve Jupyter |

**Adımlar**

1. Vast.ai'da yukarıdaki özelliklerde bir makine kiralayın ve Jupyter'i açın.
2. Jupyter'de bir terminal açıp depoyu indirin:
   ```bash
   git clone https://github.com/balk21/stark-cls-finetune.git
   ```
3. `notebooks/00_setup.ipynb` dosyasını açın ve hücreleri sırayla çalıştırın:
   `vot1` ortamı kurulur, checkpoint ve veri seti indirilir, kısa bir duman testi yapılır.
4. `notebooks/01_run_experiment.ipynb` dosyasında parametreleri düzenleyip deneyi çalıştırın.
5. `notebooks/02_compare.ipynb` ile deneyleri karşılaştırın.

> **Notebook çekirdeği:** Herhangi bir Python 3 çekirdeği yeterlidir. Notebook'lar asıl işi arka planda
> `vot1` ortamının Python'u ile yaptırır (`notebooks/nbhelper.py`). Çekirdek değiştirmeniz gerekmez.

### 1.2 Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/00_setup_colab.ipynb)

1. Yukarıdaki rozetle `notebooks/00_setup_colab.ipynb` dosyasını Colab'de açın ve
   *Runtime → Change runtime type → GPU* (T4, L4 veya A100) seçin.
2. Hücreleri çalıştırın. Google Drive bağlanır ve Vast.ai'daki ile **aynı `vot1` ortamı** micromamba ile kurulur
   (aynı paket sürümleri). Not: sonuçlar yalnızca aynı GPU tipinde birebir aynıdır; bkz.
   [GPU'lar arası sonuçlar](#gpular-arası-sonuçlar).
3. Ardından `01_run_experiment.ipynb` [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/01_run_experiment.ipynb) ve `02_compare.ipynb`
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/02_compare.ipynb) notebook'larını Colab'de açın.

Colab her oturumda yeni bir makine verir. Bu yüzden hazırlaması uzun süren her şey Google Drive'da
`MyDrive/stark-cls-finetune/` altında saklanır ve her notebook'un ilk hücresi tarafından geri yüklenir
(`notebooks/colab_setup.py`):

| | İlk oturum (bir kez) | Sonraki oturumlar |
|---|---|---|
| `vot1` ortamı | kurulur (~5–10 dk), Drive'a arşivlenir (`cache/vot1_env_<hash>.tar`, 7.8 GB) | geri yüklenir (~1–3 dk) |
| Checkpoint | Drive'a indirilir (`checkpoints/`) | kopyalanır (saniyeler) |
| VOT-LT2020 veri seti | indirilir (20–60 dk), Drive'a arşivlenir (`cache/votlt2020_sequences.tar`, 17 GB) | geri yüklenir (birkaç dk) |
| Çıktılar | Drive'a yazılır (`outputs/`) | korunur; yarıda kalan koşular devam ettirilebilir |

**Gereken Google Drive alanı:** ≈ 26 GB + deney çıktıları. Kendi checkpoint'lerinizi
`MyDrive/stark-cls-finetune/checkpoints/<stark_st2|stark_s>/<model_config>/` altına yükleyebilirsiniz.

## 2. Kendi bilgisayarınızda

Aynı notebook'lar kullanılır. Checkpoint veya veri seti zaten başka bir yerdeyse indirmek yerine
`configs/paths.local.example.yaml` dosyasını `configs/paths.local.yaml` adıyla kopyalayıp yolları yazın:

```yaml
checkpoints: /home/kullanici/stark/checkpoints/train   # <checkpoints>/stark_st2/baseline_R101/STARKST_ep0050.pth.tar
dataset: /home/kullanici/vot/votlt2020/sequences        # <dataset>/<dizi>/{color/, groundtruth.txt, sequence}
outputs: /home/kullanici/stark_outputs
```

Bu dosya git'e girmez. Aynı ayarlar `STARK_CLEAN_CHECKPOINTS`, `STARK_CLEAN_DATASET` ve `STARK_CLEAN_OUTPUTS`
ortam değişkenleriyle de verilebilir (öncelik: ortam değişkeni > `paths.local.yaml` > `paths.yaml`).

Veri seti klasörüne **hiçbir dosya yazılmaz**. VOT-toolkit'in ihtiyaç duyduğu `list.txt`, her deneyin kendi
`vot_workspace/` klasöründe, dizilerin mutlak yollarıyla oluşturulur.

## 3. Klasör yapısı

```
stark-cls-finetune/
├── notebooks/
│   ├── 00_setup.ipynb            Conda'lı makinede kurulum (ortam, checkpoint, veri seti, duman testi)
│   ├── 00_setup_colab.ipynb      Google Colab'de kurulum (ortam / veri seti Google Drive'da önbelleklenir)
│   ├── 01_run_experiment.ipynb   Parametreler → çalıştırma → sonuçlar
│   ├── 02_compare.ipynb          Deney karşılaştırma
│   ├── nbhelper.py               Notebook yardımcıları (yalnızca standart kütüphane)
│   └── colab_setup.py            Google Colab oturum hazırlığı (yalnızca standart kütüphane)
├── stark_ft/                     Deney altyapısı
│   ├── config.py                 ExperimentConfig: TÜM parametreler ve doğrulama
│   ├── paths.py                  Yol çözümleme
│   ├── runner.py                 VOT workspace hazırlığı + `vot evaluate` + sonuç toplama
│   ├── vot_entry.py              VOT-toolkit'in her dizi için başlattığı tracker süreci
│   ├── tracker_factory.py        Parametrelerden tracker nesnesi
│   ├── evaluation.py             COCO AP, P/R/F1, legacy COCO
│   ├── analysis.py               Metrik tabloları + grafik üretimi
│   ├── plots.py                  Grafikler
│   ├── compare.py                Deney karşılaştırma
│   ├── setup_utils.py            Ortam kontrolü, indirme, duman testi
│   └── __main__.py               CLI (python -m stark_ft ...)
├── lib/                          STARK çekirdeği (yalnızca inference için gerekenler)
│   ├── models/stark/             Ağ mimarisi (orijinal STARK)
│   ├── config/                   Model config varsayılanları (orijinal STARK)
│   ├── test/tracker/
│   │   ├── stark_st.py           STARK-ST tracker (orijinal + update kayıtları)
│   │   ├── stark_st_ft.py        ★ Fine-tune'lu tracker
│   │   ├── ft_sampling.py        ★ Pozitif jitter ve negatif bölge üretimi
│   │   └── stark_s.py            STARK-S tracker
│   └── utils/
├── model_configs/                STARK model YAML'ları (stark_st2/, stark_s/)
├── configs/
│   ├── paths.yaml                Varsayılan yollar
│   ├── paths.local.example.yaml  Kişisel yol ayarı şablonu
│   └── experiments/example.yaml  CLI için örnek deney dosyası
├── environment/vot1_environment.yml
├── tests/test_sampling.py
├── checkpoints/   data/   outputs/   (git'e girmez)
```

## 4. Yöntem

### 4.1 STARK-ST'de güven skoru

STARK-ST, her karede hedefin kutusunun yanında bir **güven skoru** üretir. Bu skor, transformer decoder
çıktısı `hs` (256 boyutlu) üzerinde çalışan 3 katmanlı bir MLP'nin (**cls_head**) sigmoid çıktısıdır.
Skor iki yerde kullanılır:

1. **Template update kararı:** Her `update_interval` karede bir, skor `update_conf_thr` (0.5) değerinden
   büyükse dinamik template o karedeki tahminle güncellenir.
2. **Long-term değerlendirme:** Hedefin görünmediği karelerde tracker'ın düşük skor vermesi beklenir.
   Bu depodaki metriklerde skor, "hedefi buldu / bulamadı" kararının temelidir.

### 4.2 Fine-tune

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

## 5. Parametreler

Tüm parametreler `stark_ft/config.py` içindeki `ExperimentConfig` sınıfında tanımlıdır. Notebook'ta
`PARAMS = dict(...)` olarak, CLI'da YAML dosyası veya `--set anahtar=değer` olarak verilir.
Verilmeyen parametre varsayılan değerini alır.

### Kimlik ve model

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `name` | `None` | Çıktı klasörünün adı (`outputs/<name>/`). `None` ise parametrelerden üretilir, örn. `st101_online_pos_lr0.0001_i15_o1_int100_s0`. Boşluk ve `/\:*?"<>\|` içeremez. |
| `model` | `"stark_st"` | `"stark_st"` (güven skorlu, fine-tune edilebilir) veya `"stark_s"` (skor yok; sabit 1.0 raporlanır, `ft_mode="none"` olmalı). |
| `model_config` | `"baseline_R101"` | `model_configs/<stark_st2\|stark_s>/` altındaki YAML. `stark_st`: `baseline_R101` (ST101), `baseline` (ST50), `baseline_R101_got10k_only`, `baseline_got10k_only`. `stark_s`: `baseline`, `baseline_got10k_only`. |
| `checkpoint` | `None` | `None` ise resmi dosya adı kullanılır (`STARKST_ep0050.pth.tar` / `STARKS_ep0500.pth.tar`). Yalnızca dosya adı verilirse `<checkpoints>/<stark_st2\|stark_s>/<model_config>/` içinde aranır, örn. `"STARKSTcoco_ep0050.pth.tar"`. `/` içeren değer yol olarak kullanılır (göreli ise depo köküne göre). |

### Veri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `sequences` | `"all"` | `"all"` veya dizi adları listesi, örn. `["bull", "ballet"]`. Dizi adı dataset klasöründeki klasör adıdır. |

### Template update (yalnızca `stark_st`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `update_interval` | `100` | Kaç karede bir template update denenir. `99999` = hiç update yok. (Model YAML'larındaki `TEST.UPDATE_INTERVALS` bu depoda **kullanılmaz**; değer her zaman bu parametreden gelir.) |
| `update_conf_thr` | `0.5` | Update için skorun **büyük** olması gereken eşik (orijinal STARK: 0.5). |
| `max_template_updates` | `-1` | Bir dizide yapılabilecek en fazla template update sayısı. `-1` = sınırsız. |

### Fine-tune (yalnızca `stark_st`)

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `ft_mode` | `"online"` | `"none"`, `"init"`, `"online"` (bkz. §4.2). |
| `ft_samples` | `"pos"` | `"pos"` veya `"posneg"` (bkz. §4.2). |
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
| `eval_score_thr` | `0.35` | **Sabit** eşik: "tracker hedefi buldu" kararı için skor eşiği. Yalnızca `precision / recall / F1` kolonlarını etkiler; AP'yi ve F-maksimum eşik sonucunu etkilemez. |
| `eval_iou_thr` | `0.5` | Bulunan kutunun doğru sayılması için en düşük IoU (hem sabit eşikte hem F-maksimum aramasında). |
| `eval_thr_resolution` | `100` | F-maksimum eşik aramasındaki aday eşik sayısı (vot-toolkit varsayılanı: 100). |

Bu üç değer tracking'i tekrar çalıştırmadan değiştirilebilir: `nb.analyze(OUT, score_thr=0.5, thr_resolution=200)`.
Yalnızca bunları değiştirmek, deneyi "farklı parametreli" yapmaz (devam etme kuralını etkilemez).

### VOT-toolkit

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `run_redetection` | `False` | VOT-LT2020 stack'indeki ikinci deney (`redetection`) de koşulsun mu? Süreyi yaklaşık iki katına çıkarır; bu depodaki metrikler yalnızca `longterm` sonuçlarını kullanır. |
| `tracker_timeout` | `300` | Tracker'ın tek bir yanıtı için zaman aşımı (saniye). Model yükleme dahil. |

### Deney adı, devam etme ve üzerine yazma

- Aynı `name` + **aynı** parametrelerle tekrar çalıştırmak **kaldığı yerden devam eder**: VOT-toolkit
  tamamlanmış dizileri atlar. Kesilen bir koşuyu sürdürmek için hücreyi yeniden çalıştırmanız yeterlidir.
  (`eval_*` parametreleri bu karşılaştırmaya dahil değildir.)
- Aynı `name` + **farklı** parametreler hata verir ve eski sonuçlar korunur. Üzerine yazmak için
  `nb.run(PARAMS, overwrite=True)` (CLI: `--overwrite`).

## 6. Çıktılar ve dosya formatları

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
| `metrics/summary.txt` | Okunabilir özet (ilk satırlar: F-maksimum eşik ve sabit eşik sonuçları) + dizi bazında tablo |
| `metrics/metrics.xlsx` | `summary` (kolonlar: `mean over sequences`, `pooled`), `optimal_threshold`, `per_sequence`, `f_curve`, `parameters` sayfaları |
| `metrics/f_curve.csv` | Her aday eşikte dizi ortalaması `precision`, `recall`, `F` |
| `metrics/f_curve.png` | Solda P / R / F – eşik (mor: seçilen eşik, siyah: sabit eşik), sağda precision–recall eğrisi |
| `metrics/metrics.json` | Aynı veriler, makine tarafından okunabilir |
| `vot_workspace/` | Deneye özel VOT workspace'i (`config.yaml`, `trackers.ini`, ham `results/`) |

**Kare numaralandırması:** Her yerde `frame = 0` ilk (init) karedir. VOT sonuç dosyalarında bu, satır numarasının
1 eksiğidir. İlk kare değerlendirmeye girmez (orada GT tracker'a verilir).

## 7. Metrikler

Değerlendirme **detection** tarzındadır: her kare bir "görüntü", hedef tek nesnedir. Hesap `pycocotools`
kütüphanesiyle yapılır (`stark_ft/evaluation.py`).

### 7.1 COCO AP / AP50 / AP75 (standart kullanım)

- İlk kare hariç **tüm** kareler değerlendirmeye girer.
- Hedefin görünmediği (GT = NaN) kareler, annotation'ı olmayan görüntü olarak eklenir. Bu karelerde verilen
  her tahmin **false positive** sayılır.
- **Skor eşiği uygulanmaz**: tüm tahminler skorlarıyla verilir. AP, skora göre sıralamanın kalitesini
  zaten tüm eşikler üzerinden ölçer. Eşik koymak AP'yi hiçbir zaman artıramaz.
- Koordinatlar float olarak kullanılır.

### 7.2 Eşikte "buldu / bulmadı" metrikleri

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

### 7.3 F-maksimum eşik (VOT-LT yöntemi)

`eval_score_thr` sabit bir eşiktir. Asıl karşılaştırma ölçütü, VOT-LT protokolündeki gibi **F'yi maksimize eden
eşikte** hesaplanan F'dir. Yöntem vot-toolkit 0.5.3'ün `vot/analysis/tpr.py` dosyasıyla aynıdır; tek fark P ve R'nin
nasıl sayıldığıdır:

1. **Aday eşikler:** Deneydeki tüm dizilerin skorları (init karesi hariç) havuzlanıp büyükten küçüğe sıralanır.
   Bu sıradan eşit aralıklı `eval_thr_resolution − 2` değer seçilir, başa +∞ ve sona −∞ eklenir
   (vot-toolkit'in `determine_thresholds` fonksiyonunun birebir kopyası).
2. **Her eşikte, her dizi için P ve R:** vot-toolkit IoU ağırlıklı P/R kullanır. Burada §7.2'deki
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

### 7.4 Diğer

- `mean_iou_visible`: hedefin görünür olduğu karelerdeki ortalama IoU (eşik yok).
- `legacy_AP / legacy_AP50 / legacy_AP75`: eski `testler/detailed_analysis/coco_eval.py` ile **birebir aynı**
  hesap (hedefsiz kareler dışarıda, skoru eşiğin altındaki tahminler atılır, koordinatlar tam sayıya yuvarlanır).
  Yalnızca eski sonuçlarla karşılaştırma içindir; hedefsiz karelerdeki yanlış tespitleri cezalandırmaz.
  5 dizide eski betikle aynı sonucu verdiği doğrulanmıştır.
- **Dizi ortalaması:** her dizinin metriğinin ortalaması (eski `MEAN` satırıyla aynı mantık). İstisna: `F1` ve
  `F_opt`, VOT tanımıyla ortalama P ve R'den hesaplanır.
- **Havuzlanmış (pooled):** tüm dizilerin kareleri tek veri seti gibi; uzun diziler daha çok ağırlık alır.

## 8. Komut satırı (CLI)

Notebook'lar bu komutları çağırır; doğrudan da kullanılabilir (`vot1` ortamında, depo kökünde):

```bash
python -m stark_ft check                                   # ortam + yollar
python -m stark_ft show --config configs/experiments/example.yaml
python -m stark_ft smoke --set ft_mode=online --frames 50  # VOT'suz hızlı test
python -m stark_ft run --config configs/experiments/example.yaml --set 'sequences=[bull]' --set ft_samples=posneg
python -m stark_ft analyze outputs/<deney> --score-thr 0.5 --thr-resolution 100
python -m stark_ft list
python -m stark_ft compare <deney1> <deney2> --out comparison.xlsx --plot comparison.png
python -m stark_ft download-checkpoints --model stark_st --model-config baseline_R101 baseline
python -m stark_ft download-dataset
python -m tests.test_sampling                              # negatif örnek geometrisi testleri
```

`--set` değerleri YAML olarak yorumlanır: `1e-4` → sayı, `true` → bool, `[a, b]` → liste.

## 9. Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| `Conda environment 'vot1' not found` | `00_setup.ipynb` 1. adım. Ortam başka yerdeyse `VOT1_PYTHON=/yol/envs/vot1/bin/python` ortam değişkenini tanımlayın. |
| `Checkpoint not found` | `00_setup.ipynb` 3. adım veya dosyayı hata mesajındaki konuma koyun. |
| Google Drive "kota aşıldı" | Hata mesajındaki linkten tarayıcıyla indirip belirtilen klasöre koyun. |
| `No VOT sequences found in the dataset folder` | `00_setup.ipynb` 4. adım veya `configs/paths.local.yaml` içinde `dataset`. |
| `Completed sequences: 47/50` gibi eksik | Hata veren dizi atlanır, diğerleri koşmaya devam eder. Tracker'ın hata çıktısı `outputs/<deney>/vot_workspace/logs/` içindedir. Aynı deneyi tekrar çalıştırmak tamamlananları atlar, eksikleri yeniden dener. |
| `no kernel image is available` / sm_120 uyarısı | GPU, ortamdaki PyTorch tarafından desteklenmiyor (RTX 5000 serisi). RTX 3000/4000 serisi kullanın. |
| "A newer version of the VOT toolkit is available" | Görmezden gelin; depo vot-toolkit **0.5.3** ile test edilmiştir. Güncellemeyin. |
| Colab: `WARNING: no GPU in this session` | *Runtime → Change runtime type → GPU* seçip ilk hücreyi tekrar çalıştırın. |
| Colab: Drive'a arşivleme başarısız | Google Drive'da yeterli yer yok (≈ 26 GB gerekir). Yer açıp ilk hücreyi tekrar çalıştırın; tamamlanan adımlar tekrarlanmaz. |

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

### GPU'lar arası sonuçlar

Aynı ortam, yalnızca **aynı GPU tipinde** bit düzeyinde aynı sonucu verir. Farklı GPU'lar farklı sayısal çekirdekler
kullanır; ayrıca Ampere ve sonrası GPU'larda (RTX 3000/4000, A100, ...) PyTorch konvolüsyonları varsayılan olarak TF32
ile hesaplarken T4 (Colab) gibi eski GPU'lar tam FP32 hesaplar. Tracking her kareyi bir sonrakine aktardığı için çok
küçük sayısal farklar büyüyebilir.

Örnek (`bull`, online, pos, 15+15 adım, lr 1e-5, interval 100; legacy AP):

| Koşu | legacy AP |
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

Bu depo, eski çalışma dizinindeki (`Stark/`) kodun düzenlenmiş halidir. Eski dizin değiştirilmemiştir.

| # | Eski davranış | Bu depoda |
|---|---|---|
| 1 | `STARK_FT_MODE=all` tanımsızdı; tracker yalnızca `online`'ı tanıdığı için "online" diye adlandırılan deneylerde **online fine-tune hiç çalışmıyordu** (`run_all_finetune.py`, `run_bull_posonly_update.py`). | Modlar `none / init / online`; geçersiz değer hata verir. |
| 2 | Negatif bölge GT'den `2·max(w,h)` kaydırılıyordu; arama kırpımının yarı kenarı `2.5·sqrt(w·h)` olduğundan en-boy oranı ≈ 1.56'dan küçük hedeflerde **negatif kırpım hedefi içeriyordu**; kenara kırpma (clamp) bunu kötüleştiriyordu. | Kaydırma kırpım boyutuna göre hesaplanır; hedefin kırpımın tamamen dışında kaldığı 20.000 rastgele durumda test edilmiştir. |
| 3 | `lib/test/evaluation/tracker.py` test dizisinin tüm GT'sini tracker'a veriyordu (negatif için gelecekteki görünmez kareler seçiliyordu): **test etiketi sızıntısı**. | Kaldırıldı. Tracker yalnızca ilk kare kutusunu görür. |
| 4 | Yerel `lib/test/parameter/stark_st.py`, `STARKSTcoco_ep0050.pth.tar` varsa **sessizce onu** yüklüyordu; "baseline" sonuçları COCO checkpoint'i ile alınmış olabilir. | Checkpoint yalnızca `checkpoint` parametresiyle seçilir ve `experiment.json`'a yazılır. |
| 5 | Seed sabit değildi; jitter her koşuda farklıydı. | `seed` parametresi; aynı seed ile sonuçlar birebir aynı (doğrulandı). |
| 6 | `coco_eval.py` hedefsiz kareleri dışarıda bırakıyor ve skor eşiğini AP'den önce uyguluyordu. | Standart COCO kullanımı + eşikte P/R/F1; eski hesap `legacy_*` olarak korunur. |
| 7 | `Preprocessor` görüntüyü `tolist()` ile tensöre çeviriyordu (çok yavaş). | Orijinal STARK hali (`torch.tensor(ndarray)`); `vot1`'de sorunsuz çalışıyor. |
| 8 | `vot evaluate` VOT-LT2020 stack'indeki `redetection` deneyini de koşuyordu (~2× süre). | Varsayılan olarak yalnızca `longterm`; `run_redetection=True` ile açılabilir. |
| 9 | Yollar geliştirme makinesine özel mutlak yollar olarak sabitti; Vast.ai kurulum betiği dizileri yanlış workspace'e taşıyordu. | Sabit yol yok; her deney kendi VOT workspace'ini oluşturur. |
| 10 | Model her başlatmada ImageNet ResNet ağırlıklarını indiriyordu (sonra checkpoint ile eziliyordu). | İndirme atlanır. |
| 11 | Her ayar kombinasyonu için ayrı bir `lib/test/vot20/stark_st101_ft_*.py` dosyası (~40 adet) ve onlarca `run_*.py` betiği vardı. | Tek giriş noktası: `ExperimentConfig` + `vot_entry.py`. |

**Doğrulamalar:** `ft_mode="none"` ile tracker çıktısı eski `STARK_ST` ile 150 karede bit düzeyinde aynıdır
(aynı checkpoint ile). STARK-S de eski kodla aynı sonucu verir.

## 12. Lisans ve atıf

STARK kodu MIT lisansı ile dağıtılır (`LICENSE`). STARK'ı kullanıyorsanız orijinal makaleye atıf yapın:

```bibtex
@inproceedings{yan2021learning,
  title={Learning Spatio-Temporal Transformer for Visual Tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={ICCV},
  year={2021}
}
```

### Önceki çalışma

Bu repo, önceki çalışmamızın devamıdır:

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
