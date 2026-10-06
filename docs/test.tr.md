# Test: VOT-LT2020'de test sırasında fine-tune

STARK-ST, VOT-LT2020 üzerinde vot-toolkit (0.5.3) ile, sınıflandırma başlığı test sırasında fine-tune edilerek ya da
edilmeden çalıştırılır ve COCO tarzı detection metrikleriyle değerlendirilir.

[English](test.md) | **Türkçe**

- [Yöntem](#yöntem)
- [Çalıştırma](#çalıştırma)
- [Sık kullanılan ayarlar](#sık-kullanılan-ayarlar)
- [Ağırlıklar](#ağırlıklar)
- [VOT-LT2020 dizileri](#vot-lt2020-dizileri)
- [Parametreler](#parametreler)
- [Çıktılar](#çıktılar)
- [Metrikler](#metrikler)
- [Sınırlamalar](#sınırlamalar)
- [GPU'lar arası sonuçlar](#gpular-arası-sonuçlar)
- [Eski araştırma kodundan farklar](#eski-araştırma-kodundan-farklar)
- [Sık karşılaşılan sorunlar](#sık-karşılaşılan-sorunlar)

## Yöntem

STARK-ST her karede bir **güven skoru** üretir: transformer decoder çıktısına uygulanan 3 katmanlı bir MLP'nin
(**cls_head**) sigmoid'i. Skor **template update** kararını verir (her `update_interval` karede bir, skor
`update_conf_thr`'den büyükse) ve bu değerlendirmede tracker'ın hedefi **bulduğunu** söyleyip söylemediğini belirler.

Base eğitimde cls_head genel bir "hedef var / yok" kararı öğrenir. Burada her videonun başında base ağırlıklarına
döndürülür ve o videonun hedefi için kısa süre eğitilir. Backbone, transformer ve kutu başlığı dondurulur.

```
kare + hedef kutusu
  ├─ pozitif kutu = kutunun kendisi (ft_pos_jitter ise aşama-2 eğitim jitter'ı ile)   -> etiket 1
  └─ negatif kutu = aynı karede hedefi İÇERMEYEN bir bölge                            -> etiket 0  (yalnızca posneg)
        │  her kutu: arama kırpması -> backbone -> template'lerle -> transformer -> cls_head
        ▼
  BCEWithLogitsLoss -> AdamW (ft_lr, ft_weight_decay) -> gradyan kırpma (ft_grad_clip_norm)
  (1 epoch = 1 optimizasyon adımı; her fine-tune oturumunda yeni bir optimizer)
```

| `ft_mode` | Ne zaman | Kutu |
|---|---|---|
| `none` | hiçbir zaman (sade STARK-ST) | — |
| `init` | ilk kare, `ft_epochs_init` adım | ilk karenin ground truth'u |
| `online` | ilk kare + her template update'te `ft_epochs_online` adım | ilk kare: ground truth; sonra: tracker'ın kendi tahmini |

| `ft_samples` | Örnekler |
|---|---|
| `pos` | yalnızca pozitif |
| `posneg` | pozitif + aynı karenin bir negatif bölgesi; hedefin arama kırpmasının tamamen dışında kalacak şekilde kaydırılır (8 yön denenir; görüntünün en çok içinde kalan kullanılır). Daha iyi bir negatif örnekleme bulunana kadar geçicidir. |

## Çalıştırma

```bash
python -m stark_ft weights                                             # resmi ağırlıklar ve burada eğitilen koşular
python -m stark_ft smoke --frames 50                                   # isteğe bağlı hızlı kontrol (ballet'i indirir)

python -m stark_ft test --set 'sequences=[bull]' --set ft_mode=none    # sade STARK-ST, resmi ağırlıklar
python -m stark_ft test --config configs/test_example.yaml --set ft_samples=posneg --set 'sequences=[bull]'
python -m stark_ft test --set 'sequences=[bull]' --set weights=<koşu adı>   # stark_ft train ile eğitilmiş bir model
python -m stark_ft analyze outputs/<deney> --score-thr 0.5             # tracking olmadan metrikleri yeniden hesaplar
python -m stark_ft compare <deney 1> <deney 2> --out comparison.xlsx --plot comparison.png
```

Notebook'lar: `notebooks/test.ipynb`, `notebooks/compare.ipynb`. 50 dizinin tamamı (≈ 215 bin kare) RTX 3060 laptop
GPU'da ≈ 2 saat sürer.

- `show` (notebook: `nb.describe`) çıktı klasörünü, ağırlıkları ve hangi dizilerin indirileceğini gösterir.
- Aynı `name` aynı parametrelerle tekrar çalıştırılırsa **devam eder**: tamamlanan diziler atlanır (`vot evaluate` bu
  yüzden `-f` olmadan çalışır; her deneyin kendi VOT workspace'i vardır). `vot analysis` kullanılmaz; metrikler her
  zaman ham sonuçlardan hesaplanır.
- Aynı `name` farklı parametrelerle ya da tracking kodunun farklı bir sürümüyle (`lib/`, `model_configs/`, tracker
  giriş noktası) hata verir; başka bir `name` ya da `--overwrite` kullanın. `eval_*` parametreleri serbestçe
  değiştirilebilir (`analyze`).

## Sık kullanılan ayarlar

| İstediğiniz | Ayar |
|---|---|
| Sade STARK-ST (baseline) | `ft_mode="none"` |
| Yalnızca ilk karede fine-tune | `ft_mode="init"` |
| İlk karede ve her template update'te fine-tune | `ft_mode="online"` |
| Yalnızca pozitif / pozitif + negatif örnekler | `ft_samples="pos"` / `"posneg"` |
| Fine-tune'un gücü | `ft_lr`, `ft_epochs_init` (1. karedeki adım), `ft_epochs_online` (update başına adım) |
| Diziler | `sequences=["bull", "ballet"]` veya `"all"` |
| ST101 yerine STARK-ST50 | `model_config="baseline"` |
| Bu repoyla eğitilmiş bir model | `weights="<aşama-2 koşu adı>"` (aynı `model_config`) |

## Ağırlıklar

`weights` bir testin ağ ağırlıklarını seçer; deney adı ve `experiment.json` bunu kaydeder.

| `weights` | Ağırlıklar |
|---|---|
| `"official"` (varsayılan) | `model_config` için STARK'ın yayımladığı ağırlıklar (LaSOT + GOT-10k + COCO + TrackingNet ile eğitildi; `*_got10k_only`: GOT-10k). İlk kullanıldığında `<checkpoints>` klasörüne otomatik indirilir. |
| `"<koşu adı>"` | Burada eğitilmiş, tamamlanmış bir **aşama-2** koşusu, örn. `"st101_coco_stage2_s42"` ([train.tr.md](train.tr.md#bizim-koşularımız-ve-resmi-ağırlıklar)). Aynı `model_config` ile kullanılmalıdır; aşama-1 ve tamamlanmamış koşular reddedilir. |
| dosya adı / yol | Başka herhangi bir checkpoint: dosya adı `<checkpoints>/stark_st2/<model_config>/` içinde aranır, `/` içeren değer yoldur. |

`python -m stark_ft weights` (notebook: `nb.list_weights()`) bunları kökenlerine göre listeler: resmi ağırlıklar,
burada ImageNet'ten eğitilen koşular ve burada resmi ağırlıkların üzerine eğitilen koşular. `show` / `nb.describe` bir
testin hangi ağırlıkları kullanacağını gösterir. Deney adları, eğitilmiş ağırlıklarda koşu adıyla
(`st101_coco_stage2_s42_online_...`), resmi ağırlıklarda modelle (`st101_online_...`) başlar; böylece farklı
ağırlıkların sonuçları asla karışmaz.

## VOT-LT2020 dizileri

Önceden hiçbir şey indirmek gerekmez. Bir test başladığında, kullandığı dizilerden `dataset` klasöründe henüz
olmayanlar resmi VOT sunucusundan tek tek indirilir (VOT-LT2020 yığınının kullandığı VOT-LT2019 dizileri), resmi
SHA-1 değerleriyle kontrol edilir ve vot-toolkit'in yazdığı şekilde yazılır (`stark_ft/test/vot_data.py`).
`sequences=["bull"]` yalnızca bull'u indirir (58 MB); `"all"` 50 dizinin tamamını (17.6 GB, 20–60 dk). Önceden indirmek
için: `python -m stark_ft download-dataset [--sequences bull ballet]` (`--sequences` olmadan: 50 dizinin tamamı).

`dataset_cache` ayarlıysa (`configs/paths.local.yaml`), indirilen her dizinin bir kopyası (`<dizi>.tar`) orada saklanır
ve bir sonraki sefer sunucu yerine o kullanılır (Colab'de: Drive klasörü `cache/votlt2019_sequences/`). Dataset
klasöründe zaten bulunan diziler yalnızca okunur; vot-toolkit'in ihtiyaç duyduğu `list.txt` her deneyin kendi
`vot_workspace/` klasörüne yazılır.

## Parametreler

`stark_ft/test/config.py` içindeki `ExperimentConfig`. Notebook: `PARAMS = dict(...)`; CLI: `--config dosya.yaml`
ve/veya `--set anahtar=değer`.

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `name` | `None` | Çıktı klasörü `<outputs>/<name>/`. `None`: üretilir, örn. `st101_online_pos_lr0.0001_i15_o1_int100_s0`. |
| `model` | `"stark_st"` | `"stark_st"` veya `"stark_s"` (skor yok; 1.0 raporlar; `ft_mode="none"` gerekir). |
| `model_config` | `"baseline_R101"` | `model_configs/stark_st2/` (`baseline_R101`, `baseline`, `*_got10k_only`) veya `model_configs/stark_s/` altındaki YAML. |
| `weights` | `"official"` | `"official"`, burada eğitilmiş bir aşama-2 koşusu ya da bir checkpoint dosyası / yolu (bkz. [Ağırlıklar](#ağırlıklar)). (Eski adı: `checkpoint`.) |
| `sequences` | `"all"` | `"all"` (50 dizi) veya bir liste, örn. `["bull", "ballet"]`; eksik olanlar indirilir. |
| `update_interval` | `100` | Her N karede bir template update denemesi; `99999` = hiç. (YAML'lardaki `TEST.UPDATE_INTERVALS` kullanılmaz.) |
| `update_conf_thr` | `0.5` | Yalnızca skor bundan büyükse update (STARK: 0.5). |
| `max_template_updates` | `-1` | Dizi başına; `-1` = sınırsız. |
| `ft_mode` | `"online"` | `"none"`, `"init"`, `"online"` (bkz. [Yöntem](#yöntem)). |
| `ft_samples` | `"pos"` | `"pos"` veya `"posneg"`. |
| `ft_lr` | `1e-4` | AdamW öğrenme oranı (aşama-2 eğitimi: 1e-4). |
| `ft_epochs_init` | `15` | İlk karedeki adım sayısı; `0` = yok. |
| `ft_epochs_online` | `1` | Online fine-tune başına adım. |
| `ft_weight_decay` | `1e-4` | Eğitimdeki gibi. |
| `ft_grad_clip_norm` | `0.1` | Eğitimdeki gibi. |
| `ft_pos_jitter` | `True` | Pozitif kutuya aşama-2 eğitimindeki jitter'ı uygula; `False`: kutunun kendisi. |
| `ft_center_jitter` | `4.5` | Merkez, `sqrt(w·h)·4.5` genişliğinde bir pencerede kaydırılır (aşama 2: 4.5). |
| `ft_scale_jitter` | `0.5` | Boyut × `exp(N(0,1)·0.5)` (aşama 2: 0.5). |
| `max_ft_updates` | `-1` | Dizi başına online fine-tune sayısı; `-1` = sınırsız. |
| `seed` | `0` | Aynı seed ve parametreler birebir aynı sonucu verir (aynı GPU tipinde). |
| `eval_score_thr` | `0.35` | P / R / F1 için sabit eşik (mAP'yi ve F-maksimum eşiği etkilemez). |
| `eval_iou_thr` | `0.5` | Doğru tespit için IoU (sabit eşik ve F-maksimum araması). |
| `eval_thr_resolution` | `100` | F-maksimum aramasındaki aday eşik sayısı (vot-toolkit: 100). |
| `run_redetection` | `False` | VOT-LT2020 `redetection` deneyini de çalıştır (~2× süre; metriklerde kullanılmaz). |
| `tracker_timeout` | `300` | Model yükleme dahil tracker yanıtı başına saniye. |

## Çıktılar

`<outputs>/<deney adı>/`:

| Dosya | İçerik |
|---|---|
| `experiment.json` | Parametreler (`weights` dahil), checkpoint dosyası, dataset, diziler, git commit, kod hash'i, tarih |
| `run.log`, `run_status.json` | vot-toolkit çıktısı; tamamlanan / eksik diziler (tracker hataları: `vot_workspace/logs/`) |
| `predictions/<dizi>/<dizi>_001.txt` | Kare başına kutu `x,y,w,h` (ilk satır `1`: init karesi, VOT formatı) |
| `predictions/<dizi>/<dizi>_001_confidence.value`, `_time.value` | Kare başına skor ve süre |
| `predictions/<dizi>/frames.csv` | `frame, x, y, w, h, conf, time, gt_visible, gt_x, gt_y, gt_w, gt_h, iou` |
| `tracker_logs/<dizi>/finetune_loss.txt` | `frame, session, epoch, loss, pos_prob, neg_prob, n_pos, n_neg, neg_coverage` (adım başına bir satır; olasılıklar güncellemeden önce) |
| `tracker_logs/<dizi>/events.txt` | `frame, event, conf_score` (`template_update`, `ft_update`) |
| `plots/<dizi>/iou_conf.png` | Kareye göre IoU ve skor (gri: hedef yok; kırmızı: template update; yeşil: fine-tune; siyah: sabit eşik; mor: F-maksimum eşik) |
| `plots/<dizi>/finetune_loss.png` | Adım başına loss ve pozitif / negatif olasılıklar |
| `metrics/summary.txt`, `metrics.xlsx`, `metrics.json` | Özet ve dizi başına metrikler |
| `metrics/f_curve.csv`, `f_curve.png` | Eşiğe göre P / R / F, PR eğrisi |
| `vot_workspace/` | Deneyin kendi VOT workspace'i (ham `results/`) |

Kare 0 ilk (init) karedir (VOT dosyalarında satır numarası − 1); değerlendirilmez.

## Metrikler

Detection tarzı: her kare (en fazla) tek nesneli bir görüntüdür; `pycocotools` ile hesaplanır
(`stark_ft/test/evaluation.py`). **Birincil metrikler mAP, AP50 ve AP75'tir.**

**mAP / AP50 / AP75** (standart COCO): mAP = COCO'nun AP'si (`stats[0]`, IoU 0.50:0.95), AP50 / AP75 tek IoU eşiğinde.
İlk kare hariç tüm kareler değerlendirilir; hedefin olmadığı kareler anotasyonsuz görüntülerdir, yani oradaki her
tahmin false positive'dir. Skor eşiği uygulanmaz (mAP zaten skor sıralamasını ölçer). "Diziler üzerinden ortalama" dizi
başına değerlerin ortalamasıdır; "pooled" tüm kareleri tek bir veri seti sayar.

**Sabit eşik** (`eval_score_thr`): skor ≥ eşik ise tracker "hedefi buldum" der.

| | buldum der (skor ≥ eşik) | demez |
|---|---|---|
| hedef görünür, IoU ≥ `eval_iou_thr` | TP | FN |
| hedef görünür, IoU < `eval_iou_thr` | FP + FN | FN |
| hedef yok | FP | TN |

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`, `absent_reject_rate` = TN / hedefin olmadığı kare
sayısı. COCOeval ile aynı eşleştirme kuralı (doğrulandı).

**F-maksimum eşik** (VOT-LT protokolü, vot-toolkit 0.5.3'ün `vot/analysis/tpr.py` dosyasındaki gibi): aday eşikler
birleştirilmiş skorlardan vot-toolkit'in `determine_thresholds` fonksiyonuyla birebir aynı şekilde alınır
(`eval_thr_resolution` değer); her eşikte P ve R dizi başına yukarıdaki sayımla hesaplanır (vot-toolkit IoU ağırlıklı
P/R kullanır), diziler üzerinden ortalanır ve F = 2PR/(P+R) ortalamalardan hesaplanır; en büyük F'yi veren eşik tüm
diziler için kullanılır (`precision_opt / recall_opt / F_opt`). Eşik, VOT-LT raporlamasındaki gibi ground truth ile
seçilir; tracker bu eşiği bilmez.

**Diğer:** `mean_iou_visible` (hedefin göründüğü karelerde ortalama IoU); `legacy_mAP / AP50 / AP75` eski
`coco_eval.py`'yi birebir yeniden üretir (hedefin olmadığı kareler hariç, eşiğin altındaki tahminler atılır, tam sayı
koordinatlar); yalnızca eski sonuçlarla karşılaştırma içindir.

## Sınırlamalar

- **`pos` skoru genel olarak şişirir.** Yalnızca 1 etiketiyle BCE'yi en kolay düşürmenin yolu her girdi için çıktıyı
  büyütmektir; AdamW loss çok küçükken bile her adımda ≈ `ft_lr` kadar ilerler. Skor hedefin olmadığı yerlerde de
  yükselir (`absent_reject_rate` düşer) ve template update daha sık olur.
- **`posneg` negatifleri geçicidir.** Aynı karedeki bir bölge benzer nesneler içerebilir (`finetune_loss.txt`'de yüksek
  `neg_prob`). Yalnızca `lib/test/tracker/ft_sampling.py` / `stark_st_ft.py` içindeki `posneg` dalı değişmelidir.
- **`online` tracker'ın kendi tahminiyle eğitir** ve update kararı da fine-tune edilen başlıktan gelir: kendini
  doğrulayan bir döngü riski vardır.
- **`ft_lr` / `ft_epochs_*` deneyseldir.** Bunları test dizilerinde seçmek sonuçları iyimser gösterir.
- vot-toolkit her dizide tracker sürecini yeniden başlatır (model yeniden yüklenir, ~3–4 sn).

## GPU'lar arası sonuçlar

Sonuçlar yalnızca **aynı GPU tipinde** bit düzeyinde aynıdır. GPU'lar farklı sayısal çekirdekler kullanır; ayrıca Ampere
ve sonrası GPU'lar (RTX 3000/4000, A100, L4) konvolüsyonları varsayılan olarak TF32 ile hesaplar, örn. T4 ise FP32
kullanır. Tracking her kareyi bir sonrakine aktardığı için küçük farklar büyüyebilir. Örnek (`bull`, online, pos,
15+15 adım, lr 1e-5, interval 100; legacy mAP):

| Koşu | legacy mAP |
|---|---|
| RTX 3060 (TF32, varsayılan), 8 seed | 0.521 ± 0.001 |
| RTX 3060, TF32 kapalı (`NVIDIA_TF32_OVERRIDE=0`) | 0.458 |
| Tesla T4 | 0.432 |
| Fine-tune'suz STARK-ST, RTX 3060, TF32 açık / kapalı | 0.476 / 0.479 |

Koşular 1599. kareye kadar aynıdır; 1600. karede tracker bir dikkat dağıtıcının üzerindedir ve skoru ya 0.19 (update
yok) ya da 1.00'dır (template update ve dikkat dağıtıcı üzerinde 15 fine-tune adımı). **Karşılaştırılan tüm deneyleri
aynı GPU tipinde çalıştırın ve GPU'yu raporlayın.**

## Eski araştırma kodundan farklar

| # | Eski davranış | Burada |
|---|---|---|
| 1 | `STARK_FT_MODE=all` tanımsızdı; bu yüzden "online" deneylerde online fine-tune hiç çalışmadı | `ft_mode` = `none / init / online`; geçersiz değer hata verir |
| 2 | Negatif bölge `2·max(w,h)` kaydırılıyordu; en-boy oranı ≈ 1.56'nın altındaki hedeflerde negatif kırpma hedefi içeriyordu | Kaydırma kırpma boyutundan hesaplanır; 20 000 rastgele durumda test edildi |
| 3 | Test dizisinin tüm ground truth'u tracker'a veriliyordu (test etiketi sızıntısı) | Kaldırıldı; tracker yalnızca ilk karenin kutusunu görür |
| 4 | Sabit seed yoktu | `seed`; birebir tekrarlanabilir |
| 5 | `coco_eval.py` hedefin olmadığı kareleri dışarıda bırakıyor ve mAP'den önce eşik uyguluyordu | Standart COCO + sabit eşik + F-maksimum eşik; eski hesap `legacy_*` olarak duruyor |
| 6 | Görüntüler `tolist()` ile tensöre çevriliyordu (yavaş) | Orijinal STARK sürümü |
| 7 | `redetection` deneyi her zaman çalışıyordu (~2× süre) | Varsayılan olarak yalnızca `longterm` |
| 8 | Geliştirme makinesine özel sabit yollar | Sabit yol yok; deney başına bir VOT workspace |
| 9 | ImageNet ağırlıkları her başlatmada indiriliyordu | Atlanır |
| 10 | Her parametre kombinasyonu için ayrı giriş dosyası (~40) | Tek giriş noktası (`ExperimentConfig` + `vot_entry.py`) |

`ft_mode="none"` ile çıktı orijinal `STARK_ST` ile bit düzeyinde aynıdır (150 kare, aynı checkpoint).

## Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| `No training run '...'` / `is a stage-1 run` / `was trained with model_config=...` | `python -m stark_ft weights` listesinden tamamlanmış bir aşama-2 koşusunu, onun `model_config`'iyle seçin. |
| Resmi ağırlıklar için Google Drive "quota exceeded" | Mesajdaki linkten tarayıcıyla indirip mesajdaki konuma koyun. |
| `Unknown sequence(s): [...]` | `sequences` içinde yazım hatası; mesaj 50 VOT-LT2020 adını listeler. |
| VOT dizisi indirilemiyor / `Checksum mismatch` | Ağ veya sunucu sorunu; tekrar çalıştırın (tamamlanan diziler kalır, yarıda kalan dosya devam eder). |
| `Not enough disk space for N VOT sequence(s)` | Disk alanı açın ya da daha az dizide test edin. |
| `Completed sequences: 47/50` | Hata veren dizi atlanır; bkz. `vot_workspace/logs/`. Tekrar çalıştırmak yalnızca eksikleri dener. |
| `no kernel image is available` / sm_120 | GPU, PyTorch 2.4.1 tarafından desteklenmiyor (RTX 5000 serisi); RTX 3000/4000 sınıfı bir GPU kullanın. |
| "A newer version of the VOT toolkit is available" | Görmezden gelin; vot-toolkit 0.5.3 ile test edildi. |
