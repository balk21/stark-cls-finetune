# STARK-ST eğitimi

Orijinal STARK eğitimi (`lib/train/`, resmi repodan taşındı): tek GPU'da, orijinal effective batch boyutuyla,
GOT-10k, COCO, LaSOT ve TrackingNet'in herhangi bir kombinasyonuyla.

[English](train.md) | **Türkçe**

- [Ne eğitilir](#ne-eğitilir)
- [Bizim koşularımız ve resmi ağırlıklar](#bizim-koşularımız-ve-resmi-ağırlıklar)
- [Veri setleri](#veri-setleri)
- [Çalıştırma](#çalıştırma)
- [Tipik koşular](#tipik-koşular)
- [Parametreler](#parametreler)
- [Koşu klasörü ve devam etme](#koşu-klasörü-ve-devam-etme)
- [Sekiz yerine tek GPU](#sekiz-yerine-tek-gpu)
- [Orijinal STARK eğitim kodundan farklar](#orijinal-stark-eğitim-kodundan-farklar)
- [Sık karşılaşılan sorunlar](#sık-karşılaşılan-sorunlar)

## Ne eğitilir

| Aşama | Eğitilen parametreler | Epoch (LR ÷10) | Loss | Başlangıç |
|---|---|---|---|---|
| 1 | backbone + transformer + kutu başlığı | 500 (400'de) | GIoU × 2 + L1 × 5 | ImageNet ResNet backbone |
| 2 | yalnızca sınıflandırma başlığı (geri kalanı dondurulur) | 50 (40'ta) | BCE | aşama-1 ağırlıkları (`init`) |

İki aşamada da (model YAML'ları `model_configs/stark_st1/` ve `model_configs/stark_st2/`): epoch başına 60 000
eğitim örneği, AdamW (lr 1e-4, backbone × 0.1, weight decay 1e-4), norm 0.1'de gradyan kırpma, 20 (aşama 1) / 10
(aşama 2) epoch'ta bir 10 000 örnekle doğrulama, deterministik cuDNN. Kare örnekleme, augmentation'lar ve
loss'lar orijinal koddakilerdir.

## Bizim koşularımız ve resmi ağırlıklar

STARK, dört veri setinin tamamıyla (LaSOT, GOT-10k, COCO, TrackingNet; `*_got10k_only` olanlar yalnızca GOT-10k ile)
eğitilmiş ağırlıklar yayımlar. Burada bunlar yalnızca oldukları gibi kullanılır: test için (`weights="official"`,
[test.tr.md](test.tr.md#ağırlıklar)). Bu reponun koşuları o eğitimden ayrı tutulur:

| Köken | Nasıl | Koşu adı | Listedeki başlık |
|---|---|---|---|
| ImageNet'ten | aşama 1 ImageNet backbone'undan, aşama 2 kendi aşama 1'imizin üzerine (varsayılan) | `st101_coco_stage1`, `st101_coco_stage2` | "trained here from ImageNet" |
| resmi ağırlıkların üzerine | `init="official"` ile aşama 2: sınıflandırma başlığı çıkarılmış resmi checkpoint | `st101_coco_stage2_on-official` | "trained here on top of the official STARK weights" |
| başka bir dosyanın üzerine | `init="<checkpoint yolu>"` ile aşama 2 | `..._from-<dosya>_...` | "trained here on top of another checkpoint file" |

Koşular **aile**ye göre gruplanır (model + veri setleri + örnekleme oranları): `st101_coco`, `st101_got10k+coco`, ...
Bir ailenin aşama 2'si kendi aşama-1 koşusundan başlar (`init=None`, yani aynı parametrelerle `stage=1` olan koşu);
böylece ağırlıkları yalnızca o ailenin veri setlerini görmüş olur. Köken her koşunun `train_config.json` dosyasında
saklanır ve zincir boyunca aktarılır; `python -m stark_ft weights` (notebook: `nb.list_weights()`) resmi ağırlıkları ve
bizim koşularımızı kökenlerine göre listeler. `init="official"` ile backbone, transformer ve kutu başlığı STARK'ın
aşama-1 ağırlıklarıdır (aşama 2 bunları dondurur) ve dört veri setiyle eğitilmiştir: böyle bir koşu "daha az veri
setiyle eğitilmiş" sayılmaz, bu yüzden adı ayrıdır.

## Veri setleri

| `datasets` anahtarı | STARK veri seti | Boyut | Hazırlık |
|---|---|---|---|
| `got10k` | `GOT10K_vottrain`: VOT ile örtüşen 1000 video çıkarılmış GOT-10k train (7 086 video) | 73.9 GB (GOT-10k train klasörü) | resmi arşivlerden (kayıt gerekir, aşağıya bakın) |
| `got10k_full` | `GOT10K_train_full`: GOT-10k train'in 9 335 videosunun tamamı (GOT-10k protokolü, `*_got10k_only` config'leri) | (aynı klasör) | (aynı) |
| `coco` | `COCO17`: COCO 2017 train (118 287 görüntü; her nesne tek karelik bir "video") | 20.1 GB | otomatik (images.cocodataset.org) |
| `lasot` | `LASOT`, train bölümü (1 120 video) | çok büyük | elle |
| `trackingnet` | `TRACKINGNET` (≈ 30 000 video) | ≈ 1 TB | elle |

Doğrulama (`val_datasets=["got10k"]`), `GOT10K_vottrain` ile kesişmeyen 1 249 GOT-10k train videosundan oluşan
`GOT10K_votval` bölümünü kullanır (STARK'taki gibi). Yalnızca COCO ile eğitimde bile GOT-10k gerektirir;
`val_datasets=[]` doğrulamayı kapatır.

```
<train_data>                      varsayılan data/train (configs/paths.yaml)
├── got10k/train/                 list.txt, GOT-10k_Train_000001/ ...
├── coco/                         annotations/instances_train2017.json, images/train2017/
├── lasot/                        <sınıf>/<sınıf>-<n>/
├── trackingnet/                  TRAIN_0 ... TRAIN_11/
└── _archives/{coco,got10k}/      indirilen arşivler
```

```bash
python -m stark_ft prepare-train-data --datasets got10k coco [--got10k-url URL_VEYA_YOL ...] [--archives KLASÖR]
```

- **COCO** indirilir (`train2017.zip`, `annotations_trainval2017.zip`; yarıda kalan indirme devam eder) ve
  118 287 görüntünün tamamı kontrol edilir.
- **GOT-10k** [got-10k.aitestunion.com/downloads](http://got-10k.aitestunion.com/downloads) adresinde ücretsiz kayıt
  gerektirir; indirme linkleri e-postayla gelir (örn. `full_data.zip` için bir Google Drive linki, 70.7 GB). Linkleri ya
  da indirilmiş arşivlerin yollarını `--got10k-url` ile verin (notebook: `got10k_sources`) veya arşivleri
  `<archives>/got10k/` klasörüne koyun. Google Drive paylaşım linkleri doğrudan indirme linkine çevrilir; paylaşılan
  Drive dosyaları sık sık Google'ın günlük indirme sınırına takılır ("Quota exceeded"), o durumda dosyanın kendi
  Drive'ınızdaki bir kopyasını kullanın (bkz. [colab.tr.md](colab.tr.md#google-driveda-got-10k)). Yalnızca train
  videoları ve `list.txt` açılır, arşiv içindeki arşivlerden de; 9 335 videonun tamamı bulunmalıdır. Aynı arşiv iki kez
  bulunursa (örn. kendi kopyanız ve paylaşılan dosyaya bir kısayol) yalnızca bir kez açılır.
- Arşivler saklanır; başka bir makine / oturum yalnızca açma işlemini yapar. Açma işlemi geçici bir klasöre yapılır ve
  tamamlanınca yeniden adlandırılır, önce boş disk alanı kontrol edilir, beklenen yapıdaki bir veri seti klasörü
  yalnızca okunur. Zaten açılmış bir GOT-10k kopyası doğrudan kullanılabilir: `train_data`'yı
  `<train_data>/got10k/train/list.txt` var olacak şekilde ayarlayın.

## Çalıştırma

```bash
# GOT-10k + COCO ile STARK-ST101: ImageNet'ten aşama 1, ardından onun üzerine aşama 2 (STARK-ST50: model_config=baseline)
python -m stark_ft train --set 'datasets=[got10k, coco]'               # st101_got10k+coco_stage1
python -m stark_ft train --set 'datasets=[got10k, coco]' --set stage=2 # st101_got10k+coco_stage2
# STARK'ın ağırlıkları üzerinde yalnızca sınıflandırma başlığı (on-official olarak işaretlenir)
python -m stark_ft train --set 'datasets=[coco]' --set stage=2 --set init=official --set 'val_datasets=[]'
python -m stark_ft train --config configs/train_example.yaml
python -m stark_ft weights                      # resmi ağırlıklar ve bizim koşularımız, kökene göre
python -m stark_ft train ... --dry-run          # yalnızca parametreleri ve veriyi kontrol eder
python -m stark_ft train-report <koşu adı>      # ilerleme, son loss'lar, history.png
python -m stark_ft train-list
```

Notebook: `notebooks/train.ipynb`. İlerleme çıktısı saniyedeki örnek sayısını, epoch'un ve eğitimin kalan süresini
gösterir.

## Tipik koşular

| Hedef | `stage` | `datasets` | `init` | `val_datasets` |
|---|---|---|---|---|
| COCO, aşama 1 | `1` | `["coco"]` | `None` | `[]` |
| COCO, aşama 2 | `2` | `["coco"]` | `None` (= `st101_coco_stage1`) | `[]` |
| GOT-10k, aşama 1 | `1` | `["got10k"]` | `None` | `["got10k"]` |
| GOT-10k, aşama 2 | `2` | `["got10k"]` | `None` (= `st101_got10k_stage1`) | `["got10k"]` |
| GOT-10k + COCO, aşama 1 | `1` | `["got10k", "coco"]` | `None` | `["got10k"]` |
| GOT-10k + COCO, aşama 2 | `2` | `["got10k", "coco"]` | `None` (= `st101_got10k+coco_stage1`) | `["got10k"]` |
| STARK'ın ağırlıkları üzerinde yalnızca sınıflandırma başlığı | `2` | örn. `["coco"]` | `"official"` | `[]` |
| Yalnızca GOT-10k, aşama 1 olmadan | `2` | `["got10k_full"]` | `"official"`, `model_config="baseline_R101_got10k_only"` ile | `["got10k"]` |

Aşama 2, aşama-1 satırının `stage=2` ile aynısıdır; başka bir aşama-1 koşusu adıyla verilebilir (`init`,
`<train_outputs>` içindeki klasör adı). `"official"` satırları STARK'ın ağırlıkları üzerine kurulur ve `on-official`
olarak işaretlenir.
`val_datasets=["got10k"]` eğitimde kullanılmayan GOT-10k videolarında doğrulama yapar; GOT-10k yoksa `[]` kullanın.
STARK'ın yalnızca GOT-10k ağırlıkları, `got10k`'nın dışarıda bıraktığı VOT ile örtüşen 1 000 video dahil 9 335
GOT-10k videosunun tamamını kullandı.

## Parametreler

`stark_ft/train/config.py` içindeki `TrainConfig`. Notebook: `TRAIN = dict(...)`; CLI: `--config dosya.yaml` ve/veya
`--set anahtar=değer`. `None` orijinal STARK değeri demektir.

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `name` | `None` | Koşu klasörü `<train_outputs>/<name>/`. `None`: `<aile>_stage<N>[_e<epochs>][_on-official / _from-<init>]`, örn. `st101_got10k+coco_stage1`. |
| `model_config` | `"baseline_R101"` | `baseline_R101` (STARK-ST101) veya `baseline` (STARK-ST50). `*_got10k_only` config'leri yalnızca veride farklıdır (`datasets=["got10k_full"]` ile kullanın); aşama 2'de `init` olarak resmi GOT-10k-only checkpoint'ini seçerler. |
| `stage` | `1` | `1` veya `2`. |
| `init` | `None` | Aşama 1: `None` (ImageNet backbone). Aşama 2: `None` = aynı parametrelerle bizim aşama-1 koşumuz, başka bir aşama-1 koşu adı, `"official"` (STARK'ın ağırlıkları; `on-official` olarak işaretlenir) ya da bir checkpoint yolu. |
| `datasets` | `["got10k"]` | `got10k` (veya `got10k_full`), `coco`, `lasot`, `trackingnet` değerlerinin boş olmayan herhangi bir kombinasyonu. Sıra önemli değildir. |
| `dataset_ratios` | `None` | Veri seti başına örnekleme ağırlığı (`datasets` ile aynı sırada); her örnek önce bu ağırlıklarla bir veri seti, sonra bir video seçer. `None`: eşit. |
| `val_datasets` | `["got10k"]` | `["got10k"]` veya `[]`. |
| `epochs` | `None` | 500 (aşama 1) / 50 (aşama 2). |
| `lr_drop_epoch` | `None` | Bu epoch'tan sonra öğrenme oranı ÷ 10: 400 / 40. |
| `samples_per_epoch` | `None` | 60 000. |
| `val_samples_per_epoch` | `None` | 10 000. |
| `val_interval` | `None` | 20 / 10 epoch'ta bir doğrulama. |
| `effective_batch` | `128` | Optimizer adımı başına örnek. 128 = orijinal (8 GPU × 16); değiştirmek eğitimi değiştirir. |
| `micro_batch` | `16` | Bir GPU geçişindeki örnek sayısı; `effective_batch / micro_batch` geçiş biriktirilir. Yalnızca bellek ve hız buna bağlıdır. |
| `num_workers` | `8` | Veri yükleme süreçleri. |
| `keep_every` | `None` | Her N epoch'ta ağırlıkları sakla: 50 / 10. |

## Koşu klasörü ve devam etme

| `<train_outputs>/<koşu adı>/` içindeki dosya | İçerik |
|---|---|
| `train_config.json` | Parametreler, köken, veri seti klasörleri, başlangıç ağırlıkları, GPU, TF32 ayarı, kod hash'i, tarih |
| `history.csv`, `history.png` | Epoch başına bir satır: süre, öğrenme oranı, eğitim / doğrulama loss'ları; grafik (kesikli: LR düşüşleri) |
| `logs/train.log` | İlerleme çıktısı |
| `checkpoints/latest.pth.tar` | Son tamamlanan epoch'tan sonraki eksiksiz durum (ağ, optimizer, scheduler, rastgele üreteçler); her epoch atomik olarak yazılır. ST101: 0.56 GB (aşama 1), 0.19 GB (aşama 2) |
| `checkpoints/STARKST_epXXXX.pth.tar` | Her `keep_every` epoch'ta ve sonda ağırlıklar (her biri 0.19 GB) |
| `final.pth.tar` | Son ağırlıklar |

- Aynı parametrelerle tekrar çalıştırmak `latest.pth.tar` dosyasından **devam eder**; sonuç kesilmemiş bir koşuyla bit
  düzeyinde aynıdır (eğitim sırasında öldürülen bir koşuyla doğrulandı). Aynı `name` farklı parametrelerle ya da eğitim
  kodunun farklı bir sürümüyle (`lib/train`, `lib/models`, `lib/config`, `lib/utils`, `model_configs/stark_st1|2`,
  `stark_ft/train/run.py`) hata verir; başka bir `name` ya da `--overwrite` kullanın.
- Tamamlanmış bir aşama-2 koşusu `weights="<koşu adı>"` ile test edilir (aynı `model_config`), bkz.
  [test.tr.md](test.tr.md#ağırlıklar). Aşama-1 koşularının sınıflandırma başlığı eğitilmemiştir ve test edilemezler;
  aşama 2'nin başlangıç noktasıdırlar.
- `checkpoints` klasörüne hiçbir şey yazılmaz.

## Sekiz yerine tek GPU

Orijinal model, her birinde 16 örnek olan 8 GPU ile eğitildi; DistributedDataParallel gradyanlarının ortalamasını
alır, yani her optimizer adımı 128 örneğin ortalama gradyanını kullanır. Burada her mikro-batch'in loss'u
`effective_batch / micro_batch`'e bölünür, gradyanlar biriktirilir, ardından gradyan kırpılıp adım atılır. STARK
backbone'daki tüm BatchNorm katmanlarını dondurduğu için bir örneğin ileri geçişi batch'in geri kalanına bağlı
değildir: bu aynı güncellemedir. Sayısal olarak 8 × 16 örneğin biriktirilmiş gradyanı ile tek bir 128'lik batch'in
gradyanı float64'te 3·10⁻¹⁵ (bağıl), float32'de ≈ 10⁻⁴ (toplama sırası) farklıdır. Epoch başına adım sayısı da
aynıdır (60 000 / 128 = 468). Orijinalden farklı olan yalnızca rastgele örnek akışıdır (kendi veri işçilerine sahip
8 süreç); aynı GPU tipinde, `micro_batch` ve `num_workers` dahil aynı parametrelerle yapılan koşular bit düzeyinde
aynıdır.

**Süre:** aşama 1 30 M, aşama 2 3 M örnek işler. RTX 3060 laptop GPU'da ölçülen: aşama 2 ≈ 35 örnek/sn (≈ 24 saat),
aşama 1 ≈ 20 örnek/sn (≈ 17 gün); veri merkezi GPU'ları birkaç kat hızlıdır. Ampere ve sonrası GPU'lar
konvolüsyonlarda varsayılan olarak TF32 kullanır, T4 kullanmaz; koşuları aynı GPU tipinde karşılaştırın
(bkz. [test.tr.md](test.tr.md#gpular-arası-sonuçlar)).

## Orijinal STARK eğitim kodundan farklar

| Orijinal STARK | Burada |
|---|---|
| 8 GPU (DistributedDataParallel), GPU başına 16 örnek | 1 GPU, aynı effective batch'e gradyan biriktirme ile |
| Veri seti yolları makineye özel bir `local.py` içinde | `train_data` (`configs/paths.yaml`) ve `datasets` parametresi |
| Devam ettirilen bir aşama-2 koşusu, devam ettikten sonra aşama-1 ağırlıklarını yeniden yükleyip eğitilmiş sınıflandırma başlığını eziyordu | Başlangıç ağırlıkları yalnızca koşu sıfırdan başlarken yüklenir |
| Checkpoint yalnızca son 10 epoch'ta ve 100 epoch'ta bir, scheduler ve rastgele üreteç durumları olmadan | Her epoch'tan sonra eksiksiz durum (atomik), her `keep_every` epoch'ta ağırlıklar; bit düzeyinde aynı devam |
| Checkpoint'lerde pickle'lanmış settings nesneleri | Düz sözlükler |
| Yeni PyTorch / pandas ile çalışmıyordu (`torch._six`, `storage()._new_shared`, `read_csv(squeeze=True)`) | Davranış değiştirilmeden düzeltildi |

Orijinaldeki gibi bırakılan: aşama 1'de ağ, bir batch'teki herhangi bir örnek için geçersiz bir kutu (x2 < x1 veya
y2 < y1) tahmin ederse GIoU hesabı başarısız olur ve o batch'in GIoU loss'u 0 sayılır (yalnızca L1 kullanılır). Bu,
sıfırdan eğitimin ilk adımlarında olur ve hızla kaybolur; burada mikro-batch başına uygulanır (varsayılan 16 örnek;
orijinalde 8 GPU'nun her birindekiyle aynı).

## Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| `CUDA out of memory` | `micro_batch`'i düşürün (örn. 8), `effective_batch=128` kalsın; güncelleme aynı kalır. |
| Makine çöküyor veya RAM yetmiyor | `num_workers`'ı düşürün (COCO'nun anotasyon dosyası büyüktür ve her veri işçisi bellek kullanır). |
| `Not enough disk space for ...` | Açılmış veri setleri sığmıyor (GOT-10k ≈ 74 GB, COCO ≈ 20 GB). Daha büyük disk veya daha az veri seti kullanın. |
| `Stage 2 starts from the stage-1 run of the same family ...` | Önce aşama 1'i eğitin (aynı parametreler, `stage=1`) ya da `init` olarak tamamlanmış başka bir aşama-1 koşusu / `"official"` verin. |
| `No GOT-10k archive in ...` | GOT-10k arşivlerini mesajdaki klasöre koyun (klasör oluşturulur) ya da link / arşiv yollarını verin. |
| `... 'Google Drive - Quota exceeded'` | Paylaşılan dosya günlük indirme sınırına ulaştı; kendi kopyanızı kullanın ([colab.tr.md](colab.tr.md#google-driveda-got-10k)). |
| `... already exists with different parameters` / `different version of the training code` | Başka bir `name` ya da sıfırdan koşu için `--overwrite` (notebook: `overwrite=True`) kullanın. |
