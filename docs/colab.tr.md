# Google Colab

[English](colab.md) | **Türkçe**

| Notebook | |
|---|---|
| Eğitim | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/train.ipynb) |
| Test | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/test.ipynb) |
| Karşılaştırma | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/balk21/stark-cls-finetune/blob/main/notebooks/compare.ipynb) |

Bir GPU oturumu seçin (*Runtime → Change runtime type*; eğitim için A100 veya L4) ve hücreleri sırayla çalıştırın.
İlk hücre depoyu klonlar ve `nb.init(colab_drive=...)` çağırır: Google Drive bağlanır ve diğer makinelerdekiyle aynı
`vot1` ortamı geri yüklenir (ilk sefer micromamba ile kurulur, ~5 dk, sonra Drive'da saklanır).

Oturumdan sonra da kalması gereken her şey `colab_drive` ile verilen Drive klasöründe tutulur
(varsayılan `MyDrive/LOKAP`):

| Drive klasörü | İçerik |
|---|---|
| `cache/` | ortam (7.8 GB) ve bir testin kullandığı her VOT dizisi (`votlt2019_sequences/<dizi>.tar`; 50'si: 17.6 GB); gerektiğinde yerel diske geri yüklenir |
| `checkpoints/` | indirilen / kendi checkpoint'leriniz (`checkpoints/stark_st2/<model_config>/`), yerel diske kopyalanır |
| `train_archives/{coco,got10k}/` | eğitim veri seti arşivleri; her oturumda yerel diske açılır |
| `outputs/` | test deneyleri (yarıda kalanlar devam eder) |
| `training/` | eğitim koşuları (yarıda kalanlar devam eder) |

Notlar:

- Notebook'ları yukarıdaki rozetlerden açın. Önceden açılmış bir notebook eski hücrelerini korur; ilk hücre kodu
  günceller (`git pull`), notebook'u değil.
- Kod güncellenirken bir oturum açıksa *Runtime → Disconnect and delete runtime* ile oturumu silip yeniden başlayın.
- Sonuçlar GPU tipine bağlıdır (A100 / L4'te TF32 var, T4'te yok); bkz. [test.tr.md](test.tr.md#gpular-arası-sonuçlar).
- Eğitim günler sürer; tamamlanan her epoch Drive'a kaydedilir, yeni bir oturum koşuya devam eder. Colab compute
  unit'leri saat başına harcanır.

## Google Drive'da GOT-10k

GOT-10k indirme linkleri (kayıttan sonra e-postayla gelir) Google Drive'a işaret edebilir, örn. `full_data.zip`
(70.7 GB). Böyle paylaşılan dosyalar sık sık Google'ın günlük indirme sınırına takılır ("Quota exceeded"); o durumda ne
indirme ne de kısayol çalışır. Bunun yerine kendi kopyanızı kullanın (Drive içinde yapılır, hiçbir şey indirilmez;
Drive'da 70.7 GB yer gerekir):

1. Linki tarayıcıda (Colab'de kullandığınız Google hesabıyla) açın → *Drive'a kısayol ekle*.
2. Kısayola sağ tık → *Kopyasını oluştur*.
3. Kopyayı `<colab_drive>/train_archives/got10k/` klasörüne taşıyın (`train.ipynb`'nin veri adımı bu klasörü
   oluşturur), kısayolu silin ve `train.ipynb`'yi `got10k_sources=[]` ile çalıştırın.

Arşiv sonra Drive'dan okunur ve her oturumun başında yalnızca train videoları yerel diske açılır (≈ 74 GB, yaklaşık
yarım saat); çünkü eğitim sırasında görüntüleri Drive'dan okumak çok daha yavaş olurdu. Hazırlık önce boş disk alanını
kontrol eder.
