# Beyin Tümörü Tespit Projesi

Bu proje, derin öğrenme kullanarak beyin tümörü tespiti yapmak için geliştirilmiştir. TensorFlow ve OpenCV gibi kütüphaneleri kullanarak resimleri analiz eder ve tümör olup olmadığını tahmin eder.

## Dosya Yapısı

```
BrainTumorDetection/
│── model/                    # Eğitilmiş model dosyası
│   ├── brain_tumor_model.keras
│── dataset/                   # Eğitim verileri
│   ├── tumor/                 # Tümörlü hastaların görselleri
│   ├── normal/                # Sağlıklı bireylerin görselleri
│── src/                       # Proje kaynak kodları
│   ├── train.py               # Model eğitimi için Python betiği
│   ├── predict.py             # Resim için tahmin yapma betiği
│   ├── model.py               # Modelin yüklenmesi ve özeti
│── requirements.txt           # Gerekli Python kütüphaneleri
│── README.md                  # Proje açıklamaları ve kurulum talimatları
```

## Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki kütüphanelerin yüklenmesi gerekmektedir:

```sh
pip install -r requirements.txt
```

## Kullanım

### Modeli Yükleme

Modeli yükleyip özetini görmek için:

```sh
python3 src/model.py
```

### Tahmin Yapma

Bir görsel üzerinde tahmin yapmak için:

```sh
python3 src/predict.py test_images/Test.jpeg
```

Dışarıdaki Bir Görseli Kullanma:

```sh
python3 src/predict.py /tam/yol/test_images/Test.jpeg
```

Eğer resim belirtilmezse, varsayılan bir test görseli kullanılır.

## Model Eğitimi

Eğer modeli yeniden eğitmek istiyorsanız:

```sh
python3 src/train.py
```

Ancak, bu işlem için yeterli donanım kaynağına ihtiyacınız olabilir.

## Notlar

- Model dosyası `model/brain_tumor_model.keras` dizininde bulunmalıdır.
- Test görselleri `test_images/` dizininde yer almalıdır.
- Kod, GPU olmadan da çalışabilir ancak eğer TensorFlow GPU desteği etkinse daha hızlı çalışır.

Herhangi bir hata alırsanız, dosya yollarının doğru olduğundan emin olun ve terminalde hata mesajlarını kontrol edin.
