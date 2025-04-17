# Beyin Tümörü Tespit Projesi

Bu proje, derin öğrenme kullanarak beyin tümörü tespiti yapmak için geliştirilmiştir. TensorFlow ve OpenCV gibi kütüphaneleri kullanarak resimleri analiz eder ve tümör olup olmadığını tahmin eder.

## Dosya Yapısı


```
BrainTumorDetection/
│── model/                     # Eğitilmiş model dosyası (GitHub'a yüklenmedi)
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

```bash
pip install -r requirements.txt
```

## Kullanım

### Modeli Yükleme

Modeli yükleyip özetini görmek için:
```bash
python3 src/model.py
```

### Tahmin Yapma

Bir görsel üzerinde tahmin yapmak için:
```bash
python3 src/predict.py test_images/Test.jpeg
```

Dışarıdan bir görsel kullanmak için:
```bash
python3 src/predict.py /tam/yol/test_images/Test.jpeg
```

Eğer resim belirtilmezse, varsayılan bir test görseli kullanılır.

## Model Dosyası

Model dosyası GitHub dosya boyutu sınırını aştığı için repoya eklenmemiştir. Aşağıdaki Google Drive bağlantısından indirilebilir:

**Model İndirme Bağlantısı:** [brain_tumor_model.keras](https://drive.google.com/file/d/1m3lpRUntqI03ElS8fu8z-mU4VYjOtZ7S/view?usp=sharing)

Indirdikten sonra `model/` klasörü altına yerleştirilmelidir.

## Model Eğitimi (Opsiyonel)

Eğer modeli yeniden eğitmek istiyorsanız:
```bash
python3 src/train.py
```
Bu işlem için yeterli donanım (GPU) gerekebilir.

## Notlar

- Model dosyası `model/brain_tumor_model.keras` dizininde olmalıdır.
- Test görselleri `test_images/` klasörüne konulmalıdır.
- TensorFlow GPU desteği etkinse daha hızlı çalışır.

Herhangi bir hata durumunda terminal çıktısını kontrol ediniz ve dosya yollarının doğru olduğundan emin olun.

