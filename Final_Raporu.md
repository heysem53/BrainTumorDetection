### Beyin Tümörü Tespiti Projesi - Heysem Eliyad - 22040301150

#### 1. **Projenin Amacı ve Genel Tanım**

Bu proje, derin öğrenme yöntemlerini kullanarak beyin tümörlerini tespit etmek amacıyla geliştirilmiştir. Proje, TensorFlow ve Keras kütüphanelerini kullanarak bir sinir ağı modelini eğitmekte ve bu modelin, beyin tümörlü ve sağlıklı kişilerin görselleri üzerinde tahmin yapmasını sağlamaktadır. Model, görsellerin analiz edilmesi ve sınıflandırılması için bir CNN (Convolutional Neural Network) mimarisi kullanır.

#### 2. **Veri Seti (Dataset) Hazırlığı**

Projenin temelinde, beyin tümörlü hastalar ile sağlıklı bireylerin beyin MR görüntülerinden oluşan bir veri seti bulunmaktadır. Veri seti, şu şekilde iki ana klasöre ayrılmıştır:

- **tumor/**: Beyin tümörü taşıyan bireylerin MRI görüntüleri.
- **normal/**: Sağlıklı bireylerin MRI görüntüleri.

Bu görüntüler, modelin eğitim sürecinde kullanılmıştır. Veri seti, modelin tümörlü ve sağlıklı beyin dokularını öğrenmesine olanak tanımaktadır.

#### 3. **Derin Öğrenme Modelinin Eğitimi**

Eğitim aşaması, derin öğrenme modelinin oluşturulup eğitilmesi için aşağıdaki adımlarla gerçekleştirilmiştir:

1. **Modelin Tasarımı:**

   - Model, konvolüsyonel sinir ağları (CNN) mimarisi kullanılarak tasarlanmıştır.
   - Veri setindeki görsellerin boyutu 224x224 px'e indirgenmiş ve her bir görselin piksel değerleri [0, 1] aralığına normalleştirilmiştir.

2. **Modelin Eğitilmesi:**

   - Eğitim sırasında model, **Adam optimizasyon algoritması** ve **binary crossentropy loss fonksiyonu** kullanarak optimize edilmiştir.
   - Modelin katmanları arasında **Conv2D**, **MaxPooling2D**, **Flatten**, ve **Dense** katmanları bulunmaktadır.
   - Eğitim işlemi sonucunda modelin doğruluk oranı ve kaybı izlenerek uygun parametrelerle eğitim yapılmıştır.

   Eğitim kodu:

   ```python
   import tensorflow as tf
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   # Eğitim verisini hazırlama
   train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
   train_generator = train_datagen.flow_from_directory('dataset', target_size=(224, 224), batch_size=32, class_mode='binary')

   # Modeli tanımlama
   model = tf.keras.Sequential([
       tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
       tf.keras.layers.MaxPooling2D((2, 2)),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(1, activation='sigmoid')
   ])

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

   # Modeli eğitme
   model.fit(train_generator, epochs=10)
   model.save('brain_tumor_model.keras')
   ```

   Eğitim sırasında, model kayıpları ve doğruluk oranı takip edilmiştir. Eğitim sonunda elde edilen model `brain_tumor_model.keras` dosyasına kaydedilmiştir.

#### 4. **Eğitilmiş Modelin Kullanıma Hazırlanması**

Model eğitildikten sonra, kullanıcının eğitilmiş modelin yüklenebilmesi ve yeni görsellerle tahmin yapabilmesi için model dosyası (brain_tumor_model.keras) proje dizininde `model/` klasörüne yerleştirilmiştir. Bu, modelin herhangi bir ortamda kolayca kullanılabilmesini sağlamaktadır.

#### 5. **Proje Yapısının Düzenlenmesi ve Hiyerarşisi**

Proje, her aşamada kullanıcıya kolaylık sağlaması için uygun bir dosya yapısına sahip olacak şekilde organize edilmiştir. Proje yapısının genel görünümü şu şekildedir:

```plaintext
BrainTumorDetection/
│── model/                    # Eğitilmiş modelin bulunduğu klasör
│   ├── brain_tumor_model.keras
│── dataset/                   # Eğitim için kullanılan görsellerin bulunduğu klasör
│   ├── tumor/                 # Tümörlü hastaların görselleri
│   ├── normal/                # Sağlıklı kişilerin görselleri
│── src/                       # Projenin kaynak kodları
│   ├── model.py               # Modeli yüklemek için kullanılan dosya
│   ├── predict.py             # Yeni görseller ile tahmin yapmak için kullanılan dosya
│── test_images/               # Test için kullanılan görsellerin bulunduğu klasör
│   ├── Test.jpeg              # Örnek test görseli
│── requirements.txt           # Proje için gerekli Python kütüphaneleri
│── README.md                  # Projeyi tanımlayan ve kullanımı açıklayan dosya
```

#### 6. **Projenin Çalıştırılması**

Projeyi çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Gerekli Kütüphanelerin Yüklenmesi:**
   Projeyi çalıştırmak için gerekli olan Python kütüphaneleri `requirements.txt` dosyasına eklenmiştir. Bu dosyayı kullanarak gerekli kütüphaneleri yükleyebilirsiniz:

   ```bash
   pip install -r requirements.txt
   ```

2. **Modelin Yüklenmesi ve Tahmin Yapma:**
   Eğitilmiş modeli kullanarak yeni bir görsel üzerinde tahmin yapmak için `src/predict.py` dosyasını çalıştırabilirsiniz. Örnek olarak, aşağıdaki komutla tahmin yapılabilir:

   ```bash
   python3 src/predict.py test_images/Test.jpeg
   ```

   Burada `test_images/Test.jpeg` görseli, tahmin yapılacak görseldir. Eğer başka bir görsel kullanmak isterseniz, görselin tam yolunu belirtmeniz gerekir.

#### 7. **Proje Kullanımına İlişkin Notlar**

- **Eğitim ve Tahmin:** Model, yeni bir görsel üzerinde tahmin yapmak için tasarlanmış olup, `predict.py` dosyasını çalıştırarak herhangi bir görseli analiz edebilirsiniz.
- **Modeli Yeniden Eğitmek:** Eğer modelin yeniden eğitilmesi gerekiyorsa, eğitim verisiyle `train.py` dosyasını çalıştırarak eğitimi başlatabilirsiniz.
- **Dışarıdan Görsel Kullanma:** Eğer görsel dışarıda bir konumda ise, tam yol ile görseli gösterebilirsiniz.

#### 8. **Sonuçlar ve Gelecek Adımlar**

Bu proje, temel bir beyin tümörü tespit modeli sağlamaktadır. Gerçek dünya uygulamaları için daha fazla eğitim verisi ve daha derin model mimarileri kullanılabilir. Ayrıca, modelin doğruluğunu artırmak için hiperparametre optimizasyonu yapılabilir. Gelecekte bu sistemin farklı organlar ve hastalıklar üzerinde çalışacak şekilde genişletilmesi mümkündür.
