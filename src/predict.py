import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Geçerli dosyanın mutlak yolunu belirleme (predict.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Model yolunu otomatik olarak belirleme
MODEL_PATH = os.path.join(script_dir, "..", "model", "brain_tumor_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Görsel yolunu belirleme
if len(sys.argv) < 2:
    print("⚠️ Görsel belirtilmedi, varsayılan görsel kullanılacak.")
    image_path = os.path.join(script_dir, "..", "test_images", "Test_2.jpg")
else:
    image_path = os.path.abspath(sys.argv[1])  # Yolu mutlak hale getirme

print(f"📂 Kullanılan görselin yolu: {image_path}")

# Görselin varlığını kontrol etme
if not os.path.exists(image_path):
    print(f"❌ Hata: Görsel yolu mevcut değil: {image_path}")
    sys.exit(1)

# Test görselini yükleme
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))  # Modelle uyumlu boyutlandırma
image = image / 255.0  # Görseli normalize etme
image = np.expand_dims(image, axis=0)  # Batch boyutunu ekleme

# Tahmin yapma
prediction = model.predict(image)[0][0]
label = "Beyin Tümörü" if prediction > 0.5 else "Sağlıklı"

# Sonucu gösterme
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title(f"Tahmin: {label} ({prediction:.2%})")
plt.axis("off")
plt.show()

print(f"✅ Tahmin: {label} - %{prediction:.2%}")
