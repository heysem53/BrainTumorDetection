import tensorflow as tf
import numpy as np
import cv2  # Görüntüler için OpenCV kütüphanesi
import tensorflow as tf

# Yeni dizinden modeli yükleme
model = tf.keras.models.load_model("../model/brain_tumor_model.keras")  # Yol güncellenmiştir

# Model özetini görüntüleme
model.summary()
