import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Veri hazırlığı
train_dir = "../dataset"
image_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
train_data = datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='training')
valid_data = datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', subset='validation')

# Model oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(train_data, epochs=5, validation_data=valid_data)

# Modeli kaydetme
model.save("../model/brain_tumor_model.keras")
print("✅ Model başarıyla kaydedildi!")
