from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

data_dir = r'C:\Users\Me\Desktop\skin2\dataset'
img_height, img_width = 256, 256
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("🔍 Class indices:", train_gen.class_indices)
print("📊 Training class distribution:", np.bincount(train_gen.classes))
print("📊 Validation class distribution:", np.bincount(val_gen.classes))
print("🧪 Training samples:", train_gen.samples)
print("🧪 Validation samples:", val_gen.samples)
