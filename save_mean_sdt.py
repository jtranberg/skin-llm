import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Settings ===
data_dir = r'C:\Users\Me\Desktop\skin2\dataset'
img_height, img_width = 256, 256
batch_size = 32

# === Prepare Data Generator (only training set) ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=False
)

# === Extract all batches and compute mean/std ===
print("📊 Extracting batches for mean/std calculation...")
train_images = []
for i in range(len(train_generator)):
    batch_x, _ = train_generator[i]
    train_images.append(batch_x)

all_images = np.concatenate(train_images, axis=0)
mean = np.mean(all_images, axis=0)
std = np.std(all_images, axis=0)

# === Save files ===
os.makedirs("model", exist_ok=True)
np.save("model/mean.npy", mean)
np.save("model/std.npy", std)

print("✅ Saved mean.npy and std.npy to model/")
