import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 📁 Step 1: Define paths and image size
data_dir = r'C:\Users\Me\Desktop\skin2\dataset'  # Replace with your actual path
img_height, img_width = 224, 224
batch_size = 32

# 🧠 Step 2: Load ResNet50 base model with ImageNet weights (without top)
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Unfreeze the last 30 layers of ResNet50 for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# 🧠 Step 3: Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)  # Update the number of classes if needed

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# 🧠 Step 4: Compile the model with SGD optimizer (learning rate set to 1e-5 for fine-tuning)
model.compile(optimizer=SGD(learning_rate=1e-5, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 🧪 Step 5: Data Augmentation for Training and Validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 🧠 Step 6: Callbacks setup
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# 🏃 Step 7: Train the model
history = model.fit(
    train_generator,
    epochs=30,  # Set your desired number of epochs
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)

# 🧪 Step 8: Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 🖼️ Step 9: Plot accuracy and loss
# Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
