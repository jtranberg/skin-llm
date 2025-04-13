Skin Diagnosis Model
Description
This repository contains code to train a skin diagnosis model using the HAM10000 dataset. The model is built using ResNet50 and fine-tuned for skin cancer classification.

Setup Instructions
1. Set up a Virtual Environment
To start, create and activate a virtual environment to isolate project dependencies.

powershell
Copy
# Create a virtual environment (venv) in the project directory
python -m venv venv

# Activate the virtual environment
# For Windows:
.\venv\Scripts\activate
2. Install Dependencies
Install the necessary libraries and dependencies for the project.

powershell
Copy
# Install dependencies
pip install tensorflow matplotlib numpy pandas scikit-learn
pip install kaggle  # For downloading datasets from Kaggle
3. Set up Kaggle API Key
Make sure you have the Kaggle API key (kaggle.json). If not, create an API key by following this guide. Once you have the kaggle.json file:

powershell
Copy
# Create the Kaggle config folder
mkdir ~/.kaggle

# Copy the Kaggle API key to the configuration folder
cp "path_to_kaggle.json" ~/.kaggle/kaggle.json

# Change file permissions for the Kaggle API key
chmod 600 ~/.kaggle/kaggle.json
4. Download the Dataset
Download the HAM10000 dataset using the Kaggle API:

powershell
Copy
# Download the HAM10000 dataset from Kaggle
kaggle datasets download -d kmader/=
This will download the dataset as a zip file to the current directory.

5. Unzip the Dataset
Once the dataset is downloaded, unzip it to your desired directory using PowerShell:

powershell
Copy
# Create a directory for the dataset
mkdir C:\Users\Me\Desktop\skin2\dataset

# Unzip the downloaded dataset
Expand-Archive -Path "C:\Users\Me\Desktop\skin2\skin-cancer-mnist-ham10000.zip" -DestinationPath "C:\Users\Me\Desktop\skin2\dataset" -Force
This will unzip the contents into the dataset directory.

6. Directory Structure
The HAM10000 dataset will now be available in the C:\Users\Me\Desktop\skin2\dataset directory. It contains subdirectories with images that will be used for training.

7. Training the Model
Now that the dataset is set up, use the following code to fine-tune the ResNet50 model for skin cancer classification:

python
Copy
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt

# 📁 Define paths and image size
data_dir = 'C:/Users/Me/Desktop/skin2/dataset'  # Path to the unzipped dataset
img_height, img_width = 224, 224
batch_size = 32

# 🧠 Load ResNet50 base model with ImageNet weights (without top)
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Unfreeze the last 30 layers of ResNet50 for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# 🧠 Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(12, activation='softmax')(x)  # Change 12 to match your number of classes

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# 🧠 Compile the model with SGD optimizer
model.compile(optimizer=SGD(learning_rate=1e-5, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# 🧪 Data Augmentation for Training and Validation
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

# 🧠 Callbacks setup
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# 🏃 Train the model
history = model.fit(
    train_generator,
    epochs=30,  # Adjust the number of epochs as needed
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)

# 🧪 Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 🖼️ Plot accuracy and loss
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
8. Running the Code
To run the training script, ensure you're in the correct environment and have the dataset properly downloaded and unzipped. Execute the Python script with:

powershell
Copy
# Run the script
python skin.py
9. Results
After training, you can check the model's performance by looking at the validation accuracy and loss graphs.