import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import datetime
import glob
import shutil

# === STEP 0: Augment Normal class ===
normal_dir = r'C:\Users\Me\Desktop\skin2\dataset\Normal'
augmented_dir = os.path.join(normal_dir, 'augmented')
os.makedirs(augmented_dir, exist_ok=True)

normal_images = glob.glob(os.path.join(normal_dir, '*.jpg'))
print(f"📂 Found {len(normal_images)} original Normal images. Augmenting...")

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    brightness_range=[0.8, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

image_count = 0
for img_path in normal_images:
    img = load_img(img_path, target_size=(256, 256))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    aug_iter = datagen.flow(x, batch_size=1)
    for i in range(10):  # Generate 10 augmentations per image
        aug_img = next(aug_iter)[0].astype(np.uint8)
        filename = f"aug_{image_count}.jpg"
        tf.keras.utils.save_img(os.path.join(augmented_dir, filename), aug_img)
        image_count += 1

print(f"✅ Augmented {image_count} images in: {augmented_dir}")

# === STEP 0.1: Merge augmented images into main folder ===
for file in glob.glob(os.path.join(augmented_dir, '*.jpg')):
    shutil.move(file, os.path.join(normal_dir, os.path.basename(file)))
print(f"📂 Merged {image_count} augmented images into Normal/")
shutil.rmtree(augmented_dir)

# === STEP 1: Define paths and image parameters ===
data_dir = r'C:\Users\Me\Desktop\skin2\dataset'
img_height, img_width = 256, 256
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

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

# === STEP 2: Load or initialize model ===
best_model_path = "best_model.keras"
if os.path.exists(best_model_path):
    print("🔁 Loading existing best model from checkpoint...")
    model = load_model(best_model_path)
else:
    print("🆕 No checkpoint found. Starting fresh with ResNet50...")
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=SGD(learning_rate=1e-5, momentum=0.9),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

# === STEP 3: Compute class weights ===
class_labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights = dict(enumerate(class_weights_array))

# === STEP 4: Setup callbacks ===
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir)

checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# === STEP 5: Train model ===
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, lr_scheduler, tensorboard_cb]
)

# === STEP 6: Final evaluation ===
loss, accuracy = model.evaluate(val_generator)
print(f"\n✅ Final Validation Accuracy: {accuracy * 100:.2f}%")

# === STEP 7: Plot metrics ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
