from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
model_path = "best_model.keras"
data_dir = "dataset"  # or full path like r"C:\Users\Me\Desktop\skin2\dataset"
img_size = (256, 256)
batch_size = 32

# Load validation data
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load and evaluate
model = load_model(model_path)
loss, acc = model.evaluate(val_generator)
print(f"\n✅ Validation Accuracy: {acc * 100:.2f}%")
