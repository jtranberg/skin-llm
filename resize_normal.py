from PIL import Image
import os
import shutil

# Config
INPUT_DIR = os.path.expanduser("~/Desktop/normal_skin")
OUTPUT_DIR = r"C:\Users\Me\Desktop\skin2\dataset\Normal"  # <- absolute path
TARGET_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"normal_{count}.jpg")

        try:
            img = Image.open(input_path).convert("RGB")
            img = img.resize(TARGET_SIZE)
            img.save(output_path)
            print(f"✅ Saved: {output_path}")  # 👈 Print exact save location
            count += 1
        except Exception as e:
            print(f"⚠️ Failed to process {filename}: {e}")

print(f"\n✅ Processed and moved {count} images to: {OUTPUT_DIR}")
