import os
import pandas as pd
import shutil

# Paths
metadata_csv = r"C:\Users\Me\Desktop\skin2\dataset\HAM10000_metadata.csv"
image_dir1 = r"C:\Users\Me\Desktop\skin2\dataset\HAM10000_images_part_1"
image_dir2 = r"C:\Users\Me\Desktop\skin2\dataset\HAM10000_images_part_2"
output_dir = r"C:\Users\Me\Desktop\skin2\dataset"

# Diagnosis to class mapping
malignant_labels = ['mel']
benign_labels = ['nv', 'bkl', 'bcc', 'akiec', 'df', 'vasc']

# Read metadata
df = pd.read_csv(metadata_csv)

# Create output folders
os.makedirs(os.path.join(output_dir, "Benign"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "Malignant"), exist_ok=True)

# Combine both image folders
image_lookup = {img: os.path.join(image_dir1, img) for img in os.listdir(image_dir1)}
image_lookup.update({img: os.path.join(image_dir2, img) for img in os.listdir(image_dir2)})

# Sort images into folders
for idx, row in df.iterrows():
    img_file = row["image_id"] + ".jpg"
    label = row["dx"]

    src_path = image_lookup.get(img_file)
    if not src_path or not os.path.exists(src_path):
        continue

    if label in malignant_labels:
        dest = os.path.join(output_dir, "Malignant", img_file)
    elif label in benign_labels:
        dest = os.path.join(output_dir, "Benign", img_file)
    else:
        continue  # skip unknowns

    shutil.copy(src_path, dest)

print("✅ Benign and Malignant folders created successfully.")
