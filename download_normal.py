import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Setup
KAGGLE_DATASET = 'jessicali9530/lfw-dataset'
DOWNLOAD_PATH = 'data/lfw'
EXTRACTED_DIR = os.path.join(DOWNLOAD_PATH, 'lfw-deepfunneled')
NORMAL_CLASS_DIR = 'dataset/normal'

# Authenticate Kaggle
api = KaggleApi()
api.authenticate()

# Download and unzip
os.makedirs(DOWNLOAD_PATH, exist_ok=True)
api.dataset_download_files(KAGGLE_DATASET, path=DOWNLOAD_PATH, unzip=True)

# Move and rename to normal class folder
if not os.path.exists(NORMAL_CLASS_DIR):
    os.makedirs(NORMAL_CLASS_DIR)

for person_dir in os.listdir(EXTRACTED_DIR):
    person_path = os.path.join(EXTRACTED_DIR, person_dir)
    for img_file in os.listdir(person_path):
        src = os.path.join(person_path, img_file)
        dst = os.path.join(NORMAL_CLASS_DIR, f"{person_dir}_{img_file}")
        os.rename(src, dst)
