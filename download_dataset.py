import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "data"
DATASET_SLUG = "chrisfilo/urbansound8k"

os.makedirs(DATA_DIR, exist_ok=True)
api = KaggleApi()
api.authenticate()
api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=True)
print("Download complete. Data is in", os.path.abspath(DATA_DIR))
