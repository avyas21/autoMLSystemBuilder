#!/usr/bin/env python3
"""
Download Oxford 102 Flowers Dataset and convert to CSV format.
Images are resized to 224x224 and flattened into CSV rows:
[label, pixel_0, pixel_1, ..., pixel_N]

This version only creates train and test CSVs (train includes original train + val).
"""

import os
from pathlib import Path
import urllib.request
import tarfile
import scipy.io
import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path("./data")
CSV_DIR = Path("./datasets")
DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

# -----------------------------
# URLs
# -----------------------------
IMAGES_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
SPLIT_URL  = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

IMAGES_TAR = DATA_DIR / "102flowers.tgz"
LABELS_MAT = DATA_DIR / "imagelabels.mat"
SPLIT_MAT  = DATA_DIR / "setid.mat"

# -----------------------------
# Download files
# -----------------------------
def download(url, path):
    if not path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"✓ Downloaded {path.name}")
    else:
        print(f"{path.name} already exists.")

download(IMAGES_URL, IMAGES_TAR)
download(LABELS_URL, LABELS_MAT)
download(SPLIT_URL, SPLIT_MAT)

# -----------------------------
# Extract images
# -----------------------------
EXTRACTED_DIR = DATA_DIR / "jpg"

if not EXTRACTED_DIR.exists():
    print("Extracting images...")
    with tarfile.open(IMAGES_TAR, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("✓ Extraction complete.")
else:
    print("Images already extracted.")

# -----------------------------
# Load labels and splits
# -----------------------------
labels_mat = scipy.io.loadmat(LABELS_MAT)
image_labels = labels_mat['labels'][0] - 1  # zero-based

split_mat = scipy.io.loadmat(SPLIT_MAT)
train_idx = split_mat['trnid'][0] - 1
val_idx   = split_mat['valid'][0] - 1
test_idx  = split_mat['tstid'][0] - 1

# Merge train + val for single training CSV
train_idx = np.concatenate([train_idx, val_idx])

print(f"Total images: {len(image_labels)}")
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
print("Classes: 102")

# -----------------------------
# Helper to convert list of indices to CSV
# -----------------------------
def convert_to_csv(split_name, indices, output_csv):
    print(f"\nProcessing {split_name} split...")

    rows = []
    target_size = (224, 224)

    for idx in indices:
        img_name = f"image_{idx+1:05d}.jpg"
        img_path = EXTRACTED_DIR / img_name
        label = int(image_labels[idx])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print("Skipping unreadable:", img_path)
            continue

        img = img.resize(target_size)
        img_array = np.array(img).flatten()
        row = [label] + img_array.tolist()
        rows.append(row)

    columns = ["label"] + [f"pixel_{i}" for i in range(224 * 224 * 3)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(CSV_DIR / output_csv, index=False)
    print(f"✓ Saved {output_csv} ({len(df)} samples)")

# -----------------------------
# Convert train and test splits
# -----------------------------
convert_to_csv("train", train_idx, "flowers102_train.csv")
convert_to_csv("test", test_idx,  "flowers102_test.csv")

print("\n✓ Flowers-102 train/test CSV datasets ready!")
print("CSV files available in ./datasets/")
print("  - Image size: 224×224 RGB")
print("  - Pixels per image: 150,528")
print("  - Classes: 102")
