#!/usr/bin/env python3
"""Download CUB-200-2011 and convert to CSV format."""

import os
import tarfile
import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# -----------------------------
# Download & Extract
# -----------------------------

CUB_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
DATA_DIR = Path("./data")
EXTRACTED_DIR = DATA_DIR / "CUB_200_2011"
CSV_DIR = Path("./datasets")

DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

tgz_path = DATA_DIR / "CUB_200_2011.tgz"

if not tgz_path.exists():
    print("Downloading CUB-200-2011 dataset...")
    urllib.request.urlretrieve(CUB_URL, tgz_path)
    print("✓ Download complete.")
else:
    print("Archive already exists, skipping download.")

if not EXTRACTED_DIR.exists():
    print("Extracting dataset...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("✓ Extraction complete.")
else:
    print("Dataset already extracted, skipping.")

# -----------------------------
# Read metadata file
# -----------------------------

images_file = EXTRACTED_DIR / "images.txt"
labels_file = EXTRACTED_DIR / "image_class_labels.txt"
split_file  = EXTRACTED_DIR / "train_test_split.txt"

print("Loading metadata...")

# id -> path
image_paths = {}
with open(images_file) as f:
    for line in f:
        img_id, path = line.strip().split()
        image_paths[int(img_id)] = path

# id -> class index (1–200)
image_labels = {}
with open(labels_file) as f:
    for line in f:
        img_id, label = line.strip().split()
        image_labels[int(img_id)] = int(label) - 1  # zero-based

# id -> train(1)/test(0)
image_split = {}
with open(split_file) as f:
    for line in f:
        img_id, is_train = line.strip().split()
        image_split[int(img_id)] = int(is_train)

print("✓ Metadata loaded.")

# -----------------------------
# Convert to CSV
# -----------------------------

def convert_split_to_csv(split_name, output_csv):
    print(f"\nProcessing split: {split_name}")

    rows = []
    target_size = (224, 224)  # resize images

    for img_id in image_paths.keys():
        if (split_name == "train" and image_split[img_id] == 1) or \
           (split_name == "test"  and image_split[img_id] == 0):

            img_path = EXTRACTED_DIR / "images" / image_paths[img_id]
            label = image_labels[img_id]

            # Load image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                print("Skipping unreadable:", img_path)
                continue

            # Resize to fixed size
            img = img.resize(target_size)

            # Flatten
            img_array = np.array(img).flatten()

            row = [label] + img_array.tolist()
            rows.append(row)

    columns = ["label"] + [f"pixel_{i}" for i in range(224 * 224 * 3)]

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(CSV_DIR / output_csv, index=False)

    print(f"✓ Saved {output_csv} ({len(df)} samples)")


# Create train and test CSVs
convert_split_to_csv("train", "cub200_train.csv")
convert_split_to_csv("test",  "cub200_test.csv")

print("\n✓ CUB-200 dataset converted successfully!")
print("CSV files generated in ./datasets/")
