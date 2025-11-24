#!/usr/bin/env python3
"""
Download Stanford Cars dataset and convert to CSV format.
Images are resized to 224x224 and flattened into CSV rows:
[label, pixel_0, pixel_1, ..., pixel_N]
"""

import os
import tarfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io

# -----------------------------
# URLs
# -----------------------------

CARS_TRAIN_URL = "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz"
CARS_TEST_URL  = "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz"
CARS_LABELS_URL = "http://ai.stanford.edu/~jkrause/car196/cars_annos.mat"

DATA_DIR = Path("./data")
CSV_DIR = Path("./datasets")
DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

train_tgz = DATA_DIR / "cars_train.tgz"
test_tgz  = DATA_DIR / "cars_test.tgz"
labels_mat = DATA_DIR / "cars_annos.mat"

# -----------------------------
# Download files
# -----------------------------

def download(url, path):
    if not path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        print("✓ Downloaded.")
    else:
        print(f"{path.name} already exists.")


download(CARS_TRAIN_URL, train_tgz)
download(CARS_TEST_URL, test_tgz)
download(CARS_LABELS_URL, labels_mat)

# -----------------------------
# Extract archives
# -----------------------------

def extract(tar_path, extract_to):
    if not extract_to.exists():
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)
        print("✓ Extracted to", extract_to)
    else:
        print(f"{extract_to.name} already extracted.")


extract(train_tgz, DATA_DIR / "cars_train")
extract(test_tgz,  DATA_DIR / "cars_test")

# -----------------------------
# Load annotations
# -----------------------------

print("Loading annotations...")
annos = scipy.io.loadmat(labels_mat)["annotations"][0]
print("✓ Annotations loaded.")

# Each annotation contains:
#   bbox_x1, bbox_y1, bbox_x2, bbox_y2, class, filename, test(0)/train(1)

# -----------------------------
# Convert datasets to CSV
# -----------------------------

def convert_to_csv(split, output_csv):
    print(f"\nProcessing {split} split...")

    rows = []
    target_size = (224, 224)

    for entry in annos:
        filename = entry[5][0]
        is_test = int(entry[6][0])
        label = int(entry[4][0]) - 1  # zero-based classes (196 classes)

        if split == "train" and is_test == 1:
            img_path = DATA_DIR / "cars_train" / filename
        elif split == "test" and is_test == 0:
            img_path = DATA_DIR / "cars_test" / filename
        else:
            continue

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


convert_to_csv("train", "stanford_cars_train.csv")
convert_to_csv("test",  "stanford_cars_test.csv")

print("\n✓ Stanford Cars dataset converted successfully!")
print("CSV files available in ./datasets/")
