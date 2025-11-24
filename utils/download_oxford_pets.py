#!/usr/bin/env python3
"""
Download Oxford-IIIT Pet Dataset and convert to CSV format.
Images are resized to 224x224 and flattened into CSV rows:
[label, pixel_0, pixel_1, ..., pixel_N]
"""

import os
from pathlib import Path
import urllib.request
import tarfile
import numpy as np
import pandas as pd
from PIL import Image

try:
    from torchvision import datasets, transforms
except ImportError:
    print("Installing torchvision...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    from torchvision import datasets, transforms

DATA_DIR = Path("./data")
CSV_DIR = Path("./datasets")
DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

# -----------------------------
# Download Oxford Pet Dataset
# -----------------------------

URL_IMAGES = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
URL_ANNOTS = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

IMAGES_TAR = DATA_DIR / "images.tar.gz"
ANNOTS_TAR = DATA_DIR / "annotations.tar.gz"

def download(url, path):
    if not path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
        print("✓ Download complete.")
    else:
        print(f"{path.name} already exists, skipping download.")

download(URL_IMAGES, IMAGES_TAR)
download(URL_ANNOTS, ANNOTS_TAR)

# -----------------------------
# Extract archives
# -----------------------------

def extract(tar_path, extract_to):
    if not extract_to.exists():
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_to)
        print("✓ Extraction complete.")
    else:
        print(f"{extract_to.name} already exists, skipping extraction.")

extract(IMAGES_TAR, DATA_DIR)
extract(ANNOTS_TAR, DATA_DIR)

# -----------------------------
# Load split information
# -----------------------------

split_file = DATA_DIR / "annotations" / "trainval.txt"
test_file  = DATA_DIR / "annotations" / "test.txt"

def parse_split_file(path):
    img_list = []
    with open(path) as f:
        for line in f:
            img_name, class_id, *_ = line.strip().split()
            img_list.append((img_name, int(class_id) - 1))  # zero-based
    return img_list

train_list = parse_split_file(split_file)
test_list  = parse_split_file(test_file)

print(f"Train samples: {len(train_list)}")
print(f"Test samples: {len(test_list)}")
print("Number of classes: 37")

# -----------------------------
# Convert to CSV
# -----------------------------

def convert_to_csv(split_name, img_label_list, output_csv):
    print(f"\nProcessing {split_name} split...")

    rows = []
    target_size = (224, 224)

    for img_name, label in img_label_list:
        img_path = DATA_DIR / "images" / (img_name + ".jpg")

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
# Convert splits
# -----------------------------

convert_to_csv("train", train_list, "oxford_pets_train.csv")
convert_to_csv("test", test_list, "oxford_pets_test.csv")

print("\n✓ Oxford-IIIT Pet CSV datasets ready!")
print("CSV files available in ./datasets/")
print("  - Image size: 224×224 RGB")
print("  - Pixels per image: 150,528")
print("  - Classes: 37")
