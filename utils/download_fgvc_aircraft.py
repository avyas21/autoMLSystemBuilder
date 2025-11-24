#!/usr/bin/env python3
"""
Download FGVC Aircraft dataset and convert to CSV format.
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

# -----------------------------------------
# URLs
# -----------------------------------------

AIRCRAFT_URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"

DATA_DIR = Path("./data")
CSV_DIR = Path("./datasets")
DATA_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)

tar_path = DATA_DIR / "fgvc-aircraft-2013b.tar.gz"

# -----------------------------------------
# Download dataset
# -----------------------------------------

if not tar_path.exists():
    print("Downloading FGVC Aircraft dataset...")
    urllib.request.urlretrieve(AIRCRAFT_URL, tar_path)
    print("✓ Download complete.")
else:
    print("Archive already exists, skipping download.")

# -----------------------------------------
# Extract dataset
# -----------------------------------------

EXTRACTED_DIR = DATA_DIR / "fgvc-aircraft-2013b"

if not EXTRACTED_DIR.exists():
    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    print("✓ Extraction complete.")
else:
    print("Dataset already extracted, skipping.")

# -----------------------------------------
# Load split files
# -----------------------------------------

def load_split_list(filename):
    path = EXTRACTED_DIR / "data" / filename
    with open(path) as f:
        return [line.strip() for line in f]

train_list = load_split_list("images_train.txt")
val_list   = load_split_list("images_val.txt")
test_list  = load_split_list("images_test.txt")

# Load class labels
labels_file = EXTRACTED_DIR / "data" / "images_variant_labels.txt"
label_map = {}

with open(labels_file) as f:
    for i, line in enumerate(f):
        img, label = line.strip().split(" ", 1)
        label_map[img] = label

# Convert labels to integer classes
unique_classes = sorted(list(set(label_map.values())))
class_to_idx = {c: i for i, c in enumerate(unique_classes)}

print(f"Found {len(unique_classes)} aircraft classes.")

# -----------------------------------------
# Convert to CSV
# -----------------------------------------

def convert_split_to_csv(split_name, image_list, output_csv):
    print(f"\nProcessing split: {split_name} ({len(image_list)} images)")

    rows = []
    target_size = (224, 224)

    for img_name in image_list:
        img_path = EXTRACTED_DIR / "data" / "images" / (img_name + ".jpg")
        label_str = label_map[img_name]
        label_idx = class_to_idx[label_str]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            print("Skipping unreadable:", img_path)
            continue

        img = img.resize(target_size)
        img_array = np.array(img).flatten()

        row = [label_idx] + img_array.tolist()
        rows.append(row)

    columns = ["label"] + [f"pixel_{i}" for i in range(224 * 224 * 3)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(CSV_DIR / output_csv, index=False)

    print(f"✓ Saved {output_csv} ({len(df)} samples)")


convert_split_to_csv("train", train_list, "fgvc_aircraft_train.csv")
convert_split_to_csv("val",   val_list,   "fgvc_aircraft_val.csv")
convert_split_to_csv("test",  test_list,  "fgvc_aircraft_test.csv")

print("\n✓ FGVC Aircraft dataset converted successfully!")
print("CSV files available in ./datasets/")
