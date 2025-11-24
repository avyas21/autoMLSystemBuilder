#!/usr/bin/env python3
"""Download STL-10 and convert to CSV format."""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from torchvision import datasets
    import torch
except ImportError:
    print("Installing torchvision...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    from torchvision import datasets
    import torch

print("Downloading STL-10 dataset...")

# Download training data
train_dataset = datasets.STL10(
    root='./data',
    split='train',
    download=True
)

# Download test data
test_dataset = datasets.STL10(
    root='./data',
    split='test',
    download=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# STL-10 has 10 classes
class_names = train_dataset.classes
print("\nClasses:", class_names)

def dataset_to_csv(dataset, output_file, has_labels=True):
    """Convert STL-10 dataset to CSV format."""
    print(f"\nConverting to {output_file}...")

    data_list = []

    for img, label in dataset:
        img_array = np.array(img).flatten()  # 96x96x3 -> 27648 values
        if has_labels:
            row = [label] + img_array.tolist()
        else:
            row = [-1] + img_array.tolist()   # unlabeled -> label = -1
        data_list.append(row)

    # Pixel columns
    columns = ['label'] + [f'pixel_{i}' for i in range(96 * 96 * 3)]

    df = pd.DataFrame(data_list, columns=columns)

    Path('datasets').mkdir(exist_ok=True)
    out_path = Path('datasets') / output_file
    df.to_csv(out_path, index=False)

    print(f"✓ Saved {out_path} ({len(df)} samples)")

# Convert labeled + unlabeled splits
dataset_to_csv(train_dataset, 'stl10_train.csv', has_labels=True)
dataset_to_csv(test_dataset,  'stl10_test.csv',  has_labels=True)

print("\n✓ STL-10 CSV datasets ready!")
print("\nDataset details:")
print("  - Image size: 96×96 RGB")
print("  - Pixels per image: 27,648")
print("  - Classes: 10")
print("  - Train samples: 5,000")
print("  - Test samples: 8,000")
print("  - Unlabeled samples: 100,000\n")

print("Run your agent like:")
print("  python agent.py --train datasets/stl10_train.csv --val datasets/stl10_test.csv")
