#!/usr/bin/env python3
"""Download CIFAR-100 and convert to CSV format."""

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

print("Downloading CIFAR-100 dataset...")

# Download training data
train_dataset = datasets.CIFAR100(
    root='./data',
    train=True,
    download=True
)

# Download test data
test_dataset = datasets.CIFAR100(
    root='./data',
    train=False,
    download=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# CIFAR-100 has 100 fine-grained classes
class_names = train_dataset.classes
print("\nNumber of classes:", len(class_names))
print("Sample classes:", class_names[:20], "...")

# Convert to CSV
def dataset_to_csv(dataset, output_file):
    """Convert CIFAR-100 dataset to CSV."""
    print(f"\nConverting to {output_file}...")

    data_list = []
    for img, label in dataset:
        # Convert PIL image to numpy array and flatten (32x32x3 = 3072 values)
        img_array = np.array(img).flatten()
        row = [label] + img_array.tolist()
        data_list.append(row)

    # Column names: label + pixel_0 ... pixel_3071
    columns = ['label'] + [f'pixel_{i}' for i in range(3072)]

    df = pd.DataFrame(data_list, columns=columns)

    Path('datasets').mkdir(exist_ok=True)
    output_path = Path('datasets') / output_file

    df.to_csv(output_path, index=False)
    print(f"✓ Saved {output_path} ({len(df)} samples)")

# Convert both splits
dataset_to_csv(train_dataset, 'cifar100_train.csv')
dataset_to_csv(test_dataset, 'cifar100_test.csv')

print("\n✓ CIFAR-100 datasets ready!")
print("\nDataset details:")
print("  - Format: CSV with 3073 columns (1 label + 3072 pixels)")
print("  - Image size: 32x32 RGB")
print("  - Classes: 100 fine labels")
print(f"  - Training samples: {len(train_dataset)}")
print(f"  - Test samples: {len(test_dataset)}")

print("\nYou can now run:")
print("  python agent.py --train datasets/cifar100_train.csv --val datasets/cifar100_test.csv")
