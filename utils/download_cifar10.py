#!/usr/bin/env python3
"""Download CIFAR-10 and convert to CSV format."""

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

print("Downloading CIFAR-10 dataset...")

# Download training data
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True
)

# Download test data
test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("\nClasses:", class_names)

# Convert to CSV format
def dataset_to_csv(dataset, output_file):
    """Convert CIFAR-10 dataset to CSV."""
    print(f"\nConverting to {output_file}...")

    data_list = []
    for img, label in dataset:
        # Convert PIL image to numpy array (32x32x3) and flatten
        img_array = np.array(img).flatten()
        # Create row: [label, pixel1, pixel2, ..., pixel3072]
        row = [label] + img_array.tolist()
        data_list.append(row)

    # Create column names: label, r0, g0, b0, r1, g1, b1, ...
    # CIFAR-10 has 32x32 RGB images = 3072 values
    columns = ['label'] + [f'pixel_{i}' for i in range(3072)]

    # Create DataFrame and save
    df = pd.DataFrame(data_list, columns=columns)

    # Create datasets directory if it doesn't exist
    Path('datasets').mkdir(exist_ok=True)
    output_path = Path('datasets') / output_file

    df.to_csv(output_path, index=False)
    print(f"✓ Saved {output_path} ({len(df)} samples)")

# Convert both datasets
dataset_to_csv(train_dataset, 'cifar10_train.csv')
dataset_to_csv(test_dataset, 'cifar10_test.csv')

print("\n✓ CIFAR-10 datasets ready!")
print("\nDataset details:")
print("  - Format: CSV with 3073 columns (1 label + 3072 pixels)")
print("  - Image size: 32x32 RGB color")
print("  - Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)")
print("  - Training samples: 50,000")
print("  - Test samples: 10,000")
print("\nYou can now run:")
print("  python agent.py --train datasets/cifar10_train.csv --val datasets/cifar10_test.csv")
