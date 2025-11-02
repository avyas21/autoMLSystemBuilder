#!/usr/bin/env python3
"""Download Fashion-MNIST and convert to CSV format."""

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

print("Downloading Fashion-MNIST dataset...")

# Download training data
train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True
)

# Download test data
test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\nClasses:", class_names)

# Convert to CSV format (same as MNIST)
def dataset_to_csv(dataset, output_file):
    """Convert Fashion-MNIST dataset to CSV."""
    print(f"\nConverting to {output_file}...")

    data_list = []
    for img, label in dataset:
        # Convert PIL image to numpy array and flatten
        img_array = np.array(img).flatten()
        # Create row: [label, pixel1, pixel2, ..., pixel784]
        row = [label] + img_array.tolist()
        data_list.append(row)

    # Create column names: label, 1x1, 1x2, ..., 28x28
    columns = ['label'] + [f'{i//28+1}x{i%28+1}' for i in range(784)]

    # Create DataFrame and save
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {output_file} ({len(df)} samples)")

# Convert both datasets
dataset_to_csv(train_dataset, 'fashion_mnist_train.csv')
dataset_to_csv(test_dataset, 'fashion_mnist_test.csv')

print("\n✓ Fashion-MNIST datasets ready!")
print("\nDataset details:")
print("  - Format: CSV with 785 columns (1 label + 784 pixels)")
print("  - Image size: 28x28 grayscale")
print("  - Classes: 10 (clothing items)")
print("  - Training samples: 60,000")
print("  - Test samples: 10,000")
print("\nYou can now run:")
print("  python agent.py --train fashion_mnist_train.csv --val fashion_mnist_test.csv")
