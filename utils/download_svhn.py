#!/usr/bin/env python3
"""Download SVHN and convert to CSV format."""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from torchvision import datasets, transforms
    import torch
except ImportError:
    print("Installing torchvision...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    from torchvision import datasets, transforms
    import torch

print("Downloading SVHN dataset...")

# SVHN images are already 32x32, but we convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()  # CHW format
])

# Load datasets
train_dataset = datasets.SVHN(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

test_dataset = datasets.SVHN(
    root="./data",
    split="test",
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

def dataset_to_csv(dataset, output_file):
    print(f"\nConverting to {output_file}...")

    rows = []
    for img_tensor, label in dataset:
        # Convert CHW tensor → HWC numpy array → flatten
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        row = [label] + img_np.flatten().tolist()
        rows.append(row)

    columns = ["label"] + [f"pixel_{i}" for i in range(32 * 32 * 3)]

    df = pd.DataFrame(rows, columns=columns)

    Path("datasets").mkdir(exist_ok=True)
    out_path = Path("datasets") / output_file
    df.to_csv(out_path, index=False)

    print(f"✓ Saved {out_path} ({len(df)} samples)")

# Save train and test to CSV
dataset_to_csv(train_dataset, "svhn_train.csv")
dataset_to_csv(test_dataset,  "svhn_test.csv")

print("\n✓ SVHN CSV datasets ready!")
print("  - Image size: 32×32 RGB")
print("  - Pixels per image: 3,072")
print(f"  - Train samples: {len(train_dataset)}")
print(f"  - Test samples: {len(test_dataset)}")
print("  - Classes: digits 1–10 (10 = digit '0')\n")

print("Run your agent like:")
print("  python agent.py --train datasets/svhn_train.csv --val datasets/svhn_test.csv")
