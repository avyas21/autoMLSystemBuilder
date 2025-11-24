#!/usr/bin/env python3
"""Download GTSRB and convert to CSV format."""

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

print("Downloading GTSRB dataset...")

# Resize all images to fixed 48x48 (common GTSRB working size)
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()  # CHW format
])

# Load training dataset
train_dataset = datasets.GTSRB(
    root="./data",
    split="train",
    download=True,
    transform=transform
)

# Load testing dataset
test_dataset = datasets.GTSRB(
    root="./data",
    split="test",
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# GTSRB has 43 traffic sign classes
num_classes = len(set([label for _, label in train_dataset]))
print(f"Number of classes: {num_classes}")

def dataset_to_csv(dataset, output_file):
    print(f"\nConverting to {output_file}...")

    rows = []
    for img_tensor, label in dataset:
        # Convert tensor to numpy HWC, then flatten
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        row = [label] + img_np.flatten().tolist()
        rows.append(row)

    columns = ["label"] + [f"pixel_{i}" for i in range(48 * 48 * 3)]

    df = pd.DataFrame(rows, columns=columns)

    Path("datasets").mkdir(exist_ok=True)
    out_path = Path("datasets") / output_file
    df.to_csv(out_path, index=False)

    print(f"✓ Saved {out_path} ({len(df)} samples)")

# Convert both splits
dataset_to_csv(train_dataset, "gtsrb_train.csv")
dataset_to_csv(test_dataset, "gtsrb_test.csv")

print("\n✓ GTSRB CSV datasets ready!")
print("  - Image size (resized): 48×48 RGB")
print("  - Pixels per image: 6,912")
print(f"  - Train samples: {len(train_dataset)}")
print(f"  - Test samples: {len(test_dataset)}")
print("  - Classes: 43 traffic signs\n")

print("Run your agent like:")
print("  python agent.py --train datasets/gtsrb_train.csv --val datasets/gtsrb_test.csv")
