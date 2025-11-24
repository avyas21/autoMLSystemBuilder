#!/usr/bin/env python3
"""Download EuroSAT (RGB) and convert to CSV format."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from torchvision import datasets, transforms
    import torch
except ImportError:
    print("Installing torchvision...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision", "scikit-learn"])
    from torchvision import datasets, transforms
    import torch

print("Downloading EuroSAT dataset...")

# EuroSAT RGB images are already 64×64 but convert to Tensor
transform = transforms.Compose([
    transforms.ToTensor()  # CHW format
])

# Load full dataset
dataset = datasets.EuroSAT(
    root="./data",
    download=True,
    transform=transform,
    bands="rgb"   # Only RGB 3-channel version
)

print(f"Total samples: {len(dataset)}")
print(f"Number of classes: {len(dataset.classes)}")
print("Classes:", dataset.classes)

# 80/20 train/val split
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")

# Extract subset of dataset
def extract_subset(dataset, indices):
    pixels, labels = [], []

    for idx in indices:
        img_tensor, label = dataset[idx]
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # CHW → HWC
        pixels.append(img_np.flatten())
        labels.append(label)

    return pixels, labels

# Save CSV
def save_csv(images, labels, output_file):
    print(f"\nConverting to {output_file}...")

    rows = []
    for img, lbl in zip(images, labels):
        rows.append([lbl] + img.tolist())

    columns = ["label"] + [f"pixel_{i}" for i in range(64 * 64 * 3)]

    df = pd.DataFrame(rows, columns=columns)

    Path("datasets").mkdir(exist_ok=True)
    out_path = Path("datasets") / output_file
    df.to_csv(out_path, index=False)

    print(f"✓ Saved {out_path} ({len(df)} samples)")

# Extract & save splits
train_pixels, train_labels = extract_subset(dataset, train_idx)
val_pixels, val_labels = extract_subset(dataset, val_idx)

save_csv(train_pixels, train_labels, "eurosat_train.csv")
save_csv(val_pixels, val_labels, "eurosat_test.csv")

print("\n✓ EuroSAT CSV datasets ready!")
print("  - Image size: 64×64 RGB")
print("  - Pixels per image: 12,288")
print("  - Classes:", len(dataset.classes))
print("  - Train samples:", len(train_idx))
print("  - Test samples:", len(val_idx))

print("\nRun your agent with:")
print("  python agent.py --train datasets/eurosat_train.csv --val datasets/eurosat_test.csv")
