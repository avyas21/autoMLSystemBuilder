#!/usr/bin/env python3
"""Download Caltech-256 and convert to CSV format."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from torchvision import datasets, transforms
    import torch
except ImportError:
    print("Installing torchvision...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision", "scikit-learn"])
    from torchvision import datasets, transforms
    import torch

print("Downloading Caltech-256 dataset...")

# Transform: resize to 224x224 & convert to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # gives tensor in [C, H, W]
])

# Load full dataset
dataset = datasets.Caltech256(
    root="./data",
    download=True,
    transform=transform
)

print(f"Total samples: {len(dataset)}")
print(f"Number of classes: {len(dataset.categories)}")
print(f"Sample classes: {dataset.categories[:20]} ...")

# Create train/val split (80/20)
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")

# Helper to extract flattened images + labels
def extract_subset(dataset, indices):
    imgs, labels = [], []
    for idx in indices:
        img_tensor, label = dataset[idx]
        img_np = img_tensor.numpy().transpose(1, 2, 0)  # CHW → HWC
        imgs.append(img_np.flatten())
        labels.append(label)
    return imgs, labels

# Save CSV
def save_csv(images, labels, output_file):
    print(f"\nConverting to {output_file}...")

    rows = []
    for img, lbl in zip(images, labels):
        rows.append([lbl] + img.tolist())

    # Pixel columns
    columns = ["label"] + [f"pixel_{i}" for i in range(224 * 224 * 3)]

    df = pd.DataFrame(rows, columns=columns)

    Path("datasets").mkdir(exist_ok=True)
    out_path = Path("datasets") / output_file
    df.to_csv(out_path, index=False)

    print(f"✓ Saved {out_path} ({len(df)} samples)")

# Extract & save
train_imgs, train_labels = extract_subset(dataset, train_idx)
val_imgs, val_labels = extract_subset(dataset, val_idx)

save_csv(train_imgs, train_labels, "caltech256_train.csv")
save_csv(val_imgs, val_labels, "caltech256_test.csv")

print("\n✓ Caltech-256 CSV datasets ready!")
print("  - Image size (resized): 224×224 RGB")
print("  - Pixels per image: 150,528")
print("  - Classes: 257 (256 categories + clutter)")
print("  - Train samples:", len(train_idx))
print("  - Test samples:", len(val_idx))

print("\nRun your agent like:")
print("  python agent.py --train datasets/caltech256_train.csv --val datasets/caltech256_test.csv")
