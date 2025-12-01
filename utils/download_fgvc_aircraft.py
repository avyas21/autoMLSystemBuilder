import os
from torchvision.datasets import FGVCAircraft
from torchvision import transforms
from PIL import Image
import numpy as np
from datasets import Dataset, concatenate_datasets

OUT_DIR = "fgvc_aircraft_processed"
IMAGE_SIZE = (224, 224)
ROOT = "fgvc_aircraft_raw"   # where torchvision stores original dataset

# Resize + convert transform
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=Image.Resampling.LANCZOS),
    transforms.ConvertImageDtype(torch.float32) if False else None,  # kept for clarity
])

def load_split(split):
    """
    Loads FGVCAircraft split using torchvision and returns a HuggingFace Dataset.
    """
    tv_dataset = FGVCAircraft(
        root=ROOT,
        split=split,
        annotation_level="variant",  # same labels as official FGVC benchmark
        download=True
    )

    images = []
    labels = []

    print(f"Loading split: {split}")

    for img, label in tv_dataset:
        img = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        images.append(np.array(img))
        labels.append(int(label))

    # Convert to HF dataset
    return Dataset.from_dict({
        "image": images,
        "label": labels,
    })

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load splits
    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")

    # Merge train + val
    print("Merging train + val → train_combined")
    train_combined = concatenate_datasets([train_ds, val_ds])

    # Save as HF arrow datasets
    print("Saving training set...")
    train_combined.save_to_disk(f"{OUT_DIR}/train")

    print("Saving test set...")
    test_ds.save_to_disk(f"{OUT_DIR}/test")

    print("Done!")

if __name__ == "__main__":
    main()

