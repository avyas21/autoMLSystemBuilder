import os
from torchvision.datasets import Flowers102
from PIL import Image
from datasets import Dataset, Features, ClassLabel, Image as HFImage, concatenate_datasets

OUT_DIR = "flowers102_processed"
IMAGE_SIZE = (224, 224)
ROOT = "flowers102_raw"


def load_split(split):
    """
    Loads Flowers102 split into a HuggingFace Dataset WITHOUT loading all
    images as numpy arrays to avoid OOM.
    """
    tv_dataset = Flowers102(
        root=ROOT,
        split=split,
        download=True
    )

    print(f"Loading split: {split}")

    # Store PIL images (lazy loading in HF Dataset)
    images = []
    labels = []

    for img, label in tv_dataset:
        img = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        images.append(img)
        labels.append(int(label))

    # Use HF Image feature → avoids memory blowup
    features = Features({
        "image": HFImage(),
        "label": ClassLabel(num_classes=102)
    })

    return Dataset.from_dict({"image": images, "label": labels}, features=features)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = load_split("train")
    val_ds = load_split("val")
    test_ds = load_split("test")

    print("Merging train + val → train_combined")
    train_combined = concatenate_datasets([train_ds, val_ds])

    print("Saving training set...")
    train_combined.save_to_disk(f"{OUT_DIR}/train")

    print("Saving test set...")
    test_ds.save_to_disk(f"{OUT_DIR}/test")

    print("Done!")


if __name__ == "__main__":
    main()

