from datasets import load_dataset
from PIL import Image
import numpy as np

OUT_DIR = "stanford_cars_processed"
IMAGE_SIZE = (224, 224)

def preprocess_batch(batch):
    images = []
    for img in batch["image"]:
        img = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        images.append(np.array(img))
    batch["image"] = images
    return batch

def main():
    for split in ["train", "test"]:
        print(f"Loading split: {split}")

        ds = load_dataset("tanganke/stanford_cars", split=split)

        # batch_size=100: fast, minimal overhead
        ds = ds.map(
            preprocess_batch,
            batched=True,
            batch_size=100,
            num_proc=1,       # ← IMPORTANT: num_proc > 1 slows image preprocessing
        )

        out = f"{OUT_DIR}/{split}"
        ds.save_to_disk(out)
        print(f"Saved {split} → {out}")

if __name__ == "__main__":
    main()

