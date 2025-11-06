"""
Dataset Download and Conversion Module

This module handles:
- Downloading datasets from Kaggle and HuggingFace
- Converting various formats (image folders, parquet, etc.) to CSV
- Creating train/test splits
- Normalizing data to match the expected format
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image


def download_kaggle_dataset(dataset_ref: str, download_dir: str) -> str:
    """
    Download a dataset from Kaggle.

    Args:
        dataset_ref: Kaggle dataset reference (e.g., "username/dataset-name")
        download_dir: Directory to download the dataset to

    Returns:
        Path to the downloaded dataset directory
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print(f"📥 Downloading Kaggle dataset: {dataset_ref}")

        # Create download directory
        os.makedirs(download_dir, exist_ok=True)

        # Download and unzip dataset
        api.dataset_download_files(dataset_ref, path=download_dir, unzip=True)

        print(f"✅ Downloaded to: {download_dir}")
        return download_dir

    except ImportError:
        raise ImportError("Kaggle API not installed. Install with: pip install kaggle")
    except Exception as e:
        raise Exception(f"Failed to download Kaggle dataset: {e}")


def download_huggingface_dataset(dataset_name: str, download_dir: str, split: Optional[str] = None) -> str:
    """
    Download a dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "username/dataset-name")
        download_dir: Directory to download the dataset to
        split: Optional split to download ("train", "test", etc.)

    Returns:
        Path to the downloaded dataset directory
    """
    try:
        from datasets import load_dataset

        print(f"📥 Downloading HuggingFace dataset: {dataset_name}")

        # Create download directory
        os.makedirs(download_dir, exist_ok=True)

        # Load dataset
        if split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name)

        # Save to directory
        if hasattr(dataset, 'save_to_disk'):
            dataset.save_to_disk(download_dir)
        else:
            # If dataset is a DatasetDict, save each split
            for split_name, split_data in dataset.items():
                split_path = os.path.join(download_dir, split_name)
                split_data.save_to_disk(split_path)

        print(f"✅ Downloaded to: {download_dir}")
        return download_dir

    except ImportError:
        raise ImportError("HuggingFace datasets not installed. Install with: pip install datasets")
    except Exception as e:
        raise Exception(f"Failed to download HuggingFace dataset: {e}")


def convert_image_folder_to_csv(image_dir: str, output_csv: str, img_size: Tuple[int, int] = (32, 32)) -> None:
    """
    Convert an image folder dataset to CSV format.

    Expected folder structure:
        image_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg

    Args:
        image_dir: Path to image directory
        output_csv: Path to output CSV file
        img_size: Target image size (width, height)
    """
    print(f"🔄 Converting image folder to CSV: {image_dir}")

    data = []
    class_names = []

    # Find all subdirectories (classes)
    subdirs = [d for d in sorted(os.listdir(image_dir)) if os.path.isdir(os.path.join(image_dir, d))]

    if not subdirs:
        raise Exception(f"No subdirectories found in {image_dir}. Expected ImageFolder structure with class subdirectories.")

    for class_idx, class_name in enumerate(subdirs):
        class_path = os.path.join(image_dir, class_name)
        class_names.append(class_name)

        # Process all images in this class
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        if not image_files:
            print(f"  ⚠️  Class {class_idx} '{class_name}': No images found, skipping")
            continue

        print(f"  Processing class {class_idx} '{class_name}': {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)

            try:
                # Load and resize image
                img = Image.open(img_path)

                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize
                img = img.resize(img_size)

                # Convert to numpy array and flatten
                img_array = np.array(img).flatten()

                # Normalize to [0, 1]
                img_array = img_array / 255.0

                # Append [label, pixel1, pixel2, ...]
                row = [class_idx] + img_array.tolist()
                data.append(row)

            except Exception as e:
                print(f"    Warning: Failed to process {img_file}: {e}")
                continue

    # Validate we have data
    if not data:
        raise Exception(f"No images were successfully converted from {image_dir}. Check that subdirectories contain valid image files.")

    # Create DataFrame
    num_pixels = img_size[0] * img_size[1] * 3  # RGB
    columns = ['label'] + [f'pixel_{i}' for i in range(num_pixels)]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"✅ Saved {len(df)} images to {output_csv}")
    print(f"   Classes ({len(class_names)}): {', '.join(class_names[:10])}{'...' if len(class_names) > 10 else ''}")
    print(f"   Image size: {img_size}")


def convert_hf_dataset_to_csv(hf_data_dir: str, output_dir: str) -> Tuple[str, str]:
    """
    Convert HuggingFace dataset to CSV format.

    Args:
        hf_data_dir: Path to HuggingFace dataset directory
        output_dir: Directory to save CSV files

    Returns:
        Tuple of (train_csv_path, test_csv_path)
    """
    try:
        from datasets import load_from_disk, Dataset

        print(f"🔄 Converting HuggingFace dataset to CSV")

        os.makedirs(output_dir, exist_ok=True)

        # Check if it's a DatasetDict or single Dataset
        try:
            dataset = load_from_disk(hf_data_dir)
        except:
            # Try loading individual splits
            train_path = os.path.join(hf_data_dir, 'train')
            test_path = os.path.join(hf_data_dir, 'test')

            train_dataset = load_from_disk(train_path) if os.path.exists(train_path) else None
            test_dataset = load_from_disk(test_path) if os.path.exists(test_path) else None

            if train_dataset is None:
                raise Exception("Could not find train split")

            # Save to CSV
            train_csv = os.path.join(output_dir, 'train.csv')
            test_csv = os.path.join(output_dir, 'test.csv')

            # Convert to pandas and save
            train_df = train_dataset.to_pandas()
            train_df.to_csv(train_csv, index=False)

            if test_dataset:
                test_df = test_dataset.to_pandas()
                test_df.to_csv(test_csv, index=False)
            else:
                # Create test split from train (80/20 split)
                split_idx = int(len(train_df) * 0.8)
                test_df = train_df[split_idx:]
                train_df = train_df[:split_idx]

                train_df.to_csv(train_csv, index=False)
                test_df.to_csv(test_csv, index=False)

            print(f"✅ Converted HuggingFace dataset to CSV")
            print(f"   Train: {train_csv} ({len(train_df)} samples)")
            print(f"   Test: {test_csv} ({len(test_df)} samples)")

            return train_csv, test_csv

    except ImportError:
        raise ImportError("HuggingFace datasets not installed. Install with: pip install datasets")


def auto_detect_and_convert(dataset_dir: str, output_dir: str) -> Tuple[str, str]:
    """
    Automatically detect dataset format and convert to CSV.

    Args:
        dataset_dir: Path to downloaded dataset
        output_dir: Directory to save CSV files

    Returns:
        Tuple of (train_csv_path, test_csv_path)
    """
    print(f"\n🔍 Auto-detecting dataset format in: {dataset_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Debug: Show what's in the directory
    try:
        contents = list(Path(dataset_dir).iterdir())
        print(f"   Directory contents: {[c.name for c in contents[:10]]}")
    except:
        pass

    # Check if it's already CSV
    csv_files = list(Path(dataset_dir).rglob('*.csv'))
    if csv_files:
        print(f"   Found {len(csv_files)} CSV files")

        # Use LLM to intelligently analyze CSV structure
        from dataset_inspector import llm_inspect_dataset_csvs, merge_feature_label_csvs
        import shutil

        analysis = llm_inspect_dataset_csvs(csv_files, dataset_dir)

        # Handle based on LLM analysis
        # If format is split, we ALWAYS need to merge (override LLM if it said otherwise)
        if analysis['format'] == 'split_features_labels':
            print(f"   📊 Detected split features/labels format - merging required")

            train_csv_dest = os.path.join(output_dir, 'train.csv')
            test_csv_dest = os.path.join(output_dir, 'test.csv')

            # Find full paths for the files
            train_features = next((f for f in csv_files if f.name == analysis['train_features_file']), None)
            train_labels = next((f for f in csv_files if f.name == analysis['train_labels_file']), None)
            test_features = next((f for f in csv_files if f.name == analysis['test_features_file']), None)
            test_labels = next((f for f in csv_files if f.name == analysis['test_labels_file']), None)

            if train_features and train_labels:
                merge_feature_label_csvs(str(train_features), str(train_labels), train_csv_dest)
            else:
                raise Exception(f"Could not find train files: {analysis['train_features_file']}, {analysis['train_labels_file']}")

            if test_features and test_labels:
                merge_feature_label_csvs(str(test_features), str(test_labels), test_csv_dest)
            else:
                raise Exception(f"Could not find test files: {analysis['test_features_file']}, {analysis['test_labels_file']}")

            return train_csv_dest, test_csv_dest

        elif analysis['format'] == 'combined':
            print(f"   📊 Detected combined features+labels format")

            train_csv_src = next((f for f in csv_files if f.name == analysis['train_combined_file']), None)
            test_csv_src = next((f for f in csv_files if f.name == analysis['test_combined_file']), None)

            if train_csv_src and test_csv_src:
                train_csv_dest = os.path.join(output_dir, 'train.csv')
                test_csv_dest = os.path.join(output_dir, 'test.csv')

                print(f"   Copying CSV files to {output_dir}")
                shutil.copy2(str(train_csv_src), train_csv_dest)
                shutil.copy2(str(test_csv_src), test_csv_dest)

                return train_csv_dest, test_csv_dest

        # Fallback: try simple pattern matching
        print(f"   ⚠️  LLM analysis inconclusive (confidence: {analysis.get('confidence', 0)}), trying simple matching...")
        train_csv_src = None
        test_csv_src = None

        for csv_file in csv_files:
            filename = csv_file.name.lower()
            if 'train' in filename and 'csv' in filename:
                train_csv_src = str(csv_file)
            elif ('test' in filename or 'val' in filename) and 'csv' in filename:
                test_csv_src = str(csv_file)

        if train_csv_src and test_csv_src:
            train_csv_dest = os.path.join(output_dir, 'train.csv')
            test_csv_dest = os.path.join(output_dir, 'test.csv')

            shutil.copy2(train_csv_src, train_csv_dest)
            shutil.copy2(test_csv_src, test_csv_dest)

            return train_csv_dest, test_csv_dest

    # Check if it's an image folder dataset
    # Look for images recursively
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(Path(dataset_dir).rglob(ext)))

    if all_images:
        print(f"   Found {len(all_images)} images")

        # Find the best directory structure for ImageFolder format
        # Try to find a directory with class subdirectories containing images
        def find_class_structure(base_dir):
            """Find directory that has subdirectories with images (ImageFolder structure)"""
            for potential_root in [base_dir] + list(Path(base_dir).rglob('*')):
                if not Path(potential_root).is_dir():
                    continue

                subdirs = [d for d in Path(potential_root).iterdir() if d.is_dir()]
                if not subdirs:
                    continue

                # Check if these subdirectories contain images directly (not recursively)
                class_dirs_with_images = []
                for subdir in subdirs:
                    direct_images = [f for f in subdir.iterdir()
                                   if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
                    if direct_images:
                        class_dirs_with_images.append(subdir)

                # If we found multiple class directories with images, this is likely the right structure
                if len(class_dirs_with_images) >= 2:
                    return str(potential_root), class_dirs_with_images

            return None, []

        # Look for train/test splits first
        train_dir = None
        test_dir = None

        subdirs = [d for d in Path(dataset_dir).iterdir() if d.is_dir()]
        for subdir in subdirs:
            name = subdir.name.lower()
            if 'train' in name:
                train_dir = str(subdir)
            elif 'test' in name or 'val' in name:
                test_dir = str(subdir)

        # If no train dir found, search for ImageFolder structure
        if not train_dir:
            root_dir, class_dirs = find_class_structure(dataset_dir)
            if root_dir:
                print(f"   Found ImageFolder structure at: {Path(root_dir).name}")
                print(f"   Detected {len(class_dirs)} classes: {', '.join([d.name for d in class_dirs[:5]])}{'...' if len(class_dirs) > 5 else ''}")
                train_dir = root_dir
            else:
                # Last resort: use the dataset root if it has subdirectories
                train_dir = dataset_dir

        # Convert to CSV
        train_csv = os.path.join(output_dir, 'train.csv')
        test_csv = os.path.join(output_dir, 'test.csv')

        convert_image_folder_to_csv(train_dir, train_csv)

        if test_dir:
            convert_image_folder_to_csv(test_dir, test_csv)
        else:
            # Create test split from train (80/20)
            print("   No test split found, creating 80/20 split")
            df = pd.read_csv(train_csv)

            # Validate we have data
            if len(df) == 0:
                raise Exception(f"No data was converted from {train_dir}. Check dataset structure.")

            split_idx = int(len(df) * 0.8)
            train_df = df[:split_idx]
            test_df = df[split_idx:]

            train_df.to_csv(train_csv, index=False)
            test_df.to_csv(test_csv, index=False)

        return train_csv, test_csv

    # Check if it's a HuggingFace dataset
    try:
        return convert_hf_dataset_to_csv(dataset_dir, output_dir)
    except:
        pass

    raise Exception("Could not detect dataset format. Supported formats: CSV, image folders, HuggingFace datasets")


def download_and_convert_dataset(dataset_info: Dict, output_dir: str) -> Tuple[str, str]:
    """
    Download and convert a dataset to CSV format.

    Args:
        dataset_info: Dataset metadata dictionary (from dataset_search.py)
        output_dir: Directory to save output files

    Returns:
        Tuple of (train_csv_path, test_csv_path)
    """
    source = dataset_info['source']
    dataset_name = dataset_info['name']

    # Create temporary download directory
    temp_dir = tempfile.mkdtemp(prefix='automl_dataset_')

    try:
        # Download dataset
        if source == 'kaggle':
            download_dir = download_kaggle_dataset(dataset_name, temp_dir)
        elif source == 'huggingface':
            download_dir = download_huggingface_dataset(dataset_name, temp_dir)
        else:
            raise ValueError(f"Unsupported source: {source}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Convert to CSV (output_dir will persist after temp cleanup)
        train_csv, test_csv = auto_detect_and_convert(download_dir, output_dir)

        # Validate CSV files have data
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        if len(train_df) == 0:
            raise Exception(f"Training CSV is empty. Dataset conversion may have failed.")
        if len(test_df) == 0:
            raise Exception(f"Test CSV is empty. Dataset conversion may have failed.")

        print(f"\n✅ Dataset ready!")
        print(f"   Train: {train_csv} ({len(train_df)} samples)")
        print(f"   Test: {test_csv} ({len(test_df)} samples)")

        return train_csv, test_csv

    finally:
        # Clean up temporary directory (CSV files are already in output_dir)
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"   Warning: Could not clean up temp directory: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and convert datasets")
    parser.add_argument("--source", type=str, required=True, choices=["kaggle", "huggingface"],
                        help="Dataset source")
    parser.add_argument("--name", type=str, required=True, help="Dataset name/reference")
    parser.add_argument("--output-dir", type=str, default="./datasets", help="Output directory")

    args = parser.parse_args()

    dataset_info = {
        "source": args.source,
        "name": args.name
    }

    train_csv, test_csv = download_and_convert_dataset(dataset_info, args.output_dir)

    print(f"\n🎉 Done! You can now use these files with the agent:")
    print(f"   python agent.py --train {train_csv} --val {test_csv}")
