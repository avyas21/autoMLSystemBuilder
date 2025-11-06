#!/usr/bin/env python3
"""
Search and Build - End-to-End AutoML Pipeline

This script provides a complete end-to-end workflow:
1. User provides a natural language prompt
2. System searches for relevant datasets online
3. Downloads and converts the best dataset
4. Generates training code using the existing agent
5. Optionally runs training and evaluation

Usage:
    python search_and_build.py --prompt "I want to create an image classifier for dogs and cats"

    # With automatic training:
    python search_and_build.py --prompt "..." --train-model --epochs 5

    # Select specific dataset from search results:
    python search_and_build.py --prompt "..." --select-dataset 2
"""

import argparse
import os
import sys
from typing import Optional
import subprocess

# Import our custom modules
from dataset_search import search_datasets
from dataset_converter import download_and_convert_dataset
from agent import agent_main


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end AutoML: Search datasets, download, and build models from natural language prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - search and prepare dataset
  python search_and_build.py --prompt "I want to create an image classifier for dogs and cats"

  # Automatically train the generated model
  python search_and_build.py --prompt "..." --train-model

  # Select a specific dataset from search results
  python search_and_build.py --prompt "..." --select-dataset 2

  # Full pipeline with custom training parameters
  python search_and_build.py --prompt "..." --train-model --epochs 20 --batch-size 128
        """
    )

    # Required arguments
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help='Natural language description of ML task (e.g., "I want to create an image classifier for dogs and cats")'
    )

    # Search options
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of dataset search results to show (default: 5)"
    )

    parser.add_argument(
        "--select-dataset",
        type=int,
        default=1,
        help="Which dataset to use from search results (1-indexed, default: 1 = best match)"
    )

    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip dataset search and use existing dataset files"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated",
        help="Directory to save generated files (default: ./generated)"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./datasets",
        help="Directory to save downloaded datasets (default: ./datasets)"
    )

    # Training options
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Automatically train the generated model after code generation"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    # Evaluation options
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation harness on generated code"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🤖 AutoML System Builder - Search and Build Pipeline")
    print("=" * 80)

    # Step 1: Search for datasets (unless skipped)
    if not args.skip_search:
        print("\n" + "=" * 80)
        print("STEP 1: SEARCHING FOR DATASETS")
        print("=" * 80)

        requirements, datasets = search_datasets(args.prompt, max_results=args.max_results)

        if not datasets:
            print("\n❌ No datasets found. Please try a different prompt.")
            sys.exit(1)

        # Let user select dataset
        selected_idx = args.select_dataset - 1  # Convert to 0-indexed
        if selected_idx < 0 or selected_idx >= len(datasets):
            print(f"\n⚠️  Invalid dataset selection: {args.select_dataset}")
            print(f"   Valid range: 1-{len(datasets)}")
            selected_idx = 0
            print(f"   Using default: dataset 1")

        selected_dataset = datasets[selected_idx]

        print(f"\n✅ Selected dataset:")
        print(f"   {selected_dataset['title']}")
        print(f"   Source: {selected_dataset['source']}")
        print(f"   URL: {selected_dataset['url']}")

        # Step 2: Download and convert dataset
        print("\n" + "=" * 80)
        print("STEP 2: DOWNLOADING AND CONVERTING DATASET")
        print("=" * 80)

        try:
            train_csv, test_csv = download_and_convert_dataset(selected_dataset, args.dataset_dir)
        except Exception as e:
            print(f"\n❌ Failed to download/convert dataset: {e}")
            print(f"\n💡 Tip: Make sure you have the necessary credentials configured:")
            print(f"   - Kaggle: ~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api)")
            print(f"   - HuggingFace: Login with `huggingface-cli login`")
            sys.exit(1)

    else:
        print("\n⏭️  Skipping dataset search (--skip-search enabled)")
        print("   Please provide dataset paths manually\n")
        sys.exit(0)

    # Step 3: Generate training code
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING TRAINING CODE")
    print("=" * 80)

    print(f"\n🧠 Using LLM to generate PyTorch training code...")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Call the existing agent
        # Create args object for agent_main
        from argparse import Namespace
        agent_args = Namespace(
            train=train_csv,
            val=test_csv,
            output=os.path.join(args.output_dir, "model.py"),
            instructions=f"This is for the task: {args.prompt}",
            iterations=3
        )
        agent_main(agent_args)

        generated_model_path = os.path.join(args.output_dir, "model.py")
        generated_requirements_path = os.path.join(args.output_dir, "requirements.txt")

        print(f"\n✅ Code generation complete!")
        print(f"   Model: {generated_model_path}")
        print(f"   Requirements: {generated_requirements_path}")

    except Exception as e:
        print(f"\n❌ Code generation failed: {e}")
        sys.exit(1)

    # Step 4: Run evaluation (if requested)
    if args.evaluate:
        print("\n" + "=" * 80)
        print("STEP 4: EVALUATING GENERATED CODE")
        print("=" * 80)

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "evaluate_agent.py",
                    "--train", train_csv,
                    "--val", test_csv,
                    "--output", os.path.join(args.output_dir, "evaluation_results.json")
                ],
                check=True,
                capture_output=False
            )

            print(f"\n✅ Evaluation complete!")

        except subprocess.CalledProcessError as e:
            print(f"\n⚠️  Evaluation failed with exit code {e.returncode}")

    # Step 5: Train model (if requested)
    if args.train_model:
        print("\n" + "=" * 80)
        print("STEP 5: TRAINING MODEL")
        print("=" * 80)

        print(f"\n🏋️  Training with parameters:")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr}")

        try:
            # First install requirements
            print(f"\n📦 Installing requirements...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", generated_requirements_path],
                check=True,
                capture_output=False
            )

            # Run training
            print(f"\n🚀 Starting training...")
            result = subprocess.run(
                [
                    sys.executable,
                    generated_model_path,
                    "--train", train_csv,
                    "--val", test_csv,
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--lr", str(args.lr)
                ],
                check=True,
                capture_output=False,
                cwd=args.output_dir
            )

            print(f"\n✅ Training complete!")
            print(f"   Model checkpoint saved: {os.path.join(args.output_dir, 'best_model.pth')}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with exit code {e.returncode}")
            sys.exit(1)

    # Final summary
    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 80)

    print(f"\n📁 Generated files:")
    print(f"   Dataset (train): {train_csv}")
    print(f"   Dataset (test): {test_csv}")
    print(f"   Model code: {generated_model_path}")
    print(f"   Requirements: {generated_requirements_path}")

    if args.train_model:
        print(f"   Trained model: {os.path.join(args.output_dir, 'best_model.pth')}")

    print(f"\n💡 Next steps:")
    if not args.train_model:
        print(f"   1. Install dependencies: pip install -r {generated_requirements_path}")
        print(f"   2. Train the model:")
        print(f"      python {generated_model_path} --train {train_csv} --val {test_csv}")
    else:
        print(f"   1. Use your trained model for inference")
        print(f"   2. Fine-tune hyperparameters if needed")

    if not args.evaluate:
        print(f"   3. Run evaluation: python evaluate_agent.py --train {train_csv} --val {test_csv}")

    print(f"\n✨ Happy ML building!")


if __name__ == "__main__":
    main()
