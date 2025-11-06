# AutoML System Builder

An AI-powered agent that automatically generates PyTorch training code for machine learning datasets.

## Overview

This system uses LLMs (via LangGraph + OpenAI) to:
1. **Search for datasets online** based on natural language prompts (NEW!)
2. **Download and convert datasets** from Kaggle and HuggingFace (NEW!)
3. Inspect datasets and infer properties
4. Generate complete PyTorch training code (`model.py`)
5. Generate dependency specifications (`requirements.txt`)
6. Automatically install dependencies
7. Evaluate code quality and performance

## ✨ NEW: Search and Build Pipeline

You can now go from a natural language prompt to a trained model with a single command!

```bash
python search_and_build.py --prompt "I want to create an image classifier for dogs and cats" --train-model
```

This will:
1. Parse your prompt using LLM
2. Search Kaggle and HuggingFace for relevant datasets
3. Download and convert the best matching dataset
4. Generate PyTorch training code
5. Train the model automatically

## Project Structure

```
autoMLSystemBuilder/
├── agent.py                    # Core agent (generates ML code)
├── evaluate_agent.py           # Evaluation harness
├── search_and_build.py         # NEW: End-to-end pipeline
├── dataset_search.py           # NEW: Dataset search module
├── dataset_converter.py        # NEW: Dataset download & conversion
├── datasets/                   # Training datasets (gitignored)
│   ├── mnist_train.csv
│   ├── mnist_test.csv
│   ├── fashion_mnist_train.csv
│   └── fashion_mnist_test.csv
├── utils/                      # Utility scripts
│   ├── download_fashion_mnist.py
│   ├── download_cifar10.py
│   └── compare_results.py
├── examples/                   # Example evaluation results
│   └── evaluation_results/
│       ├── mnist.json
│       ├── fashion_mnist.json
│       └── cifar10.json
└── generated/                  # Generated code (gitignored)
    ├── model.py
    ├── requirements.txt
    └── best_model.pth
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**

   **OpenAI (Required):**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

   **Kaggle (Optional - for dataset search):**
   ```bash
   # Download API credentials from https://www.kaggle.com/account
   # Place kaggle.json in ~/.kaggle/
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

   **HuggingFace (Optional - for dataset search):**
   ```bash
   # Login to HuggingFace
   huggingface-cli login
   # Or set token in environment
   echo "HUGGINGFACE_TOKEN=your-token-here" >> .env
   ```

3. **Optional: Download example datasets manually:**
   ```bash
   # Fashion-MNIST
   python utils/download_fashion_mnist.py

   # CIFAR-10
   python utils/download_cifar10.py

   # MNIST (via git-lfs)
   git lfs pull
   ```

## Usage

### 🚀 Quick Start: Natural Language to Model

The easiest way to build a model is with a natural language prompt:

```bash
# Search, download, and generate code
python search_and_build.py --prompt "I want to create an image classifier for dogs and cats"

# Full pipeline with automatic training
python search_and_build.py \
  --prompt "Build a model to classify handwritten digits" \
  --train-model \
  --epochs 10

# Select a specific dataset from search results
python search_and_build.py \
  --prompt "Image classification for flowers" \
  --select-dataset 2 \
  --train-model
```

### 📦 Traditional Workflow (with existing datasets)

If you already have datasets in CSV format:

**Generate and evaluate code:**
```bash
python evaluate_agent.py \
  --train datasets/mnist_train.csv \
  --val datasets/mnist_test.csv \
  --run-agent \
  --output results.json
```

**Step-by-step:**

1. **Generate code only:**
   ```bash
   python agent.py \
     --train datasets/mnist_train.csv \
     --val datasets/mnist_test.csv \
     --output generated/model.py
   ```

2. **Evaluate existing code:**
   ```bash
   python evaluate_agent.py \
     --train datasets/mnist_train.csv \
     --val datasets/mnist_test.csv \
     --model generated/model.py
   ```

3. **Compare results across datasets:**
   ```bash
   python utils/compare_results.py
   ```

### 🔍 Dataset Search Only

You can also use the search functionality independently:

```bash
# Search for datasets
python dataset_search.py --prompt "dog cat classification" --max-results 10

# Download a specific dataset
python dataset_converter.py \
  --source kaggle \
  --name username/dataset-name \
  --output-dir ./datasets
```

## Evaluation Framework

The evaluation harness tests:

### Phase 1: Functional Correctness
- ✓ Syntax validity
- ✓ Import availability
- ✓ Required components (model, train, validate, argparse)
- ✓ Code execution
- ✓ CLI interface

### Phase 2: Performance
- ✓ Training runs successfully
- ✓ Model checkpoint created
- ✓ Checkpoint loadable
- ✓ Metrics tracked

### Phase 3: Dependency Management
- ✓ requirements.txt generated
- ✓ Dependencies auto-installed
- ✓ All imports successful

## Results

### Current Performance

| Dataset | Status | Accuracy | Time | Difficulty |
|---------|--------|----------|------|------------|
| MNIST | ✅ PASS | 96.76% | 25.0s | Easy (>95%) |
| Fashion-MNIST | ✅ PASS | 85.01% | 27.4s | Medium (>85%) |

### Generated Code Quality

✅ Correct argparse interface (`--train`, `--val`, `--epochs`, `--batch-size`, `--lr`)
✅ GPU/CPU device handling
✅ Proper training loop with epoch-level metrics
✅ Average loss tracking (not just last batch)
✅ Checkpoint saving with messages
✅ Docstrings and comments
✅ Clean, readable code structure

## 🔍 Dataset Search Features

### How it Works

The dataset search pipeline uses multiple components:

1. **Prompt Parser** (`dataset_search.py`):
   - Uses GPT-4o-mini to parse natural language prompts
   - Extracts task type (classification/regression)
   - Identifies data type (image/text/tabular)
   - Generates optimized search keywords

2. **Multi-Source Search**:
   - **Kaggle API**: Searches 1000+ public datasets
   - **HuggingFace Hub**: Searches ML-specific datasets
   - Ranks results by relevance and popularity

3. **Auto-Detection & Conversion** (`dataset_converter.py`):
   - Detects dataset format automatically
   - Supports:
     - CSV files (direct use)
     - Image folders (class-based structure)
     - HuggingFace datasets (Arrow/Parquet)
   - Converts to standardized CSV format
   - Handles train/test splitting

### Supported Dataset Sources

| Source | API Required | Formats Supported | Example |
|--------|--------------|-------------------|---------|
| Kaggle | Yes (free) | CSV, images, zip archives | `python search_and_build.py --prompt "dog breeds"` |
| HuggingFace | Optional | Datasets library format | `python search_and_build.py --prompt "sentiment analysis"` |

### Dataset Format Requirements

The system expects CSV format:
```
label,pixel_0,pixel_1,...,pixel_N
0,0.123,0.456,...,0.789
1,0.234,0.567,...,0.890
```

For image datasets, the converter automatically:
- Resizes images to consistent dimensions
- Flattens to pixel values
- Normalizes to [0, 1] range

## Configuration

### Agent Settings (`agent.py`)

```python
OPENAI_MODEL = "gpt-4o-mini"  # Model to use
```

### Evaluation Settings (`evaluate_agent.py`)

- Training epochs for testing: 2
- Timeout: 300 seconds
- Dependency install timeout: 180 seconds

## Advanced Usage

### Adding Custom Instructions

```bash
python agent.py \
  --train data.csv \
  --val data.csv \
  --instructions "Use dropout for regularization"
```

### Testing Different Models

Edit `agent.py`:
```python
OPENAI_MODEL = "gpt-4o"  # For better quality
# or
OPENAI_MODEL = "gpt-4o-mini"  # For faster/cheaper
```

## Development

### Adding New Datasets

1. Convert to CSV format: `label, feature1, feature2, ...`
2. Place in `datasets/` directory
3. Run evaluation:
   ```bash
   python evaluate_agent.py --train datasets/new_train.csv --val datasets/new_val.csv --run-agent
   ```

### Improving Prompts

Edit `build_model_generation_prompt()` in `agent.py` to:
- Add more requirements
- Include examples
- Specify constraints
- Guide architecture choices

## Troubleshooting

**Issue: Import errors during evaluation**
- Solution: Dependencies auto-install from `requirements.txt`
- Manual install: `pip install -r generated/requirements.txt`

**Issue: Agent fails to generate code**
- Check API key in `.env`
- Verify dataset format (CSV with label column)
- Check OpenAI API status

**Issue: Low accuracy**
- Try more epochs: Edit evaluation harness timeout
- Use better model: Switch to `gpt-4o`
- Add custom instructions for architecture

## License

MIT

## Contributing

Feel free to open issues or submit PRs!
