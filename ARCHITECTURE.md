# AutoML System Builder - Architecture

## Overview

The AutoML System Builder is now unified under a single entry point: **`agent.py`**

## Architecture

```
agent.py (MAIN ENTRY POINT)
├── Mode 1: Direct Mode
│   └── User provides dataset paths → Generate model code
│
└── Mode 2: Search Mode
    ├── dataset_search.py → Search Kaggle/HuggingFace
    ├── dataset_converter.py → Download & convert to CSV
    └── Generate model code
```

## Usage

### Mode 1: Direct Mode (Traditional)
Use when you already have datasets:

```bash
python agent.py --train data/train.csv --val data/test.csv
```

**Options:**
- `--output` - Output filename (default: model.py)
- `--iterations` - Refinement iterations (default: 3)
- `--instructions` - Extra context for LLM

**Example:**
```bash
python agent.py \
  --train examples/mnist/mnist_train.csv \
  --val examples/mnist/mnist_test.csv \
  --output my_classifier.py \
  --iterations 5
```

### Mode 2: Search Mode (New!)
Use when you want the agent to find and download datasets:

```bash
python agent.py --prompt "Build a classifier for handwritten digits"
```

**Options:**
- `--max-results` - Number of datasets to search (default: 5)
- `--select-dataset` - Which dataset to use (default: 1 = best match)
- `--dataset-dir` - Where to save datasets (default: ./datasets)
- `--output`, `--iterations` - Same as direct mode

**Example:**
```bash
python agent.py \
  --prompt "Classify different types of animals" \
  --max-results 10 \
  --select-dataset 2 \
  --dataset-dir ./my_datasets \
  --iterations 3
```

## Component Files

### Core Files
- **`agent.py`** - Main entry point (both modes)
- **`dataset_search.py`** - Searches Kaggle/HuggingFace datasets
- **`dataset_converter.py`** - Downloads and converts datasets to CSV
- **`evaluate_agent.py`** - Evaluates generated models

### Legacy Files
- **`search_and_build.py`** - OLD entry point (deprecated)
  - Now replaced by `agent.py --prompt`
  - Kept for backward compatibility

## Dataset Converter Features

The dataset converter (`dataset_converter.py`) now includes:

### Smart Structure Detection
- Automatically finds ImageFolder structures in nested directories
- Handles non-standard layouts
- Reports discovered classes and sample counts

### Validation
- Checks for empty conversions
- Validates CSV files have data before returning
- Clear error messages with troubleshooting hints

### Supported Formats
- CSV files (train/test splits)
- Image folders (ImageFolder format)
- HuggingFace datasets
- Kaggle datasets

## Workflow

### Search Mode Workflow:
```
1. User provides natural language prompt
   ↓
2. Parse requirements (task type, data type, keywords)
   ↓
3. Search Kaggle + HuggingFace
   ↓
4. Download best matching dataset
   ↓
5. Convert to CSV format (if needed)
   ↓
6. Generate PyTorch training code
   ↓
7. Run refinement iterations
   ↓
8. Save best model
```

### Direct Mode Workflow:
```
1. User provides dataset paths
   ↓
2. Inspect dataset properties
   ↓
3. Generate PyTorch training code
   ↓
4. Run refinement iterations
   ↓
5. Save best model
```

## Evaluation

Evaluate any generated model:

```bash
python evaluate_agent.py \
  --train data/train.csv \
  --val data/test.csv \
  --model generated_model.py \
  --output results.json
```

Tests include:
- Functional correctness (syntax, imports, components)
- Performance (training runs, checkpoint creation)
- Generates detailed JSON report

## Examples

### End-to-End Example (Search Mode)
```bash
# Search for dog/cat classifier dataset and generate model
python agent.py \
  --prompt "Build an image classifier for dogs and cats" \
  --max-results 5 \
  --iterations 3
```

### Traditional Example (Direct Mode)
```bash
# Use existing MNIST dataset
python agent.py \
  --train examples/mnist/mnist_train.csv \
  --val examples/mnist/mnist_test.csv \
  --iterations 2
```

### Evaluation Example
```bash
# Evaluate the generated model
python evaluate_agent.py \
  --train examples/mnist/mnist_train.csv \
  --val examples/mnist/mnist_test.csv \
  --model model.py \
  --output evaluation_results.json
```

## Key Improvements

1. **Unified Entry Point** - `agent.py` handles all use cases
2. **Smart Dataset Detection** - Handles nested/complex directory structures
3. **Better Error Messages** - Clear feedback when things go wrong
4. **Data Validation** - Catches empty datasets early
5. **Flexible Architecture** - Easy to extend with new modes

## Configuration

### Required Environment Variables
- `OPENAI_API_KEY` - For LLM code generation

### Optional Credentials
- **Kaggle**: `~/.kaggle/kaggle.json` (for dataset downloads)
- **HuggingFace**: Run `huggingface-cli login` (for HF datasets)

## Future Enhancements

- [ ] Support for more dataset formats (TFRecords, COCO, etc.)
- [ ] Interactive dataset selection CLI
- [ ] Model architecture search
- [ ] Hyperparameter optimization
- [ ] Multi-GPU training support
- [ ] Docker containerization
