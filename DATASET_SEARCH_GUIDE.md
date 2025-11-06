# Dataset Search and Build Guide

This guide explains how to use the new dataset search and automated model building features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup Requirements](#setup-requirements)
3. [Complete Workflow](#complete-workflow)
4. [Use Cases & Examples](#use-cases--examples)
5. [Troubleshooting](#troubleshooting)
6. [API Reference](#api-reference)

---

## Quick Start

The simplest way to build a model from a natural language prompt:

```bash
python search_and_build.py --prompt "I want to create an image classifier for dogs and cats"
```

This will:
1. Parse your prompt to understand the task
2. Search Kaggle and HuggingFace for relevant datasets
3. Download and convert the best matching dataset
4. Generate PyTorch training code
5. Show you the next steps to train your model

---

## Setup Requirements

### Required

- **Python 3.8+**
- **OpenAI API Key** (for LLM-based code generation)

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Optional (for dataset search)

#### Kaggle API

Kaggle hosts thousands of public datasets for ML tasks.

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json`
4. Set up credentials:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### HuggingFace Hub

HuggingFace hosts ML-optimized datasets with easy loading.

```bash
# Option 1: Login via CLI
huggingface-cli login

# Option 2: Set token in .env
echo "HUGGINGFACE_TOKEN=your-token-here" >> .env
```

Get your token from: https://huggingface.co/settings/tokens

---

## Complete Workflow

### Step 1: Search for Datasets

Use natural language to describe your ML task:

```bash
python search_and_build.py --prompt "Build an image classifier for different types of flowers"
```

The system will:
- Parse your prompt using GPT-4o-mini
- Search multiple dataset sources
- Rank results by relevance
- Show you the top 5 matches

### Step 2: Select a Dataset

By default, the best-ranked dataset is selected. To choose a different one:

```bash
python search_and_build.py \
  --prompt "..." \
  --select-dataset 3  # Use the 3rd result instead
```

### Step 3: Download and Convert

The system automatically:
- Downloads the dataset
- Detects the format (CSV, images, etc.)
- Converts to standardized CSV format
- Creates train/test splits if needed

### Step 4: Generate Training Code

The LLM agent generates:
- Complete PyTorch model architecture
- Training and validation loops
- Data loading pipeline
- Checkpoint saving
- All dependencies

### Step 5: Train the Model

```bash
# Option A: Automatic training
python search_and_build.py \
  --prompt "..." \
  --train-model \
  --epochs 20 \
  --batch-size 128

# Option B: Manual training
cd generated
pip install -r requirements.txt
python model.py --train ../datasets/train.csv --val ../datasets/test.csv --epochs 20
```

---

## Use Cases & Examples

### Example 1: Image Classification

**Task**: Build a cat vs dog classifier

```bash
python search_and_build.py \
  --prompt "I want to classify images of cats and dogs" \
  --train-model \
  --epochs 15
```

**What happens**:
1. Searches for "cat dog image classification dataset"
2. Finds datasets like "Dogs vs. Cats" from Kaggle
3. Downloads and converts images to CSV (32x32 RGB)
4. Generates CNN or MLP model
5. Trains for 15 epochs

---

### Example 2: Text Classification

**Task**: Sentiment analysis for product reviews

```bash
python search_and_build.py \
  --prompt "Sentiment analysis for product reviews" \
  --select-dataset 1
```

**What happens**:
1. Identifies task as text classification
2. Searches for sentiment analysis datasets
3. Downloads from HuggingFace or Kaggle
4. Converts to CSV with text features
5. Generates appropriate model architecture

---

### Example 3: Regression Task

**Task**: Predict housing prices

```bash
python search_and_build.py \
  --prompt "Build a model to predict house prices based on features like size and location" \
  --train-model
```

**What happens**:
1. Identifies task as regression
2. Searches for housing price datasets
3. Downloads tabular data
4. Generates regression model (MSE loss)
5. Trains and saves best model

---

### Example 4: Custom Dataset Selection

**Task**: Explore multiple options before deciding

```bash
# Step 1: Search only (no download)
python dataset_search.py --prompt "flower classification" --max-results 10

# Step 2: Review results and pick one
# Output shows:
#   1. [KAGGLE] Flower Recognition (Score: 8.5)
#   2. [HUGGINGFACE] flowers-102 (Score: 7.2)
#   ...

# Step 3: Build with selected dataset
python search_and_build.py \
  --prompt "flower classification" \
  --select-dataset 2 \
  --train-model
```

---

### Example 5: Manual Dataset Download

**Task**: Download a specific dataset without search

```bash
# Download from Kaggle
python dataset_converter.py \
  --source kaggle \
  --name username/dataset-name \
  --output-dir ./my_datasets

# Then build model
python agent.py \
  --train my_datasets/train.csv \
  --val my_datasets/test.csv
```

---

## Troubleshooting

### Issue: "Kaggle API not authenticated"

**Solution**:
```bash
# Verify kaggle.json exists
ls -la ~/.kaggle/kaggle.json

# Should show: -rw------- (permissions 600)
# If not:
chmod 600 ~/.kaggle/kaggle.json
```

---

### Issue: "No datasets found"

**Possible causes**:
1. Prompt is too specific or vague
2. No matching datasets in Kaggle/HuggingFace

**Solution**:
```bash
# Try broader keywords
# Instead of: "golden retriever vs labrador classifier"
# Try: "dog breed classification"

# Or try different phrasing
python search_and_build.py --prompt "image classification for pets"
```

---

### Issue: "Dataset conversion failed"

**Common causes**:
- Unsupported dataset format
- Corrupted download
- Insufficient disk space

**Solution**:
```bash
# Check dataset format
ls -lh datasets/

# Try manual conversion
python dataset_converter.py \
  --source kaggle \
  --name dataset-name \
  --output-dir ./datasets
```

---

### Issue: "Generated model has low accuracy"

**Solutions**:
1. **Train longer**: `--epochs 50`
2. **Adjust learning rate**: `--lr 0.0001`
3. **Use larger batch size**: `--batch-size 256`
4. **Try better LLM**: Edit `agent.py` to use `gpt-4o` instead of `gpt-4o-mini`
5. **Add custom instructions**:
   ```bash
   python agent.py \
     --train data.csv \
     --val data.csv \
     --instructions "Use a deeper CNN with dropout and batch normalization"
   ```

---

### Issue: "HuggingFace download is slow"

**Solution**:
HuggingFace datasets can be large. Options:
1. Use Kaggle alternative: `--select-dataset 1` (if Kaggle result is available)
2. Download in background and run later
3. Use a smaller subset if available

---

## API Reference

### `search_and_build.py`

End-to-end pipeline from prompt to trained model.

**Required Arguments**:
- `--prompt`: Natural language description of ML task

**Optional Arguments**:
- `--max-results`: Number of search results (default: 5)
- `--select-dataset`: Which result to use (default: 1 = best match)
- `--output-dir`: Where to save generated code (default: ./generated)
- `--dataset-dir`: Where to save datasets (default: ./datasets)
- `--train-model`: Automatically train after generation
- `--epochs`: Training epochs (default: 10)
- `--batch-size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--evaluate`: Run evaluation harness
- `--skip-search`: Use existing dataset files

**Examples**:
```bash
# Minimal
python search_and_build.py --prompt "cat vs dog classifier"

# Full control
python search_and_build.py \
  --prompt "flower classification" \
  --max-results 10 \
  --select-dataset 3 \
  --train-model \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.0005 \
  --evaluate
```

---

### `dataset_search.py`

Search for datasets without downloading.

**Required Arguments**:
- `--prompt`: Natural language description

**Optional Arguments**:
- `--max-results`: Number of results (default: 5)

**Example**:
```bash
python dataset_search.py \
  --prompt "sentiment analysis twitter" \
  --max-results 10
```

---

### `dataset_converter.py`

Download and convert specific datasets.

**Required Arguments**:
- `--source`: "kaggle" or "huggingface"
- `--name`: Dataset name/reference

**Optional Arguments**:
- `--output-dir`: Output directory (default: ./datasets)

**Examples**:
```bash
# Kaggle
python dataset_converter.py \
  --source kaggle \
  --name username/dataset-name \
  --output-dir ./data

# HuggingFace
python dataset_converter.py \
  --source huggingface \
  --name org/dataset-name \
  --output-dir ./data
```

---

## Tips & Best Practices

### 1. Prompt Writing

**Good prompts**:
- ✅ "Build an image classifier for different dog breeds"
- ✅ "Predict customer churn using historical data"
- ✅ "Sentiment analysis for movie reviews"

**Bad prompts**:
- ❌ "ML" (too vague)
- ❌ "Build me the best AI ever" (unrealistic)
- ❌ "Classify images" (missing context)

### 2. Dataset Selection

- First result is usually best, but not always
- Check dataset size (shown in search results)
- Kaggle datasets often have better documentation
- HuggingFace datasets are more ML-optimized

### 3. Training Parameters

Start with defaults, then tune:
```bash
# Quick test (2 epochs)
--epochs 2 --batch-size 64

# Standard training (10-20 epochs)
--epochs 15 --batch-size 128

# Long training (50+ epochs)
--epochs 100 --batch-size 256 --lr 0.0001
```

### 4. Resource Management

- **Small datasets** (<100MB): Train locally
- **Medium datasets** (100MB-1GB): Use GPU if available
- **Large datasets** (>1GB): Consider using cloud or reducing batch size

---

## Advanced Workflows

### Workflow 1: Ensemble Models

Build multiple models and ensemble:

```bash
# Model 1
python search_and_build.py --prompt "cat dog" --select-dataset 1 --train-model
mv generated/best_model.pth model1.pth

# Model 2
python search_and_build.py --prompt "cat dog" --select-dataset 2 --train-model
mv generated/best_model.pth model2.pth

# Ensemble predictions (custom code)
```

### Workflow 2: Transfer Learning

Start with pre-trained features:

```bash
# Generate base model
python search_and_build.py --prompt "general object classification"

# Fine-tune on specific task
python agent.py \
  --train specific_dataset.csv \
  --val specific_test.csv \
  --instructions "Use transfer learning with pre-trained features"
```

### Workflow 3: Hyperparameter Search

Automated parameter tuning:

```bash
# Try different learning rates
for lr in 0.001 0.0001 0.00001; do
  python search_and_build.py \
    --prompt "..." \
    --skip-search \
    --train-model \
    --lr $lr \
    --output-dir results_lr_$lr
done

# Compare results
ls -lh results_*/best_model.pth
```

---

## Next Steps

After building your model:

1. **Evaluate**: Use `evaluate_agent.py` for comprehensive testing
2. **Deploy**: Export model for inference (separate guide)
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Iterate**: Try different datasets or architectures

---

## Getting Help

- **Documentation**: See main `README.md`
- **Issues**: Open an issue on GitHub
- **Examples**: Check `examples/` directory

Happy building! 🚀
