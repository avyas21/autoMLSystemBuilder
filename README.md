# AutoML System Builder

An AI-powered agent that automatically generates PyTorch training code for machine learning datasets.

## Overview

This system uses LLMs (via LangGraph + OpenAI) to:
1. Inspect datasets and infer properties
2. Generate complete PyTorch training code (`model.py`)
3. Generate dependency specifications (`requirements.txt`)
4. Automatically install dependencies
5. Evaluate code quality and performance

## Project Structure

```
autoMLSystemBuilder/
├── agent.py                    # Core agent (generates ML code)
├── evaluate_agent.py           # Evaluation harness
├── datasets/                   # Training datasets (gitignored)
│   ├── mnist_train.csv
│   ├── mnist_test.csv
│   ├── fashion_mnist_train.csv
│   └── fashion_mnist_test.csv
├── utils/                      # Utility scripts
│   ├── download_fashion_mnist.py
│   └── compare_results.py
├── examples/                   # Example evaluation results
│   └── evaluation_results/
│       ├── mnist.json
│       └── fashion_mnist.json
└── generated/                  # Generated code (gitignored)
    ├── model.py
    ├── requirements.txt
    └── best_model.pth
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install openai langgraph langchain-openai python-dotenv pandas pillow torch
   ```

2. **Configure OpenAI API key:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

3. **Download datasets:**
   ```bash
   # Fashion-MNIST
   python utils/download_fashion_mnist.py

   # MNIST (via git-lfs)
   git lfs pull
   ```

## Usage

### Quick Start

Generate and evaluate code on MNIST:
```bash
python evaluate_agent.py \
  --train datasets/mnist_train.csv \
  --val datasets/mnist_test.csv \
  --run-agent \
  --output results.json
```

### Step-by-Step

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
