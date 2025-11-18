import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Any
import openai
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import subprocess
import re
import sys

# Load environment variables from .env file
load_dotenv()

# ---------- Configuration ----------
OPENAI_MODEL = "gpt-4o-mini"  # Using mini model for faster/cheaper testing
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ---------- Helpers to inspect dataset ----------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def find_first_image(path: Path) -> Optional[Path]:
    for p in path.rglob("*"):
        if p.is_file() and is_image_file(p):
            return p
    return None


def infer_image_folder_properties(root_dir: Path) -> Dict[str, Any]:
    """
    Assumes standard ImageFolder layout: root/class_x/xxx.png
    Returns: {'num_classes': int, 'example_size': (H, W), 'channels': int}
    """
    classes = [d for d in root_dir.iterdir() if d.is_dir()]
    num_classes = len(classes)
    first_img = find_first_image(root_dir)
    if first_img is None:
        raise ValueError(f"No image files found under {root_dir}")
    with Image.open(first_img) as im:
        w, h = im.size
        mode = im.mode  # "RGB", "L", etc
    channels = 1 if mode == "L" else 3
    return {"type": "image_folder", "num_classes": num_classes, "example_size": (h, w), "channels": channels}


def infer_csv_properties(csv_path: Path) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=1000)
    n_cols = df.shape[1]
    # heuristics: if a column named 'target' or 'label' present -> classification/regression detection limited
    target_candidates = [c for c in df.columns if c.lower() in ("target", "label", "y")]
    task = "regression"
    n_classes = None
    if target_candidates:
        tcol = target_candidates[0]
        if pd.api.types.is_integer_dtype(df[tcol]) or pd.api.types.is_object_dtype(df[tcol]):
            # treat as classification if few unique values
            nunique = df[tcol].nunique(dropna=True)
            if nunique <= 50:
                task = "classification"
                n_classes = int(nunique)
        elif pd.api.types.is_float_dtype(df[tcol]):
            task = "regression"
    # Limit sample data to avoid context length issues with high-dimensional data (e.g., images)
    if n_cols > 100:
        # For high-dimensional data, only include shape info, not actual samples
        return {"type": "csv", "n_columns": n_cols, "task": task, "n_classes": n_classes, "sample_shape": f"{len(df)} rows × {n_cols} columns"}
    else:
        return {"type": "csv", "n_columns": n_cols, "task": task, "n_classes": n_classes, "sample_head": df.head(3).to_dict(orient="records")}


def inspect_dataset(train_path: str, val_path: str) -> Dict[str, Any]:
    """
    Try to infer dataset type and key properties from the provided split directories/paths.
    Returns a dictionary of facts to use in the LLM prompt.
    """
    facts = {"train_path": train_path, "val_path": val_path}
    # prefer directories for image folder detection
    p_train = Path(train_path)
    p_val = Path(val_path)
    try:
        # If train path is a directory containing subdirectories -> likely image folder
        if p_train.is_dir():
            # quick check: does it contain subdirectories with image files?
            any_images = find_first_image(p_train) is not None
            if any_images:
                props = infer_image_folder_properties(p_train)
                facts.update(props)
                facts["split_type"] = "image_folder_splits"
                return facts
        # If train path is a csv file
        if p_train.is_file() and p_train.suffix.lower() in (".csv", ".tsv"):
            props = infer_csv_properties(p_train)
            facts.update(props)
            facts["split_type"] = "csv"
            return facts
        # fallback: check val/test for csv
        if p_val.is_file() and p_val.suffix.lower() in (".csv", ".tsv"):
            props = infer_csv_properties(p_val)
            facts.update(props)
            facts["split_type"] = "csv"
            return facts
    except Exception as e:
        facts["inspect_error"] = str(e)

    # generic fallback
    facts["split_type"] = "unknown"
    return facts


# ---------- LLM prompt builder ----------
def build_model_generation_prompt(facts: Dict[str, Any], additional_instructions: Optional[str] = None) -> str:
    """
    Build a prompt for generating model.py and requirements.txt.
    Updated to:
      - Handle CSV image datasets with automatic height/width inference.
      - Assume 3 channels if number of pixels is not a perfect square.
      - Ensures transfer learning compatibility.
    """
    lines = []
    lines.append("You are to generate TWO files:")
    lines.append("")
    lines.append("1. `model.py` - A Python file that includes:")
    lines.append("- A PyTorch model class (nn.Module) appropriate for the dataset.")
    lines.append("- A `train()` function that trains the model for multiple epochs.")
    lines.append("- A `validate()` function that evaluates the model on a validation dataset and reports metrics.")
    lines.append("- An `if __name__ == '__main__'` block with argparse CLI.")
    lines.append("")
    lines.append("CRITICAL REQUIREMENTS:")
    lines.append("")
    lines.append("1. ARGPARSE INTERFACE - Use these EXACT argument names:")
    lines.append("   --train (path to training data)")
    lines.append("   --val (path to validation data)")
    lines.append("   --epochs (number of epochs, default=10)")
    lines.append("   --batch-size (batch size, default=64)")
    lines.append("   --lr (learning rate, default=0.001)")
    lines.append("")
    lines.append("2. DEVICE HANDLING - MUST support both CPU and GPU:")
    lines.append("   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    lines.append("   Move model and data to device in train/validate functions")
    lines.append("")
    lines.append("3. TRAINING LOOP - Proper structure:")
    lines.append("   - Train function should accept epochs parameter and run full training loop")
    lines.append("   - Print average loss per epoch (not just last batch)")
    lines.append("   - In main: call train() ONCE with all epochs, not in a loop")
    lines.append("")
    lines.append("4. LOGGING - Print after each epoch:")
    lines.append("   Epoch X/Y: Train Loss: 0.XXXX, Train Acc: XX.XX%, Val Loss: 0.XXXX, Val Acc: XX.XX%")
    lines.append("")
    lines.append("5. CHECKPOINTING - Save model when validation metric improves:")
    lines.append("   torch.save(model.state_dict(), 'best_model.pth')")
    lines.append("   Print message when saving: 'Saved best model (Val Acc: XX.XX%)'")
    lines.append("")
    lines.append("6. CODE QUALITY:")
    lines.append("   - Use proper variable names")
    lines.append("   - Add docstrings to functions")
    lines.append("   - Handle edge cases (empty datasets, etc.)")
    lines.append("")
    lines.append("7. DEPENDENCIES - ONLY use these libraries:")
    lines.append("   - torch (and torch.nn, torch.optim, torch.utils.data)")
    lines.append("   - pandas (for CSV reading)")
    lines.append("   - numpy (optional)")
    lines.append("   - torchvision (ONLY for ImageFolder OR transfer learning)")
    lines.append("   DO NOT import: sklearn, scipy, matplotlib, seaborn")
    lines.append("")

    lines.append("Dataset facts:")
    lines.append(json.dumps(facts, indent=2))
    lines.append("")

    # -------------------------
    # IMAGE FOLDER
    # -------------------------
    if facts.get("split_type") == "image_folder":
        lines.append("DATASET TYPE: Image Folder")
        lines.append("- Use torchvision.datasets.ImageFolder")
        lines.append("- Use transforms: Resize(224,224), ToTensor(), Normalize([...])")
        lines.append("- Always use transfer learning (ResNet18) unless num_classes <=1")
        lines.append("- Input: channels={channels}, size={example_size}, classes={num_classes}".format(**facts))
        lines.append("- Loss: CrossEntropyLoss")

    # -------------------------
    # CSV IMAGE DATA
    # -------------------------
    elif facts.get("split_type") == "csv":
        lines.append("DATASET TYPE: CSV Image Data")
        lines.append("- CSV rows contain: label + flattened pixel values")
        lines.append("- Automatically infer image shape and channels from row length:")
        lines.append("      num_pixels = len(row) - 1  # exclude label")
        lines.append("      if (num_pixels ** 0.5).is_integer():")
        lines.append("          height = width = int(num_pixels ** 0.5)")
        lines.append("          channels = 1  # single-channel grayscale")
        lines.append("      else:")
        lines.append("          height = width = int((num_pixels // 3) ** 0.5)")
        lines.append("          channels = 3  # multi-channel image")
        lines.append("- Reshape tensor:")
        lines.append("      img = img.reshape(channels, height, width)")
        lines.append("- Repeat channels to 3 if single-channel for transfer learning:")
        lines.append("      if channels == 1:")
        lines.append("          img = img.repeat(3, 1, 1)")
        lines.append("- Do NOT use transforms.ToPILImage()")
        lines.append("- Use tensor transforms: Resize(224,224), Normalize([...])")
        if facts.get("task") == "classification":
            lines.append(f"- Task: Classification with {facts.get('n_classes')} classes")
            lines.append("- Loss: CrossEntropyLoss")
            lines.append("- Metrics: Accuracy")
        else:
            lines.append("- Task: Regression")
            lines.append("- Loss: MSELoss")
            lines.append("- Metrics: RMSE or MAE")

    # -------------------------
    else:
        lines.append("Dataset type unknown; create flexible transfer-learning template with TODOs")

    lines.append("")
    lines.append("MODEL ARCHITECTURE REQUIREMENTS:")
    lines.append("- ALWAYS use transfer learning with torchvision.models.resnet18 pretrained=True")
    lines.append("- Replace the final FC layer with correct output dimension.")

    lines.append("")
    lines.append("EXAMPLE STRUCTURE (follow this pattern):")
    lines.append("```")
    lines.append("# Code outline...")
    lines.append("```")
    lines.append("")

    if additional_instructions:
        lines.append("ADDITIONAL INSTRUCTIONS:")
        lines.append(additional_instructions)
        lines.append("")

    lines.append("2. `requirements.txt` - List all pip packages needed (one per line)")
    lines.append("   Include version constraints if important.")
    lines.append("   Packages: torch, torchvision, pandas, numpy, pillow")
    lines.append("")
    lines.append("OUTPUT FORMAT - Use this EXACT structure:")
    lines.append("=== requirements.txt ===")
    lines.append("package1")
    lines.append("package2>=version")
    lines.append("")
    lines.append("=== model.py ===")
    lines.append("import package1")
    lines.append("# ... rest of the file")
    lines.append("")
    lines.append("IMPORTANT: Use the === markers exactly as shown. No markdown code fences, no explanations.")
    return "\n".join(lines)

def run_model(train_path, val_path):
    cmd = ["python", "model.py", "--train", train_path, "--val", val_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

def extract_metrics(output_text: str) -> Dict[str, Optional[float]]:
    """
    Extract best (lowest) validation loss and best (highest) validation accuracy
    from model training logs. Matches both 'Validation Loss' and 'Val Loss' formats.

    Example matches:
        "Val Loss: 0.1234, Val Accuracy: 95.23%"
        "Validation Loss: 0.341, Validation Accuracy: 92.1%"
    """
    # Find all loss values (support both "Val" and "Validation")
    loss_matches = re.findall(r"(?:Val(?:idation)?\s*Loss)[:\s]+([0-9.]+)", output_text, re.IGNORECASE)

    # Find all accuracy values (support both "Val" and "Validation Accuracy")
    acc_matches = re.findall(r"(?:Val(?:idation)?\s*Acc(?:uracy)?)[:\s]+([0-9.]+)", output_text, re.IGNORECASE)

    # Convert to floats safely
    losses = [float(x) for x in loss_matches if re.match(r"^\d+(\.\d+)?$", x)]
    accs = [float(x) for x in acc_matches if re.match(r"^\d+(\.\d+)?$", x)]

    # Return best observed metrics
    metrics = {
        "val_loss": min(losses) if losses else None,
        "val_acc": max(accs) if accs else None,
    }

    return metrics


def build_refinement_prompt(prev_code: str, train_output: str, metrics: Dict[str, float]) -> str:
    """
    Build an instruction prompt for the LLM to improve the model based on
    full training/validation logs and extracted metrics.
    """
    val_loss = metrics.get("val_loss", "unknown")
    val_acc = metrics.get("val_acc", "unknown")

    prompt = f"""
        You are an AI code assistant refining a PyTorch training script.

        Below is the **full training log** from running the current model. 
        Use it to analyze learning progress and improve model design and training strategy.

        Training Output:
        ----------------
        {train_output.strip()}
        ----------------

        Summary of last metrics:
        - Validation Loss: {val_loss}
        - Validation Accuracy: {val_acc}

        Your task:
        - Modify the code to improve model performance.
        - You may change:
        - Model architecture (depth, layers, dropout, normalization, etc.)
        - Learning rate, optimizer, batch size, or scheduler
        - Data augmentations or regularization
        - Keep the same CLI interface (arguments: --train, --val)
        - The output must remain a **runnable single-file Python script**.
        - Output *only* the Python code for model.py — no explanations or markdown.

        Previous model.py:

        {prev_code.strip()}

        """
    return prompt.strip()

def build_langgraph_pipeline(prompt: str) -> Optional[str]:
    """
    Build and run a LangGraph pipeline using StateGraph and ChatOpenAI node.
    Input: prompt (string)
    Output: model code (string) or None on failure
    """
    try:
        # 1. Define the structure of the graph
        def inspector_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"prompt": state["prompt"]}

        def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
            llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
            response = llm.invoke(state["prompt"])
            # response can be either a dict or string depending on wrapper
            content = getattr(response, "content", str(response))
            return {"chat_output": content}

        def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"final_output": state["chat_output"]}

        # 2. Construct the graph using StateGraph
        builder = StateGraph(dict)

        builder.add_node("inspector", inspector_node)
        builder.add_node("llm", llm_node)
        builder.add_node("writer", writer_node)

        builder.add_edge(START, "inspector")
        builder.add_edge("inspector", "llm")
        builder.add_edge("llm", "writer")
        builder.add_edge("writer", END)

        # 3. Compile the graph
        graph = builder.compile()

        # 4. Run it with our prompt as input
        result = graph.invoke({"prompt": prompt})

        # 5. Extract output
        code_text = result.get("final_output")
        if not code_text:
            return None

        # Clean code block formatting if needed
        code_text = (
            code_text.replace("```python", "")
            .replace("```", "")
            .strip()
        )
        return code_text

    except Exception as e:
        print(f"LangGraph pipeline failed: {e}")
        return None




# ---------- Main agent logic ----------
def parse_dual_output(llm_output: str) -> Dict[str, str]:
    """
    Parse LLM output into requirements.txt and model.py.
    Expected format:
    === requirements.txt ===
    package1
    package2

    === model.py ===
    python code here
    """
    files = {}

    # Look for === markers
    if "=== requirements.txt ===" in llm_output and "=== model.py ===" in llm_output:
        parts = llm_output.split("=== requirements.txt ===")
        if len(parts) >= 2:
            rest = parts[1]
            req_and_model = rest.split("=== model.py ===")
            if len(req_and_model) >= 2:
                files["requirements.txt"] = req_and_model[0].strip()
                files["model.py"] = req_and_model[1].strip()
                return files

    # Fallback: if no markers found, treat entire output as model.py
    print("Warning: Could not find === markers, treating entire output as model.py")
    files["model.py"] = llm_output.strip()
    files["requirements.txt"] = "torch\npandas\nnumpy"  # default dependencies

    return files


def write_files(files: Dict[str, str]) -> None:
    """Write multiple files from dict."""
    for filename, content in files.items():
        # Clean up any remaining code block markers
        content = content.replace("```python", "").replace("```", "").strip()

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Wrote {filename}")


def write_model_file(contents: str, output_path: str = "model.py") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(contents)
    print(f"Wrote {output_path}")

def install_requirements(req_file="requirements.txt"):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])


def agent_main(args):
    facts = inspect_dataset(args.train, args.val)
    prompt = build_model_generation_prompt(facts, additional_instructions=args.instructions)

    # First model generation
    print("Calling LLM to generate initial code and dependencies...")
    llm_output = build_langgraph_pipeline(prompt)
    if not llm_output:
        print("✗ Initial model generation failed.")
        return

    print("✓ Obtained initial model from LangGraph pipeline.")
    files = parse_dual_output(llm_output)
    write_files(files)
    install_requirements("requirements.txt")


    # Start refinement loop
    best_code = files["model.py"]
    code = best_code
    best_loss = float("inf")

    for i in range(args.iterations):
        print(f"\n=== Iteration {i + 1}/{args.iterations} ===")

        # Write and run model
        write_model_file(code, args.output)
        stdout, stderr = run_model(args.train, args.val)

        if stderr.strip():
            print("⚠️ Model training produced errors:\n", stderr)

        print("Model output:\n", stdout)
        metrics = extract_metrics(stdout)
        print("Extracted metrics:", metrics)

        # Track best model by lowest val_loss
        if metrics.get("val_loss") is not None and metrics["val_loss"] < best_loss:
            best_loss = metrics["val_loss"]
            best_code = open(args.output).read()
            print(f"✅ New best model found (Val Loss: {best_loss:.4f})")

        # Build refinement prompt using full logs + metrics
        refine_prompt = build_refinement_prompt(code, stdout, metrics)
        refined_code = build_langgraph_pipeline(refine_prompt)

        if refined_code:
            code = refined_code
            print("Refinement iteration complete — model updated.")
        else:
            print("No refinement returned; stopping.")
            break

    # Write best final version
    write_model_file(best_code, "best_model.py")
    print(f"\nBest model saved as best_model.py with val_loss={best_loss}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoML Agent - Generate ML models from datasets or natural language prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Direct dataset paths
  python agent.py --train data/train.csv --val data/test.csv

  # Mode 2: Search and build from prompt
  python agent.py --prompt "Build a classifier for handwritten digits" --max-results 5

  # With custom output and iterations
  python agent.py --train data/train.csv --val data/test.csv --output my_model.py --iterations 5
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group('Mode: Choose one')
    mode_group.add_argument("--prompt", type=str, help="Natural language task description (enables search mode)")
    mode_group.add_argument("--train", type=str, help="Path to training split (dir or csv) (direct mode)")

    # Common arguments
    parser.add_argument("--val", type=str, help="Path to validation split (dir or csv)")
    parser.add_argument("--output", default="model.py", help="Output file for the generated model code")
    parser.add_argument("--instructions", default="", help="Extra instructions to include in prompt")
    parser.add_argument("--iterations", type=int, default=3, help="Number of model refinement iterations")

    # Search mode arguments
    search_group = parser.add_argument_group('Search Mode Options')
    search_group.add_argument("--max-results", type=int, default=5, help="Max number of datasets to search (default: 5)")
    search_group.add_argument("--select-dataset", type=int, default=1, help="Which dataset to use from results (default: 1)")
    search_group.add_argument("--dataset-dir", type=str, default="./datasets", help="Directory to save datasets (default: ./datasets)")
    search_group.add_argument("--sources", type=str, nargs='+',
                             choices=['kaggle', 'huggingface', 'openml', 'paperswithcode', 'uci'],
                             help="Data sources to search (default: kaggle huggingface paperswithcode uci)")

    args = parser.parse_args()

    # Validate arguments
    if args.prompt and args.train:
        parser.error("Cannot use both --prompt and --train. Choose one mode.")

    if not args.prompt and not args.train:
        parser.error("Must provide either --prompt (search mode) or --train (direct mode)")

    # Execute appropriate mode
    if args.prompt:
        # Search mode: Find dataset, download, then generate code
        print("="*80)
        print("🤖 AutoML Agent - Search and Build Mode")
        print("="*80)

        from dataset_search import search_datasets
        from dataset_converter import download_and_convert_dataset
        import sys

        # Step 1: Search for datasets
        print(f"\n{'='*80}")
        print("STEP 1: SEARCHING FOR DATASETS")
        print("="*80)

        requirements, datasets = search_datasets(
            args.prompt,
            max_results=args.max_results,
            sources=args.sources
        )

        if not datasets:
            print("\n❌ No datasets found. Please try a different prompt.")
            sys.exit(1)

        # Select dataset
        selected_idx = args.select_dataset - 1
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

        # Step 2: Download and convert
        print(f"\n{'='*80}")
        print("STEP 2: DOWNLOADING AND CONVERTING DATASET")
        print("="*80)

        try:
            train_csv, test_csv = download_and_convert_dataset(selected_dataset, args.dataset_dir)
        except Exception as e:
            print(f"\n❌ Failed to download/convert dataset: {e}")
            print(f"\n💡 Tip: Make sure you have credentials configured:")
            print(f"   - Kaggle: ~/.kaggle/kaggle.json")
            print(f"   - HuggingFace: Run `huggingface-cli login`")
            sys.exit(1)

        # Update args for agent_main
        args.train = train_csv
        args.val = test_csv
        if not args.instructions:
            args.instructions = f"This is for the task: {args.prompt}"

        # Step 3: Generate code
        print(f"\n{'='*80}")
        print("STEP 3: GENERATING MODEL CODE")
        print("="*80)

        agent_main(args)

        print(f"\n{'='*80}")
        print("🎉 AGENT COMPLETE!")
        print("="*80)
        print(f"\n📁 Generated files:")
        print(f"   Dataset (train): {train_csv}")
        print(f"   Dataset (test): {test_csv}")
        print(f"   Model code: {args.output}")
        print(f"   Best model: best_model.py")

    else:
        # Direct mode: Use provided dataset paths
        if not args.val:
            parser.error("--val is required when using --train (direct mode)")

        print("="*80)
        print("🤖 AutoML Agent - Direct Mode")
        print("="*80)

        agent_main(args)