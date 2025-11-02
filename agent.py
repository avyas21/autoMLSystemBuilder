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
    Build a clear prompt for ChatGPT to generate model.py and requirements.txt.
    The assistant should return structured output with both files.
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
    lines.append("7. DEPENDENCIES - ONLY use these libraries (DO NOT import anything else):")
    lines.append("   - torch (and torch.nn, torch.optim, torch.utils.data)")
    lines.append("   - pandas (for CSV reading)")
    lines.append("   - numpy (if needed for array operations)")
    lines.append("   DO NOT import: sklearn, scipy, torchvision (unless image_folder), matplotlib, seaborn")
    lines.append("")
    lines.append("Dataset facts:")
    lines.append(json.dumps(facts, indent=2))
    lines.append("")
    if facts.get("split_type") == "image_folder":
        lines.append("DATASET TYPE: Image Folder")
        lines.append("- Use torchvision.datasets.ImageFolder or custom Dataset")
        lines.append("- Apply transforms: ToTensor(), Normalize()")
        lines.append("- Input: channels={channels}, size={example_size}, classes={num_classes}".format(**facts))
        lines.append("- Use CrossEntropyLoss for classification")
        lines.append("- Architecture: Use a simple CNN (Conv2d -> Pool -> FC layers)")
    elif facts.get("split_type") == "csv":
        lines.append("DATASET TYPE: CSV Tabular Data")
        lines.append("- Create custom Dataset class that reads CSV with pandas")
        lines.append("- Normalize features: divide by 255.0 if pixel data, otherwise use standard scaling")
        lines.append("- Architecture: Use MLP (Linear -> ReLU -> Linear layers)")
        if facts.get("task") == "classification":
            lines.append("- Task: Classification with {} classes".format(facts.get("n_classes", "unknown")))
            lines.append("- Loss: CrossEntropyLoss")
            lines.append("- Metrics: Accuracy")
        else:
            lines.append("- Task: Regression")
            lines.append("- Loss: MSELoss")
            lines.append("- Metrics: RMSE or MAE")
    else:
        lines.append("Dataset type unknown; create flexible MLP template with TODOs")

    lines.append("")
    lines.append("EXAMPLE STRUCTURE (follow this pattern):")
    lines.append("```")
    lines.append("import argparse")
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("# ... other imports")
    lines.append("")
    lines.append("class MyDataset(Dataset):")
    lines.append("    # Dataset implementation")
    lines.append("")
    lines.append("class MyModel(nn.Module):")
    lines.append("    # Model architecture")
    lines.append("")
    lines.append("def train(model, train_loader, criterion, optimizer, device, epochs):")
    lines.append("    model.train()")
    lines.append("    for epoch in range(epochs):")
    lines.append("        total_loss = 0")
    lines.append("        for data, target in train_loader:")
    lines.append("            data, target = data.to(device), target.to(device)")
    lines.append("            # ... training step")
    lines.append("        avg_loss = total_loss / len(train_loader)")
    lines.append("        # ... print and return metrics")
    lines.append("")
    lines.append("def validate(model, val_loader, criterion, device):")
    lines.append("    model.eval()")
    lines.append("    # ... validation logic with data.to(device)")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    parser = argparse.ArgumentParser()")
    lines.append("    parser.add_argument('--train', required=True)")
    lines.append("    parser.add_argument('--val', required=True)")
    lines.append("    parser.add_argument('--epochs', type=int, default=10)")
    lines.append("    # ... other args")
    lines.append("    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')")
    lines.append("    # ... setup and training")
    lines.append("```")
    lines.append("")

    if additional_instructions:
        lines.append("ADDITIONAL INSTRUCTIONS:")
        lines.append(additional_instructions)
        lines.append("")

    lines.append("")
    lines.append("2. `requirements.txt` - List all pip packages needed (one per line)")
    lines.append("   Include version constraints if important (e.g., torch>=2.0.0)")
    lines.append("   Common packages: torch, pandas, numpy, scikit-learn, pillow")
    lines.append("")
    lines.append("OUTPUT FORMAT - Use this EXACT structure:")
    lines.append("=== requirements.txt ===")
    lines.append("package1")
    lines.append("package2>=version")
    lines.append("package3")
    lines.append("")
    lines.append("=== model.py ===")
    lines.append("import package1")
    lines.append("# ... rest of Python code")
    lines.append("")
    lines.append("IMPORTANT: Use the === markers exactly as shown. No markdown blocks (```), no extra explanations.")
    return "\n".join(lines)

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


def agent_main(args):
    facts = inspect_dataset(args.train, args.val)
    prompt = build_model_generation_prompt(facts, additional_instructions=args.instructions)

    print("Calling LLM to generate code and dependencies...")
    llm_output = build_langgraph_pipeline(prompt)

    if not llm_output:
        print("✗ OpenAI call failed")
        return

    print("✓ Obtained response from LangGraph pipeline")

    # Parse output into separate files
    files = parse_dual_output(llm_output)

    # Write both files
    write_files(files)

    print("\nDone! Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run model: python model.py --train <train> --val <val>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training split (dir or csv)")
    parser.add_argument("--val", required=True, help="Path to validation split (dir or csv)")
    parser.add_argument("--output", default="model.py", help="Output file for the generated model code")
    parser.add_argument("--instructions", default="", help="Extra instructions to include in prompt")
    args = parser.parse_args()
    agent_main(args)