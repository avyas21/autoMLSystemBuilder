import os

os.environ["OPENAI_API_KEY"] = "<REDACTED>"

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Any
import openai
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any

# ---------- Configuration ----------
OPENAI_MODEL = "gpt-4o"  # replace with desired chat model available to you
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
    Build a clear prompt for ChatGPT to generate a ready-to-run model.py file.
    The assistant should return only valid Python code in the final message.
    """
    lines = []
    lines.append("You are to generate a single Python file named `model.py` that includes:")
    lines.append("- A PyTorch model class (nn.Module) appropriate for the dataset.")
    lines.append("- A `train()` function that trains the model, with configurable hyperparameters.")
    lines.append("- A `validate()` function that evaluates the model on a validation dataset and reports metrics.")
    lines.append("- An `if __name__ == '__main__'` block that demonstrates how to use the train/validate functions with argparse.")
    lines.append("")
    lines.append("Important constraints:")
    lines.append("- The output MUST be valid, runnable Python. Do not include text explanations, only the file contents.")
    lines.append("- Use standard libraries and PyTorch. Keep external dependency usage minimal.")
    lines.append("- Save the trained model to `best_model.pth` when validation metric improves.")
    lines.append("- Include comments/docstrings for clarity but avoid long prose.")
    lines.append("")
    lines.append("Dataset facts (infer from the provided paths):")
    lines.append(json.dumps(facts, indent=2))
    lines.append("")
    if facts.get("split_type") == "image_folder":
        lines.append("Use a simple torchvision-based Dataset + transforms. Use CrossEntropyLoss for classification.")
        lines.append("Set input channels = {channels}, example size = {example_size}, num_classes = {num_classes}.".format(**facts))
    elif facts.get("split_type") == "csv":
        lines.append("Assume tabular data in CSV. Provide a simple PyTorch Dataset that reads CSV with pandas and returns (features_tensor, target_tensor).")
        if facts.get("task") == "classification":
            lines.append("Treat as classification with n_classes = {}.".format(facts.get("n_classes", "unknown")))
        else:
            lines.append("Treat as regression.")
    else:
        lines.append("Dataset type unknown; produce a flexible template that the user can adapt: simple MLP for tabular and a small CNN for images with clear TODOs.")
    if additional_instructions:
        lines.append("")
        lines.append("Additional instructions:")
        lines.append(additional_instructions)
    # final directive
    lines.append("")
    lines.append("Return only the contents of the file `model.py` — do not include any markdown, file markers, or explanation.")
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
def write_model_file(contents: str, output_path: str = "model.py") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(contents)
    print(f"Wrote {output_path}")


def agent_main(args):
    facts = inspect_dataset(args.train, args.val)
    prompt = build_model_generation_prompt(facts, additional_instructions=args.instructions)

    code_text = None

    code_text = build_langgraph_pipeline(prompt)
    if code_text:
        print("Obtained code from LangGraph pipeline.")
    else:
        print("OpenAI call has failed")

    if code_text.strip().startswith("```"):
        # remove common triple-backtick fences
        # find the first line with ``` and remove surrounding fences
        first = code_text.find("```")
        # remove leading fence
        code_body = code_text.split("```", 2)
        if len(code_body) >= 3:
            code_text = code_body[2]
        else:
            # fallback: remove first fence only
            code_text = code_text.replace("```python", "").replace("```", "")

    write_model_file(code_text, output_path=args.output)

    print("Done. You can run the generated model with: python model.py --train <train> --val <val> --test <test>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training split (dir or csv)")
    parser.add_argument("--val", required=True, help="Path to validation split (dir or csv)")
    parser.add_argument("--output", default="model.py", help="Output file for the generated model code")
    parser.add_argument("--instructions", default="", help="Extra instructions to include in prompt")
    args = parser.parse_args()
    agent_main(args)