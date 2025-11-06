"""
Dataset Inspector - LLM-based intelligent dataset analysis

This module uses LLMs to:
1. Analyze dataset structure before conversion
2. Detect split feature/label files
3. Recommend merge strategies
4. Validate data quality
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
import json


def analyze_csv_structure(csv_path: Path, max_rows: int = 5) -> Dict[str, Any]:
    """Quick analysis of a CSV file structure."""
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)

        return {
            'filename': csv_path.name,
            'path': str(csv_path),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'sample_data': df.head(3).to_dict(orient='records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'num_rows_sampled': len(df)
        }
    except Exception as e:
        return {
            'filename': csv_path.name,
            'path': str(csv_path),
            'error': str(e)
        }


def build_dataset_analysis_prompt(csv_structures: List[Dict[str, Any]], directory_info: Dict[str, Any]) -> str:
    """Build prompt for LLM to analyze dataset structure."""

    prompt = f"""You are a dataset format analyzer. Analyze the following CSV files found in a downloaded dataset directory.

Dataset Directory: {directory_info.get('path', 'unknown')}
Total CSV files found: {len(csv_structures)}

CSV File Structures:
"""

    for i, csv_info in enumerate(csv_structures, 1):
        prompt += f"\n--- File {i}: {csv_info['filename']} ---\n"
        if 'error' in csv_info:
            prompt += f"ERROR: {csv_info['error']}\n"
        else:
            prompt += f"Columns ({csv_info['num_columns']}): {', '.join(csv_info['columns'][:10])}"
            if csv_info['num_columns'] > 10:
                prompt += f" ... and {csv_info['num_columns'] - 10} more"
            prompt += f"\nData types: {json.dumps(csv_info['dtypes'], indent=2)}\n"
            if csv_info['sample_data']:
                prompt += f"Sample row: {csv_info['sample_data'][0]}\n"

    prompt += """

Your task:
1. Determine the dataset format:
   - 'combined': Each CSV has both features and labels in one file (e.g., many columns with one being the label)
   - 'split_features_labels': Features and labels are in separate CSV files (e.g., one file with many columns for features, another with 1 column for labels)
   - 'unknown': Cannot determine

2. If split_features_labels:
   - Identify which files contain FEATURES (typically many columns, e.g., 784 columns for images)
   - Identify which files contain LABELS (typically 1 column with target values)
   - For train set: specify both features file AND labels file
   - For test set: specify both features file AND labels file
   - Set merge_required to TRUE (these files MUST be merged)

3. If combined:
   - Identify the train file (has both features and labels)
   - Identify the test file (has both features and labels)
   - Set merge_required to FALSE

4. Detect any issues:
   - Missing data
   - Inconsistent number of rows between features/labels
   - Invalid column names
   - File naming issues

Respond ONLY with valid JSON in this exact format:
{
  "format": "combined" | "split_features_labels" | "unknown",
  "train_features_file": "filename.csv or null",
  "train_labels_file": "filename.csv or null",
  "test_features_file": "filename.csv or null",
  "test_labels_file": "filename.csv or null",
  "train_combined_file": "filename.csv or null",
  "test_combined_file": "filename.csv or null",
  "merge_required": true | false,
  "warnings": ["list", "of", "warnings"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""

    return prompt


def llm_inspect_dataset_csvs(csv_files: List[Path], dataset_dir: str) -> Dict[str, Any]:
    """
    Use LLM to intelligently analyze CSV dataset structure.

    Args:
        csv_files: List of CSV file paths found in dataset
        dataset_dir: Root directory of the dataset

    Returns:
        Dict with analysis results and recommendations
    """
    print("\n🔍 Using LLM to analyze dataset structure...")

    # Analyze each CSV file structure
    csv_structures = []
    for csv_file in csv_files:
        structure = analyze_csv_structure(csv_file)
        csv_structures.append(structure)

    # Build prompt for LLM
    directory_info = {'path': dataset_dir}
    prompt = build_dataset_analysis_prompt(csv_structures, directory_info)

    # Call LLM
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        # Parse JSON response
        # Remove markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(content)

        print(f"✅ LLM Analysis complete:")
        print(f"   Format: {analysis['format']}")
        print(f"   Merge required: {analysis['merge_required']}")
        print(f"   Confidence: {analysis['confidence']}")

        if analysis['warnings']:
            print(f"   ⚠️  Warnings:")
            for warning in analysis['warnings']:
                print(f"      - {warning}")

        return analysis

    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        return {
            'format': 'unknown',
            'merge_required': False,
            'warnings': [f'LLM analysis failed: {str(e)}'],
            'confidence': 0.0
        }


def merge_feature_label_csvs(features_path: str, labels_path: str, output_path: str) -> None:
    """
    Merge separate feature and label CSV files.

    Args:
        features_path: Path to features CSV
        labels_path: Path to labels CSV
        output_path: Where to save merged CSV
    """
    print(f"\n🔄 Merging features and labels:")
    print(f"   Features: {Path(features_path).name}")
    print(f"   Labels: {Path(labels_path).name}")

    # Read both files
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)

    # Validate row counts match
    if len(features_df) != len(labels_df):
        raise ValueError(
            f"Row count mismatch: features={len(features_df)}, labels={len(labels_df)}"
        )

    # Merge: features + label column
    # Assume label column is the first (or only) column in labels CSV
    label_col_name = labels_df.columns[0]
    merged_df = features_df.copy()
    merged_df['label'] = labels_df[label_col_name]

    # Save
    merged_df.to_csv(output_path, index=False)

    print(f"✅ Merged {len(merged_df)} rows to {output_path}")
    print(f"   Columns: {len(merged_df.columns)} ({len(features_df.columns)} features + 1 label)")


if __name__ == "__main__":
    # Test with the Arabic dataset structure
    test_files = [
        "csvTrainImages 60k x 784.csv",
        "csvTrainLabel 60k x 1.csv",
        "csvTestImages 10k x 784.csv",
        "csvTestLabel 10k x 1.csv"
    ]

    print("This module provides LLM-based dataset inspection.")
    print("Import and use llm_inspect_dataset_csvs() in dataset_converter.py")
