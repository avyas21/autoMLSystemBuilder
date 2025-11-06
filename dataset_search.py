"""
Dataset Search Module

This module handles searching for datasets online based on natural language prompts.
It supports:
- Kaggle API integration
- HuggingFace datasets integration
- LLM-based prompt parsing
- Dataset ranking and selection
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests

load_dotenv()


def parse_dataset_requirements(prompt: str) -> Dict:
    """
    Use LLM to parse natural language prompt and extract dataset requirements.

    Args:
        prompt: Natural language description (e.g., "I want to create an image classifier for dogs and cats")

    Returns:
        Dict containing:
            - task_type: "classification" or "regression"
            - data_type: "image", "text", "tabular", etc.
            - keywords: List of search keywords
            - description: Refined description for dataset search
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    system_prompt = """You are a dataset requirement analyzer. Given a natural language prompt about a machine learning task,
extract the following information in JSON format:

{
  "task_type": "classification" or "regression",
  "data_type": "image", "text", "tabular", "audio", or "video",
  "keywords": ["keyword1", "keyword2", ...],
  "description": "A refined search query for finding datasets"
}

Examples:
Input: "I want to create an image classifier for dogs and cats"
Output: {
  "task_type": "classification",
  "data_type": "image",
  "keywords": ["dogs", "cats", "animals", "binary classification"],
  "description": "dog cat image classification dataset"
}

Input: "Build a model to predict house prices"
Output: {
  "task_type": "regression",
  "data_type": "tabular",
  "keywords": ["house", "prices", "real estate", "regression"],
  "description": "house price prediction dataset"
}

Only output valid JSON, nothing else."""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ])

    # Extract JSON from response
    content = response.content.strip()

    # Try to find JSON in the response
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    try:
        requirements = json.loads(content)
        return requirements
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response: {content}")
        raise e


def search_kaggle_datasets(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search for datasets on Kaggle using their API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of dataset metadata dictionaries
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # Search for datasets
        datasets = api.dataset_list(search=query)
        # Limit results manually
        datasets = list(datasets)[:max_results]

        results = []
        for dataset in datasets:
            results.append({
                "source": "kaggle",
                "name": dataset.ref,
                "title": dataset.title,
                "size": getattr(dataset, 'totalBytes', 0),
                "download_count": getattr(dataset, 'downloadCount', 0),
                "vote_count": getattr(dataset, 'voteCount', 0),
                "url": f"https://www.kaggle.com/datasets/{dataset.ref}",
                "description": getattr(dataset, 'subtitle', ''),
            })

        return results

    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return []
    except Exception as e:
        print(f"Kaggle search failed: {e}")
        return []


def search_huggingface_datasets(query: str, max_results: int = 10, data_type: str = "image") -> List[Dict]:
    """
    Search for datasets on HuggingFace Hub.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        data_type: Type of data ("image", "text", "tabular", etc.)

    Returns:
        List of dataset metadata dictionaries
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()

        # Map data types to HuggingFace task categories
        task_mapping = {
            "image": "image-classification",
            "text": "text-classification",
            "tabular": None
        }

        task = task_mapping.get(data_type)

        # Search datasets
        datasets = api.list_datasets(
            search=query,
            limit=max_results,
            task_categories=task if task else None
        )

        results = []
        for dataset in datasets:
            results.append({
                "source": "huggingface",
                "name": dataset.id,
                "title": dataset.id,
                "downloads": getattr(dataset, 'downloads', 0),
                "likes": getattr(dataset, 'likes', 0),
                "url": f"https://huggingface.co/datasets/{dataset.id}",
                "description": getattr(dataset, 'description', ''),
                "tags": getattr(dataset, 'tags', []),
            })

        return results

    except ImportError:
        print("HuggingFace Hub not installed. Install with: pip install huggingface-hub")
        return []
    except Exception as e:
        print(f"HuggingFace search failed: {e}")
        return []


def rank_datasets(datasets: List[Dict], requirements: Dict) -> List[Dict]:
    """
    Rank datasets based on relevance to requirements.

    Args:
        datasets: List of dataset metadata
        requirements: Parsed requirements from prompt

    Returns:
        Sorted list of datasets (highest score first)
    """
    keywords = [k.lower() for k in requirements.get('keywords', [])]

    def score_dataset(dataset: Dict) -> float:
        score = 0.0

        # Check title for keywords
        title = dataset.get('title', '').lower()
        for keyword in keywords:
            if keyword in title:
                score += 2.0

        # Check description for keywords
        description = dataset.get('description', '').lower()
        for keyword in keywords:
            if keyword in description:
                score += 1.0

        # Boost by popularity metrics
        if dataset['source'] == 'kaggle':
            score += min(dataset.get('vote_count', 0) / 100, 5.0)
            score += min(dataset.get('download_count', 0) / 1000, 5.0)
        elif dataset['source'] == 'huggingface':
            score += min(dataset.get('likes', 0) / 10, 5.0)
            score += min(dataset.get('downloads', 0) / 1000, 5.0)

        dataset['relevance_score'] = score
        return score

    # Score and sort datasets
    for dataset in datasets:
        score_dataset(dataset)

    return sorted(datasets, key=lambda x: x.get('relevance_score', 0), reverse=True)


def search_datasets(prompt: str, max_results: int = 5) -> Tuple[Dict, List[Dict]]:
    """
    Main function to search for datasets based on a natural language prompt.

    Args:
        prompt: Natural language description of the desired ML task
        max_results: Maximum number of results to return

    Returns:
        Tuple of (requirements dict, ranked list of datasets)
    """
    print(f"\n🔍 Parsing prompt: {prompt}")
    requirements = parse_dataset_requirements(prompt)

    print(f"\n📊 Extracted requirements:")
    print(f"  Task type: {requirements['task_type']}")
    print(f"  Data type: {requirements['data_type']}")
    print(f"  Keywords: {', '.join(requirements['keywords'])}")
    print(f"  Search query: {requirements['description']}")

    # Search both Kaggle and HuggingFace
    print(f"\n🔎 Searching Kaggle...")
    kaggle_results = search_kaggle_datasets(requirements['description'], max_results=max_results)
    print(f"  Found {len(kaggle_results)} results")

    print(f"\n🔎 Searching HuggingFace...")
    hf_results = search_huggingface_datasets(
        requirements['description'],
        max_results=max_results,
        data_type=requirements['data_type']
    )
    print(f"  Found {len(hf_results)} results")

    # Combine and rank results
    all_datasets = kaggle_results + hf_results
    ranked_datasets = rank_datasets(all_datasets, requirements)

    print(f"\n📈 Top {min(len(ranked_datasets), max_results)} datasets:")
    for i, dataset in enumerate(ranked_datasets[:max_results], 1):
        print(f"\n  {i}. [{dataset['source'].upper()}] {dataset['title']}")
        print(f"     Score: {dataset.get('relevance_score', 0):.2f}")
        print(f"     URL: {dataset['url']}")

    return requirements, ranked_datasets[:max_results]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search for datasets based on natural language prompts")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language description of ML task")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results")

    args = parser.parse_args()

    requirements, datasets = search_datasets(args.prompt, max_results=args.max_results)

    print(f"\n✅ Search complete! Found {len(datasets)} relevant datasets.")
