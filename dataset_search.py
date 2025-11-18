"""
Dataset Search Module

This module handles searching for datasets online based on natural language prompts.
It supports:
- Multiple data sources:
  * Kaggle API
  * HuggingFace Hub
  * OpenML
  * Papers with Code
  * UCI ML Repository
- LLM-based prompt parsing
- Semantic search using sentence embeddings (all-MiniLM-L6-v2)
- Hybrid ranking combining semantic similarity, keyword matching, and popularity metrics
- Embedding caching for performance
- Automatic deduplication across sources
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import requests
import numpy as np
from functools import lru_cache

load_dotenv()

# Global embedding model - lazy loaded
_embedding_model = None


def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight but effective model
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            print("⚠ sentence-transformers not installed. Install with: pip install sentence-transformers")
            print("  Falling back to keyword-only search")
            _embedding_model = None
    return _embedding_model


@lru_cache(maxsize=1000)
def get_text_embedding(text: str) -> Optional[np.ndarray]:
    """
    Get embedding for a text string with caching.

    Args:
        text: Input text to embed

    Returns:
        Embedding vector or None if model not available
    """
    model = get_embedding_model()
    if model is None:
        return None

    return model.encode(text, convert_to_numpy=True)


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1 (or 0 if embeddings unavailable)
    """
    emb1 = get_text_embedding(text1)
    emb2 = get_text_embedding(text2)

    if emb1 is None or emb2 is None:
        return 0.0

    # Compute cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Normalize to 0-1 range (cosine similarity is -1 to 1)
    return float((similarity + 1) / 2)


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


def search_openml_datasets(query: str, max_results: int = 10, task_type: str = "classification") -> List[Dict]:
    """
    Search for datasets on OpenML.

    WARNING: OpenML's list_datasets() can be slow as it loads metadata for thousands of datasets.
    This function includes a timeout and size limit for performance.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        task_type: Type of task ("classification", "regression", etc.)

    Returns:
        List of dataset metadata dictionaries
    """
    try:
        import openml
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("OpenML search timed out")

        # Set a 30-second timeout for the API call
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        try:
            # OPTIMIZATION: Use size parameter to limit initial load
            # Only load datasets with reasonable size (not too huge, not too small)
            datasets = openml.datasets.list_datasets(
                output_format='dataframe',
                size=10000,  # Limit to datasets with ~10K instances
            )
        finally:
            signal.alarm(0)  # Cancel the alarm

        if datasets is None or (hasattr(datasets, 'empty') and datasets.empty):
            return []

        # Limit search to first 1000 datasets to avoid excessive processing
        if len(datasets) > 1000:
            datasets = datasets.head(1000)

        # Filter by query in name or description
        query_lower = query.lower()
        keywords = query_lower.split()

        def matches_query(row):
            name = str(row.get('name', '')).lower()
            desc = str(row.get('description', '')).lower()
            tags = str(row.get('tag', '')).lower()

            # Check if any keyword appears in name, description, or tags
            for keyword in keywords:
                if keyword in name or keyword in desc or keyword in tags:
                    return True
            return False

        # Filter datasets
        filtered = datasets[datasets.apply(matches_query, axis=1)]

        # Sort by downloads/qualities and limit results
        if 'NumberOfDownloads' in filtered.columns:
            filtered = filtered.sort_values('NumberOfDownloads', ascending=False)

        results = []
        for idx, dataset in filtered.head(max_results).iterrows():
            did = int(idx)
            results.append({
                "source": "openml",
                "name": f"openml-{did}",
                "title": dataset.get('name', f'OpenML Dataset {did}'),
                "downloads": int(dataset.get('NumberOfDownloads', 0)) if 'NumberOfDownloads' in dataset else 0,
                "instances": int(dataset.get('NumberOfInstances', 0)) if 'NumberOfInstances' in dataset else 0,
                "features": int(dataset.get('NumberOfFeatures', 0)) if 'NumberOfFeatures' in dataset else 0,
                "url": f"https://www.openml.org/d/{did}",
                "description": str(dataset.get('description', ''))[:500],  # Truncate long descriptions
                "dataset_id": did,
            })

        return results

    except ImportError:
        print("OpenML not installed. Install with: pip install openml")
        return []
    except TimeoutError:
        print("OpenML search timed out (>30s) - skipping")
        return []
    except Exception as e:
        print(f"OpenML search failed: {e}")
        return []


def search_paperswithcode_datasets(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search for datasets on Papers with Code.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of dataset metadata dictionaries
    """
    try:
        # Papers with Code API v1 endpoint
        # List all datasets and filter client-side
        url = "https://paperswithcode.com/api/v1/datasets/"

        response = requests.get(url, timeout=15)
        response.raise_for_status()

        data = response.json()
        all_datasets = data.get('results', [])

        # Filter by query keywords
        query_lower = query.lower()
        keywords = query_lower.split()

        def matches_query(dataset):
            name = dataset.get('name', '').lower()
            description = dataset.get('description', '').lower()
            full_name = dataset.get('full_name', '').lower()

            for keyword in keywords:
                if keyword in name or keyword in description or keyword in full_name:
                    return True
            return False

        filtered = [d for d in all_datasets if matches_query(d)]

        results = []
        for dataset in filtered[:max_results]:
            results.append({
                "source": "paperswithcode",
                "name": dataset.get('id', ''),
                "title": dataset.get('name', ''),
                "url": f"https://paperswithcode.com{dataset.get('url', '')}",
                "description": dataset.get('description', ''),
                "full_name": dataset.get('full_name', ''),
                "paper_count": dataset.get('paper_count', 0),
                "homepage": dataset.get('homepage', ''),
            })

        return results

    except requests.RequestException as e:
        print(f"Papers with Code API unavailable: {e}")
        return []
    except Exception as e:
        print(f"Papers with Code search failed: {e}")
        return []


def search_uci_datasets(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search for datasets in UCI ML Repository.

    Note: UCI's API access may be limited. This function provides a curated
    list of common UCI datasets as a fallback.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of dataset metadata dictionaries
    """
    try:
        # Try the new UCI ML Repository API
        url = "https://archive.ics.uci.edu/api/search"
        params = {"q": query}

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            all_datasets = data.get('data', [])

            results = []
            for dataset in all_datasets[:max_results]:
                dataset_id = dataset.get('id', '')
                results.append({
                    "source": "uci",
                    "name": f"uci-{dataset_id}",
                    "title": dataset.get('name', ''),
                    "url": f"https://archive.ics.uci.edu/dataset/{dataset_id}",
                    "description": dataset.get('description', ''),
                    "area": dataset.get('area', ''),
                    "task": dataset.get('task', ''),
                    "num_instances": dataset.get('num_instances', 0),
                    "num_features": dataset.get('num_features', 0),
                    "year": dataset.get('year', 0),
                })

            return results
        else:
            # Fallback: Use a curated list of popular UCI datasets
            return _search_uci_fallback(query, max_results)

    except Exception as e:
        print(f"UCI API unavailable, using fallback dataset list")
        return _search_uci_fallback(query, max_results)


def _search_uci_fallback(query: str, max_results: int = 10) -> List[Dict]:
    """
    Fallback method using a curated list of popular UCI datasets.
    """
    # Curated list of popular UCI datasets
    uci_datasets = [
        {"id": "53", "name": "Iris", "description": "Classic iris flower dataset with 3 species", "area": "Life Sciences", "task": "Classification"},
        {"id": "186", "name": "Wine Quality", "description": "Wine quality ratings based on physicochemical tests", "area": "Business", "task": "Classification"},
        {"id": "19", "name": "Adult Income", "description": "Predict whether income exceeds $50K/yr based on census data", "area": "Social Science", "task": "Classification"},
        {"id": "80", "name": "Breast Cancer Wisconsin", "description": "Breast cancer diagnosis dataset", "area": "Health and Medicine", "task": "Classification"},
        {"id": "162", "name": "Heart Disease", "description": "Heart disease diagnosis dataset", "area": "Health and Medicine", "task": "Classification"},
        {"id": "10", "name": "Car Evaluation", "description": "Car acceptability based on features", "area": "Business", "task": "Classification"},
        {"id": "109", "name": "Wine", "description": "Wine recognition dataset", "area": "Business", "task": "Classification"},
        {"id": "17", "name": "Breast Cancer", "description": "Breast cancer classification", "area": "Health and Medicine", "task": "Classification"},
        {"id": "73", "name": "Mushroom", "description": "Mushroom classification - edible or poisonous", "area": "Biology", "task": "Classification"},
        {"id": "144", "name": "Statlog (German Credit Data)", "description": "Credit risk assessment", "area": "Business", "task": "Classification"},
        {"id": "891", "name": "Diabetes 130-US hospitals", "description": "Diabetes patient records", "area": "Health and Medicine", "task": "Classification"},
        {"id": "697", "name": "SMS Spam Collection", "description": "SMS spam vs ham classification", "area": "Computer Science", "task": "Classification"},
        {"id": "102", "name": "Abalone", "description": "Predict the age of abalone from physical measurements", "area": "Life Sciences", "task": "Regression"},
    ]

    # Filter by query
    query_lower = query.lower()
    keywords = query_lower.split()

    def matches_query(dataset):
        name = dataset['name'].lower()
        description = dataset['description'].lower()
        area = dataset['area'].lower()
        task = dataset['task'].lower()

        for keyword in keywords:
            if keyword in name or keyword in description or keyword in area or keyword in task:
                return True
        return False

    filtered = [d for d in uci_datasets if matches_query(d)]

    results = []
    for dataset in filtered[:max_results]:
        dataset_id = dataset['id']
        results.append({
            "source": "uci",
            "name": f"uci-{dataset_id}",
            "title": dataset['name'],
            "url": f"https://archive.ics.uci.edu/dataset/{dataset_id}",
            "description": dataset['description'],
            "area": dataset['area'],
            "task": dataset['task'],
            "num_instances": 0,
            "num_features": 0,
            "year": 0,
        })

    return results


def deduplicate_datasets(datasets: List[Dict]) -> List[Dict]:
    """
    Remove duplicate datasets based on title similarity.

    Args:
        datasets: List of dataset metadata

    Returns:
        Deduplicated list of datasets
    """
    if not datasets:
        return []

    # Use semantic similarity to detect duplicates
    unique_datasets = []
    seen_titles = []

    for dataset in datasets:
        title = dataset.get('title', '').lower().strip()

        if not title:
            unique_datasets.append(dataset)
            continue

        # Check if this dataset is similar to any we've already seen
        is_duplicate = False
        for seen_title in seen_titles:
            # Simple approach: check if titles are very similar
            # For more sophisticated dedup, could use embeddings
            if title == seen_title:
                is_duplicate = True
                break

            # Check if one title is a substring of another (likely same dataset)
            if title in seen_title or seen_title in title:
                # Only consider duplicate if titles are very similar in length
                if abs(len(title) - len(seen_title)) < 10:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_datasets.append(dataset)
            seen_titles.append(title)

    removed = len(datasets) - len(unique_datasets)
    if removed > 0:
        print(f"  Removed {removed} duplicate dataset(s)")

    return unique_datasets


def rank_datasets(datasets: List[Dict], requirements: Dict) -> List[Dict]:
    """
    Rank datasets based on relevance to requirements using both semantic similarity and keyword matching.

    Args:
        datasets: List of dataset metadata
        requirements: Parsed requirements from prompt

    Returns:
        Sorted list of datasets (highest score first)
    """
    keywords = [k.lower() for k in requirements.get('keywords', [])]
    search_query = requirements.get('description', '')

    def score_dataset(dataset: Dict) -> float:
        score = 0.0

        title = dataset.get('title', '').lower()
        description = dataset.get('description', '').lower()

        # --- Semantic Similarity (Primary Signal) ---
        # Combine title and description for better context
        dataset_text = f"{title}. {description}"

        # Compute semantic similarity with the search query
        semantic_sim = compute_semantic_similarity(search_query, dataset_text)
        # Weight semantic similarity heavily (0-20 points)
        score += semantic_sim * 20.0

        # Also check semantic similarity with individual keywords
        keyword_text = ' '.join(keywords)
        keyword_sim = compute_semantic_similarity(keyword_text, dataset_text)
        score += keyword_sim * 10.0

        # --- Keyword Matching (Secondary Signal) ---
        # Still useful for exact matches
        for keyword in keywords:
            if keyword in title:
                score += 3.0
            if keyword in description:
                score += 1.5

        # --- Popularity Metrics (Tertiary Signal) ---
        # Boost by popularity but cap to avoid overwhelming semantic scores
        if dataset['source'] == 'kaggle':
            score += min(dataset.get('vote_count', 0) / 100, 3.0)
            score += min(dataset.get('download_count', 0) / 1000, 3.0)
        elif dataset['source'] == 'huggingface':
            score += min(dataset.get('likes', 0) / 10, 3.0)
            score += min(dataset.get('downloads', 0) / 1000, 3.0)
        elif dataset['source'] == 'openml':
            score += min(dataset.get('downloads', 0) / 1000, 3.0)
        elif dataset['source'] == 'paperswithcode':
            score += min(dataset.get('paper_count', 0) / 10, 3.0)
        elif dataset['source'] == 'uci':
            # UCI datasets are generally well-curated, give a small boost
            score += 2.0

        # Store individual components for debugging
        dataset['relevance_score'] = score
        dataset['semantic_score'] = semantic_sim * 20.0 + keyword_sim * 10.0
        dataset['keyword_score'] = sum(3.0 if kw in title else 1.5 if kw in description else 0.0 for kw in keywords)

        return score

    # Score and sort datasets
    for dataset in datasets:
        score_dataset(dataset)

    return sorted(datasets, key=lambda x: x.get('relevance_score', 0), reverse=True)


def search_datasets(prompt: str, max_results: int = 5, sources: Optional[List[str]] = None) -> Tuple[Dict, List[Dict]]:
    """
    Main function to search for datasets based on a natural language prompt.

    Args:
        prompt: Natural language description of the desired ML task
        max_results: Maximum number of results to return
        sources: Optional list of sources to search. If None, searches all sources.
                 Options: ['kaggle', 'huggingface', 'openml', 'paperswithcode', 'uci']

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

    # Default to fast/reliable sources if none specified
    # OpenML is slow (loads all datasets), so excluded by default
    if sources is None:
        sources = ['kaggle', 'huggingface', 'paperswithcode', 'uci']

    all_datasets = []

    # Search Kaggle
    if 'kaggle' in sources:
        print(f"\n🔎 Searching Kaggle...")
        kaggle_results = search_kaggle_datasets(requirements['description'], max_results=max_results)
        print(f"  Found {len(kaggle_results)} results")
        all_datasets.extend(kaggle_results)

    # Search HuggingFace
    if 'huggingface' in sources:
        print(f"\n🔎 Searching HuggingFace...")
        hf_results = search_huggingface_datasets(
            requirements['description'],
            max_results=max_results,
            data_type=requirements['data_type']
        )
        print(f"  Found {len(hf_results)} results")
        all_datasets.extend(hf_results)

    # Search OpenML
    if 'openml' in sources:
        print(f"\n🔎 Searching OpenML...")
        openml_results = search_openml_datasets(
            requirements['description'],
            max_results=max_results,
            task_type=requirements['task_type']
        )
        print(f"  Found {len(openml_results)} results")
        all_datasets.extend(openml_results)

    # Search Papers with Code
    if 'paperswithcode' in sources:
        print(f"\n🔎 Searching Papers with Code...")
        pwc_results = search_paperswithcode_datasets(requirements['description'], max_results=max_results)
        print(f"  Found {len(pwc_results)} results")
        all_datasets.extend(pwc_results)

    # Search UCI ML Repository
    if 'uci' in sources:
        print(f"\n🔎 Searching UCI ML Repository...")
        uci_results = search_uci_datasets(requirements['description'], max_results=max_results)
        print(f"  Found {len(uci_results)} results")
        all_datasets.extend(uci_results)

    print(f"\n📦 Total datasets found: {len(all_datasets)}")

    # Deduplicate datasets
    all_datasets = deduplicate_datasets(all_datasets)

    # Rank results
    ranked_datasets = rank_datasets(all_datasets, requirements)

    print(f"\n📈 Top {min(len(ranked_datasets), max_results)} datasets:")
    for i, dataset in enumerate(ranked_datasets[:max_results], 1):
        print(f"\n  {i}. [{dataset['source'].upper()}] {dataset['title']}")
        print(f"     Total Score: {dataset.get('relevance_score', 0):.2f}")
        print(f"       ↳ Semantic: {dataset.get('semantic_score', 0):.2f} | Keyword: {dataset.get('keyword_score', 0):.2f}")
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
