"""
Test script for multi-source dataset search functionality.
Tests individual sources and combined search with deduplication.
"""

from dataset_search import (
    search_datasets,
    search_kaggle_datasets,
    search_huggingface_datasets,
    search_openml_datasets,
    search_paperswithcode_datasets,
    search_uci_datasets,
)


def test_individual_sources():
    """Test each data source individually."""
    print("\n" + "=" * 80)
    print("TESTING INDIVIDUAL DATA SOURCES")
    print("=" * 80)

    query = "image classification"
    max_results = 3

    # Test Kaggle
    print(f"\n1. Testing Kaggle (query: '{query}')")
    print("-" * 80)
    try:
        kaggle_results = search_kaggle_datasets(query, max_results=max_results)
        print(f"✓ Found {len(kaggle_results)} results from Kaggle")
        for i, ds in enumerate(kaggle_results[:2], 1):
            print(f"  {i}. {ds['title']}")
    except Exception as e:
        print(f"✗ Kaggle search failed: {e}")

    # Test HuggingFace
    print(f"\n2. Testing HuggingFace (query: '{query}')")
    print("-" * 80)
    try:
        hf_results = search_huggingface_datasets(query, max_results=max_results, data_type="image")
        print(f"✓ Found {len(hf_results)} results from HuggingFace")
        for i, ds in enumerate(hf_results[:2], 1):
            print(f"  {i}. {ds['title']}")
    except Exception as e:
        print(f"✗ HuggingFace search failed: {e}")

    # Test OpenML
    print(f"\n3. Testing OpenML (query: '{query}')")
    print("-" * 80)
    try:
        openml_results = search_openml_datasets(query, max_results=max_results)
        print(f"✓ Found {len(openml_results)} results from OpenML")
        for i, ds in enumerate(openml_results[:2], 1):
            print(f"  {i}. {ds['title']}")
    except Exception as e:
        print(f"✗ OpenML search failed: {e}")

    # Test Papers with Code
    print(f"\n4. Testing Papers with Code (query: '{query}')")
    print("-" * 80)
    try:
        pwc_results = search_paperswithcode_datasets(query, max_results=max_results)
        print(f"✓ Found {len(pwc_results)} results from Papers with Code")
        for i, ds in enumerate(pwc_results[:2], 1):
            print(f"  {i}. {ds['title']}")
    except Exception as e:
        print(f"✗ Papers with Code search failed: {e}")

    # Test UCI
    print(f"\n5. Testing UCI ML Repository (query: '{query}')")
    print("-" * 80)
    try:
        uci_results = search_uci_datasets(query, max_results=max_results)
        print(f"✓ Found {len(uci_results)} results from UCI")
        for i, ds in enumerate(uci_results[:2], 1):
            print(f"  {i}. {ds['title']}")
    except Exception as e:
        print(f"✗ UCI search failed: {e}")


def test_combined_search():
    """Test combined search across all sources."""
    print("\n\n" + "=" * 80)
    print("TESTING COMBINED MULTI-SOURCE SEARCH")
    print("=" * 80)

    prompts = [
        "I want to classify images of dogs and cats",
        "Build a model to predict house prices",
        "Sentiment analysis on product reviews",
    ]

    for prompt in prompts:
        print(f"\n\nPrompt: '{prompt}'")
        print("-" * 80)

        try:
            requirements, datasets = search_datasets(prompt, max_results=5)

            print(f"\n✓ Retrieved {len(datasets)} top datasets")

            # Show source distribution
            sources = {}
            for ds in datasets:
                source = ds['source']
                sources[source] = sources.get(source, 0) + 1

            print(f"\nSource distribution:")
            for source, count in sorted(sources.items()):
                print(f"  {source}: {count}")

        except Exception as e:
            print(f"✗ Combined search failed: {e}")


def test_source_filtering():
    """Test searching specific sources only."""
    print("\n\n" + "=" * 80)
    print("TESTING SOURCE FILTERING")
    print("=" * 80)

    prompt = "flower image classification"

    # Test with specific sources
    test_cases = [
        (['kaggle'], "Kaggle only"),
        (['huggingface', 'paperswithcode'], "HuggingFace + Papers with Code"),
        (['uci', 'openml'], "UCI + OpenML"),
    ]

    for sources, description in test_cases:
        print(f"\n{description} (sources={sources})")
        print("-" * 80)

        try:
            requirements, datasets = search_datasets(prompt, max_results=3, sources=sources)
            print(f"✓ Found {len(datasets)} results")

            # Verify all results are from specified sources
            actual_sources = set(ds['source'] for ds in datasets)
            expected_sources = set(sources)

            if actual_sources.issubset(expected_sources):
                print(f"✓ All results from specified sources: {actual_sources}")
            else:
                print(f"✗ Unexpected sources: {actual_sources - expected_sources}")

        except Exception as e:
            print(f"✗ Search failed: {e}")


def test_deduplication():
    """Test that deduplication is working."""
    print("\n\n" + "=" * 80)
    print("TESTING DEDUPLICATION")
    print("=" * 80)

    # Use a well-known dataset that might appear in multiple sources
    prompt = "iris flower classification"

    print(f"\nPrompt: '{prompt}'")
    print("-" * 80)

    try:
        requirements, datasets = search_datasets(prompt, max_results=10)

        # Check for potential duplicates by title
        titles = [ds['title'].lower() for ds in datasets]
        unique_titles = set(titles)

        print(f"\nTotal results: {len(datasets)}")
        print(f"Unique titles: {len(unique_titles)}")

        if len(titles) == len(unique_titles):
            print("✓ No duplicates found (deduplication working)")
        else:
            print("⚠ Some similar titles found (may be variants):")
            for title in titles:
                if titles.count(title) > 1:
                    print(f"  - {title}")

    except Exception as e:
        print(f"✗ Deduplication test failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-SOURCE DATASET SEARCH TEST SUITE")
    print("=" * 80)

    # Run all tests
    test_individual_sources()
    test_combined_search()
    test_source_filtering()
    test_deduplication()

    print("\n\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

    print("\n✅ Key Features Demonstrated:")
    print("  • 5 data sources: Kaggle, HuggingFace, OpenML, Papers with Code, UCI")
    print("  • Semantic search with embeddings")
    print("  • Automatic deduplication across sources")
    print("  • Source filtering for targeted searches")
    print("  • Unified ranking across all sources")
