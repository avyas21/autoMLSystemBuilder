"""
Test script to demonstrate semantic search improvements.
This shows how semantic similarity finds relevant datasets even without exact keyword matches.
"""

from dataset_search import compute_semantic_similarity, get_embedding_model

def test_semantic_vs_keyword():
    """Demonstrate semantic search superiority over keyword matching."""

    print("🧪 Testing Semantic Search vs Keyword Matching\n")
    print("=" * 70)

    # Load the model first
    print("\nLoading embedding model...")
    model = get_embedding_model()
    if model is None:
        print("❌ Failed to load embedding model. Please install sentence-transformers.")
        return

    print("✓ Model loaded successfully!\n")

    # Test cases showing semantic understanding
    test_cases = [
        {
            "query": "dog cat image classification dataset",
            "datasets": [
                "Dogs vs Cats - Binary Image Classification",  # High relevance - exact match
                "Pet Animal Recognition Dataset",  # High relevance - semantic match
                "Automobile Classification Images",  # Low relevance - different domain
                "Canine and Feline Photo Collection",  # High relevance - synonyms
            ]
        },
        {
            "query": "house price prediction regression",
            "datasets": [
                "Real Estate Price Forecasting Data",  # High relevance - semantic match
                "Housing Market Analysis Dataset",  # High relevance - semantic match
                "Stock Price Prediction",  # Medium relevance - different domain
                "Residential Property Valuation",  # High relevance - semantic match
            ]
        },
        {
            "query": "sentiment analysis movie reviews",
            "datasets": [
                "IMDB Film Review Opinions",  # High relevance - semantic match
                "Customer Product Feedback Sentiment",  # Medium relevance - similar task
                "Stock Market News Analysis",  # Low relevance - different domain
                "Twitter Emotion Classification",  # Medium relevance - similar task
            ]
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        datasets = test_case["datasets"]

        print(f"\nTest Case {i}:")
        print(f"Query: '{query}'")
        print(f"\nSemantic Similarity Scores:")
        print("-" * 70)

        scores = []
        for dataset in datasets:
            similarity = compute_semantic_similarity(query, dataset)
            scores.append((dataset, similarity))

            # Simple keyword matching for comparison
            query_words = set(query.lower().split())
            dataset_words = set(dataset.lower().split())
            keyword_overlap = len(query_words & dataset_words)

            print(f"  {dataset}")
            print(f"    Semantic Score: {similarity:.4f} | Keyword Matches: {keyword_overlap}")

        # Show ranking
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  Ranking by Semantic Similarity:")
        for rank, (dataset, score) in enumerate(scores, 1):
            print(f"    {rank}. {dataset} ({score:.4f})")

        print("=" * 70)


def test_embedding_cache():
    """Test that embedding caching is working."""
    print("\n\n🧪 Testing Embedding Cache\n")
    print("=" * 70)

    import time

    text = "This is a test sentence for caching embeddings"

    # First call - should compute
    start = time.time()
    compute_semantic_similarity(text, text)
    first_time = time.time() - start

    # Second call - should use cache
    start = time.time()
    compute_semantic_similarity(text, text)
    second_time = time.time() - start

    print(f"First call time:  {first_time:.4f}s (computed)")
    print(f"Second call time: {second_time:.4f}s (cached)")
    print(f"Speedup: {first_time/second_time:.2f}x")

    if second_time < first_time:
        print("✓ Caching is working!")
    else:
        print("⚠ Caching may not be working as expected")

    print("=" * 70)


if __name__ == "__main__":
    test_semantic_vs_keyword()
    test_embedding_cache()

    print("\n\n✅ All tests complete!")
    print("\nKey Benefits of Semantic Search:")
    print("  • Finds relevant datasets even with different terminology")
    print("  • Understands synonyms (e.g., 'canine' = 'dog', 'residence' = 'house')")
    print("  • Captures semantic meaning beyond keyword matching")
    print("  • Caching makes repeated searches fast")
