#!/usr/bin/env python3
"""
Test script for dataset search functionality

This script tests the prompt parsing without requiring Kaggle/HuggingFace credentials.
"""

from dataset_search import parse_dataset_requirements

def test_prompt_parsing():
    """Test various prompt parsing scenarios"""

    test_cases = [
        # {
        #     "prompt": "I want to create an image classifier for dogs and cats",
        #     "expected": {
        #         "task_type": "classification",
        #         "data_type": "image",
        #     }
        # },
        # {
        #     "prompt": "Build a model to predict house prices",
        #     "expected": {
        #         "task_type": "regression",
        #         "data_type": "tabular",
        #     }
        # },
        # {
        #     "prompt": "Sentiment analysis for movie reviews",
        #     "expected": {
        #         "task_type": "classification",
        #         "data_type": "text",
        #     }
        # },
        {
            "prompt": "Classify handwritten digits from 0 to 9",
            "expected": {
                "task_type": "classification",
                "data_type": "image",
            }
        },
    ]

    print("=" * 80)
    print("Testing Prompt Parsing")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'='*80}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'='*80}")
        print(f"\nPrompt: {test_case['prompt']}")

        try:
            result = parse_dataset_requirements(test_case['prompt'])

            print(f"\n✅ Parsing successful!")
            print(f"   Task type: {result['task_type']}")
            print(f"   Data type: {result['data_type']}")
            print(f"   Keywords: {', '.join(result['keywords'])}")
            print(f"   Search query: {result['description']}")

            # Validate expected values
            expected = test_case['expected']
            if result['task_type'] != expected['task_type']:
                print(f"\n⚠️  Warning: Expected task_type '{expected['task_type']}', got '{result['task_type']}'")

            if result['data_type'] != expected['data_type']:
                print(f"\n⚠️  Warning: Expected data_type '{expected['data_type']}', got '{result['data_type']}'")

        except Exception as e:
            print(f"\n❌ Parsing failed: {e}")

    print(f"\n\n{'='*80}")
    print("Testing Complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please create a .env file with your OpenAI API key:")
        print("   echo 'OPENAI_API_KEY=your-key-here' > .env")
        exit(1)

    print("\n🚀 Starting prompt parsing tests...\n")
    test_prompt_parsing()
    print("\n✅ All tests completed!\n")
