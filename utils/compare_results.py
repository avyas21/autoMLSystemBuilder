#!/usr/bin/env python3
"""Compare evaluation results across different datasets."""

import json
from pathlib import Path

def load_results(filepath):
    """Load evaluation results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(results):
    """Extract key metrics from results."""
    training = results.get('performance_tests', {}).get('training', {})
    output = training.get('final_output', '')

    # Parse final accuracy from output
    lines = output.strip().split('\n')
    final_acc = None
    for line in reversed(lines):
        if 'Val Acc:' in line:
            parts = line.split('Val Acc:')
            if len(parts) > 1:
                acc_str = parts[1].strip().replace('%', '')
                try:
                    final_acc = float(acc_str)
                    break
                except:
                    pass

    return {
        'training_time': training.get('training_time'),
        'final_accuracy': final_acc,
        'completed': training.get('completed', False),
        'functional_pass': all([
            results['functional_tests'].get('syntax_valid', False),
            results['functional_tests'].get('imports_valid', False),
            results['functional_tests'].get('code_runs', False)
        ])
    }

# Load results - check both old and new paths
datasets = {}

mnist_paths = ['evaluation_results_v3.json', 'examples/evaluation_results/mnist.json', '../examples/evaluation_results/mnist.json']
fashion_paths = ['evaluation_fashion_mnist.json', 'examples/evaluation_results/fashion_mnist.json', '../examples/evaluation_results/fashion_mnist.json']

for path in mnist_paths:
    if Path(path).exists():
        datasets['MNIST'] = load_results(path)
        break

for path in fashion_paths:
    if Path(path).exists():
        datasets['Fashion-MNIST'] = load_results(path)
        break

# Print comparison table
print("="*80)
print("AGENT PERFORMANCE COMPARISON")
print("="*80)
print()

if not datasets:
    print("No results found!")
    exit(1)

# Table header
print(f"{'Dataset':<20} {'Status':<10} {'Accuracy':<12} {'Time (s)':<12} {'Difficulty':<15}")
print("-"*80)

# Expected difficulty/baseline
difficulty = {
    'MNIST': 'Easy (>95%)',
    'Fashion-MNIST': 'Medium (>85%)'
}

for dataset_name, results in datasets.items():
    metrics = extract_metrics(results)

    status = "✓ PASS" if results['overall_status'] == 'PASS' else "✗ FAIL"
    acc = f"{metrics['final_accuracy']:.2f}%" if metrics['final_accuracy'] else "N/A"
    time = f"{metrics['training_time']:.1f}" if metrics['training_time'] else "N/A"
    diff = difficulty.get(dataset_name, 'Unknown')

    print(f"{dataset_name:<20} {status:<10} {acc:<12} {time:<12} {diff:<15}")

print("-"*80)
print()

# Detailed comparison
print("="*80)
print("DETAILED ANALYSIS")
print("="*80)
print()

for dataset_name, results in datasets.items():
    metrics = extract_metrics(results)
    print(f"\n{dataset_name}:")
    print(f"  Overall Status: {results['overall_status']}")
    print(f"  Functional Tests: {'PASS' if metrics['functional_pass'] else 'FAIL'}")
    print(f"  Training Completed: {'Yes' if metrics['completed'] else 'No'}")
    print(f"  Final Accuracy: {metrics['final_accuracy']:.2f}%" if metrics['final_accuracy'] else "  Final Accuracy: N/A")
    print(f"  Training Time: {metrics['training_time']:.1f}s" if metrics['training_time'] else "  Training Time: N/A")

    # Show training output
    training_output = results.get('performance_tests', {}).get('training', {}).get('final_output', '')
    if training_output:
        print(f"\n  Training Log:")
        for line in training_output.strip().split('\n'):
            print(f"    {line}")

print()
print("="*80)
print("KEY OBSERVATIONS")
print("="*80)

if 'MNIST' in datasets and 'Fashion-MNIST' in datasets:
    mnist_acc = extract_metrics(datasets['MNIST'])['final_accuracy']
    fashion_acc = extract_metrics(datasets['Fashion-MNIST'])['final_accuracy']

    if mnist_acc and fashion_acc:
        diff = mnist_acc - fashion_acc
        print(f"\n• Accuracy drop from MNIST to Fashion-MNIST: {diff:.2f}%")
        print(f"  - This is expected as Fashion-MNIST is more challenging")
        print(f"  - MNIST: Simple digits (0-9)")
        print(f"  - Fashion-MNIST: Clothing items with more complex patterns")

        if fashion_acc >= 85:
            print(f"\n✓ Agent performs well on Fashion-MNIST (>85% after 2 epochs)")
        else:
            print(f"\n⚠ Agent could improve on Fashion-MNIST (<85% after 2 epochs)")
            print(f"  Suggestions: More epochs, deeper network, data augmentation")

print()
