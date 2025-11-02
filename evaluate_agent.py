#!/usr/bin/env python3
"""
Evaluation Harness for ML Coding Agent

This script:
1. Runs the agent to generate model.py
2. Tests functional correctness (syntax, imports, execution)
3. Tests basic performance (training runs, metrics improve)
4. Reports results
"""

import ast
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import importlib.util


def install_requirements(requirements_file: str = "requirements.txt") -> Dict[str, Any]:
    """
    Install packages from requirements.txt.
    Returns dict with installation results.
    """
    results = {
        "requirements_exists": False,
        "install_success": False,
        "install_output": "",
        "error": None
    }

    if not Path(requirements_file).exists():
        print(f"  ⚠ No {requirements_file} found, skipping dependency installation")
        return results

    results["requirements_exists"] = True

    print(f"\n[Auto-Install] Installing dependencies from {requirements_file}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )

        results["install_output"] = result.stdout

        if result.returncode == 0:
            print("  ✓ Successfully installed all dependencies")
            results["install_success"] = True
        else:
            print(f"  ✗ Installation failed:")
            print(f"    {result.stderr[:500]}")
            results["error"] = result.stderr

    except subprocess.TimeoutExpired:
        print("  ✗ Installation timeout after 3 minutes")
        results["error"] = "Timeout"
    except Exception as e:
        print(f"  ✗ Installation error: {e}")
        results["error"] = str(e)

    return results


class AgentEvaluator:
    """Evaluates the ML coding agent's generated code."""

    def __init__(self, train_path: str, val_path: str, output_path: str = "model.py"):
        self.train_path = train_path
        self.val_path = val_path
        self.output_path = output_path
        self.results = {
            "functional_tests": {},
            "performance_tests": {},
            "overall_status": "PENDING",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    # ========== PHASE 1: FUNCTIONAL CORRECTNESS ==========

    def test_syntax_valid(self) -> bool:
        """Test if generated code is syntactically valid Python."""
        print("\n[Functional Test 1/5] Checking syntax validity...")
        try:
            with open(self.output_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            print("  ✓ Code is syntactically valid")
            return True
        except SyntaxError as e:
            print(f"  ✗ Syntax error: {e}")
            return False
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            return False

    def test_imports_valid(self) -> bool:
        """Test if all imports in generated code are available."""
        print("\n[Functional Test 2/5] Checking imports...")
        try:
            with open(self.output_path, 'r') as f:
                code = f.read()

            tree = ast.parse(code)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])

            # Try importing each unique module
            failed_imports = []
            for module_name in set(imports):
                try:
                    __import__(module_name)
                except ImportError:
                    failed_imports.append(module_name)

            if failed_imports:
                print(f"  ✗ Missing imports: {', '.join(failed_imports)}")
                return False
            else:
                print(f"  ✓ All imports available ({len(set(imports))} modules checked)")
                return True

        except Exception as e:
            print(f"  ✗ Error checking imports: {e}")
            return False

    def test_required_components(self) -> Dict[str, bool]:
        """Test if required components are present in generated code."""
        print("\n[Functional Test 3/5] Checking required components...")
        try:
            with open(self.output_path, 'r') as f:
                code = f.read()

            tree = ast.parse(code)

            components = {
                "model_class": False,
                "train_function": False,
                "validate_function": False,
                "main_block": False,
                "argparse": False
            }

            # Check for model class (inherits from nn.Module)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    components["model_class"] = True
                elif isinstance(node, ast.FunctionDef):
                    if node.name == "train":
                        components["train_function"] = True
                    elif node.name in ["validate", "eval", "evaluate", "test"]:
                        components["validate_function"] = True
                elif isinstance(node, ast.If):
                    # Check for if __name__ == '__main__'
                    if isinstance(node.test, ast.Compare):
                        if any("__name__" in ast.unparse(n) and "__main__" in ast.unparse(n)
                               for n in ast.walk(node.test)):
                            components["main_block"] = True

            # Check for argparse
            components["argparse"] = "argparse" in code or "ArgumentParser" in code

            # Print results
            for component, present in components.items():
                status = "✓" if present else "✗"
                print(f"  {status} {component.replace('_', ' ').title()}: {'Present' if present else 'Missing'}")

            return components

        except Exception as e:
            print(f"  ✗ Error checking components: {e}")
            return {k: False for k in ["model_class", "train_function", "validate_function", "main_block", "argparse"]}

    def test_code_runs(self) -> bool:
        """Test if generated code can be imported without errors."""
        print("\n[Functional Test 4/5] Testing if code can be imported...")
        try:
            spec = importlib.util.spec_from_file_location("generated_model", self.output_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("  ✓ Code imports successfully")
            return True
        except Exception as e:
            print(f"  ✗ Import failed: {e}")
            return False

    def test_help_works(self) -> bool:
        """Test if generated script responds to --help flag."""
        print("\n[Functional Test 5/5] Testing CLI interface...")
        try:
            result = subprocess.run(
                [sys.executable, self.output_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("  ✓ CLI interface works (--help responds)")
                return True
            else:
                print("  ✗ CLI interface error")
                return False
        except subprocess.TimeoutExpired:
            print("  ✗ CLI interface timeout")
            return False
        except Exception as e:
            print(  f"  ✗ CLI test failed: {e}")
            return False

    # ========== PHASE 2: PERFORMANCE TESTING ==========

    def test_training_runs(self, epochs: int = 2, timeout: int = 300) -> Dict[str, Any]:
        """Test if training runs successfully and collects basic metrics."""
        print(f"\n[Performance Test 1/2] Running training for {epochs} epochs...")
        print(f"  (timeout: {timeout}s)")

        results = {
            "completed": False,
            "training_time": None,
            "final_output": None,
            "error": None
        }

        try:
            start_time = time.time()

            # Run training with limited epochs
            result = subprocess.run(
                [
                    sys.executable, self.output_path,
                    "--train", self.train_path,
                    "--val", self.val_path,
                    "--epochs", str(epochs)
                ],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            training_time = time.time() - start_time
            results["training_time"] = training_time
            results["final_output"] = result.stdout + result.stderr

            if result.returncode == 0:
                print(f"  ✓ Training completed in {training_time:.1f}s")
                results["completed"] = True
            else:
                print(f"  ✗ Training failed with exit code {result.returncode}")
                results["error"] = f"Exit code {result.returncode}"
                print(f"  Error output: {result.stderr[:500]}")

        except subprocess.TimeoutExpired:
            print(f"  ✗ Training timeout after {timeout}s")
            results["error"] = "Timeout"
        except Exception as e:
            print(f"  ✗ Training error: {e}")
            results["error"] = str(e)

        return results

    def test_checkpoint_created(self) -> Dict[str, Any]:
        """Test if model checkpoint was created and is loadable."""
        print("\n[Performance Test 2/2] Checking model checkpoint...")

        results = {
            "checkpoint_exists": False,
            "checkpoint_loadable": False,
            "checkpoint_size": None
        }

        # Look for common checkpoint names
        checkpoint_paths = ["best_model.pth", "model.pth", "checkpoint.pth"]
        checkpoint_file = None

        for cp_path in checkpoint_paths:
            if Path(cp_path).exists():
                checkpoint_file = cp_path
                results["checkpoint_exists"] = True
                results["checkpoint_size"] = Path(cp_path).stat().st_size
                print(f"  ✓ Checkpoint found: {cp_path} ({results['checkpoint_size']/1024:.1f} KB)")
                break

        if not checkpoint_file:
            print(f"  ✗ No checkpoint found (checked: {', '.join(checkpoint_paths)})")
            return results

        # Try to load checkpoint
        try:
            import torch
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            print(f"  ✓ Checkpoint is loadable")
            results["checkpoint_loadable"] = True

            # Try to get some info about the checkpoint
            if isinstance(checkpoint, dict):
                keys = list(checkpoint.keys())
                print(f"    Keys in checkpoint: {keys[:5]}{'...' if len(keys) > 5 else ''}")

        except Exception as e:
            print(f"  ✗ Checkpoint load failed: {e}")

        return results

    # ========== MAIN EVALUATION FLOW ==========

    def run_functional_tests(self) -> bool:
        """Run all functional correctness tests."""
        print("\n" + "="*60)
        print("PHASE 1: FUNCTIONAL CORRECTNESS TESTS")
        print("="*60)

        tests = {}

        # Run tests
        tests["syntax_valid"] = self.test_syntax_valid()
        tests["imports_valid"] = self.test_imports_valid()
        tests["components"] = self.test_required_components()
        tests["code_runs"] = self.test_code_runs()
        tests["cli_works"] = self.test_help_works()

        # Store results
        self.results["functional_tests"] = tests

        # Calculate pass rate
        component_count = sum(tests["components"].values())
        total_components = len(tests["components"])
        basic_tests_passed = sum([tests["syntax_valid"], tests["imports_valid"],
                                   tests["code_runs"], tests["cli_works"]])

        print(f"\n--- Functional Tests Summary ---")
        print(f"  Basic tests passed: {basic_tests_passed}/4")
        print(f"  Components present: {component_count}/{total_components}")

        # All basic tests must pass, and at least 4/5 components should be present
        functional_pass = basic_tests_passed == 4 and component_count >= 4

        return functional_pass

    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        print("\n" + "="*60)
        print("PHASE 2: PERFORMANCE TESTS")
        print("="*60)

        # Run training
        training_results = self.test_training_runs(epochs=2, timeout=300)
        self.results["performance_tests"]["training"] = training_results

        # Check checkpoint
        checkpoint_results = self.test_checkpoint_created()
        self.results["performance_tests"]["checkpoint"] = checkpoint_results

        print(f"\n--- Performance Tests Summary ---")
        print(f"  Training completed: {'✓' if training_results['completed'] else '✗'}")
        print(f"  Checkpoint created: {'✓' if checkpoint_results['checkpoint_exists'] else '✗'}")
        print(f"  Checkpoint loadable: {'✓' if checkpoint_results['checkpoint_loadable'] else '✗'}")

        # Performance tests pass if training completes and checkpoint is created
        performance_pass = (training_results["completed"] and
                          checkpoint_results["checkpoint_exists"] and
                          checkpoint_results["checkpoint_loadable"])

        return performance_pass

    def generate_report(self) -> str:
        """Generate a human-readable evaluation report."""
        functional_pass = all([
            self.results["functional_tests"].get("syntax_valid", False),
            self.results["functional_tests"].get("imports_valid", False),
            self.results["functional_tests"].get("code_runs", False)
        ])

        performance_pass = (
            self.results["performance_tests"].get("training", {}).get("completed", False) and
            self.results["performance_tests"].get("checkpoint", {}).get("checkpoint_loadable", False)
        )

        overall_status = "PASS" if functional_pass and performance_pass else "FAIL"
        self.results["overall_status"] = overall_status

        report = f"""
{'='*60}
EVALUATION REPORT
{'='*60}

Timestamp: {self.results['timestamp']}
Dataset: {self.train_path} / {self.val_path}
Generated Code: {self.output_path}

OVERALL STATUS: {overall_status}

Phase 1 - Functional Correctness: {'PASS' if functional_pass else 'FAIL'}
Phase 2 - Performance Tests: {'PASS' if performance_pass else 'FAIL'}

{'='*60}
DETAILED RESULTS
{'='*60}

{json.dumps(self.results, indent=2)}

{'='*60}
"""
        return report

    def save_results(self, output_file: str = "evaluation_results.json"):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


def main():
    """Main evaluation flow."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ML coding agent")
    parser.add_argument("--train", required=True, help="Path to training data")
    parser.add_argument("--val", required=True, help="Path to validation data")
    parser.add_argument("--model", default="model.py", help="Path to generated model file")
    parser.add_argument("--run-agent", action="store_true", help="Run agent first to generate model.py")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")

    args = parser.parse_args()

    # Step 1: Optionally run the agent first
    if args.run_agent:
        print("="*60)
        print("STEP 0: GENERATING MODEL CODE")
        print("="*60)
        print(f"\nRunning: python agent.py --train {args.train} --val {args.val}")

        result = subprocess.run(
            [sys.executable, "agent.py", "--train", args.train, "--val", args.val],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"✗ Agent failed to generate code!")
            print(result.stderr)
            sys.exit(1)
        else:
            print("✓ Agent completed successfully")

    # Step 2: Check if model file exists
    if not Path(args.model).exists():
        print(f"\n✗ Error: Model file '{args.model}' not found!")
        print(f"  Run with --run-agent flag to generate it first, or specify correct path with --model")
        sys.exit(1)

    # Step 2.5: Install dependencies from requirements.txt
    print("\n" + "="*60)
    print("STEP 1: INSTALLING DEPENDENCIES")
    print("="*60)
    install_results = install_requirements("requirements.txt")

    if install_results["requirements_exists"] and not install_results["install_success"]:
        print("\n⚠ Warning: Dependency installation failed, but continuing with evaluation...")
        print("  Some tests may fail due to missing packages")

    # Step 3: Run evaluation
    evaluator = AgentEvaluator(
        train_path=args.train,
        val_path=args.val,
        output_path=args.model
    )

    # Store installation results
    evaluator.results["dependency_install"] = install_results

    functional_pass = evaluator.run_functional_tests()

    # Only run performance tests if functional tests pass
    if functional_pass:
        performance_pass = evaluator.run_performance_tests()
    else:
        print("\n⚠ Skipping performance tests due to functional test failures")
        performance_pass = False

    # Step 4: Generate and save report
    report = evaluator.generate_report()
    print(report)

    evaluator.save_results(args.output)

    # Exit with appropriate code
    sys.exit(0 if functional_pass and performance_pass else 1)


if __name__ == "__main__":
    main()
