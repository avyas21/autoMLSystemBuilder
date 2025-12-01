#!/usr/bin/env python3
"""
Agent GPA Evaluation Framework - PAPER-ALIGNED VERSION

Implements the exact 5-metric framework from the research paper:
arXiv:2510.08847 - "What Is Your Agent's GPA?"

METRICS (as specified in paper):
1. Goal Fulfillment (GF) - 25%
2. Plan Quality (PQ) - 20% [includes tool selection]
3. Plan Adherence (PA) - 15%
4. Execution Efficiency (EE) - 20% [includes tool calling correctness]
5. Logical Consistency (LC) - 20%

CHANGES FROM evaluate_gpa_full.py:
- Merged Tool Selection (TS) into Plan Quality (PQ)
- Merged Tool Calling (TC) into Execution Efficiency (EE)
- Updated weights to paper-aligned distribution
- Consolidated from 7 judges to 5 metrics

Reference: https://arxiv.org/abs/2510.08847
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()


class AgentGPAPaperAligned:
    """
    Paper-aligned 5-metric GPA evaluation framework using LLM-as-a-Judge
    """

    def __init__(self, results_dir: str = "results", model: str = "gpt-4o-mini"):
        self.results_dir = Path(results_dir)
        self.datasets = self._discover_datasets()
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        self.evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "framework": "Agent GPA (Paper-Aligned 5 Metrics)",
            "reference": "arXiv:2510.08847",
            "model": model,
            "datasets": {}
        }

    def _discover_datasets(self) -> List[str]:
        """Find all evaluated datasets"""
        if not self.results_dir.exists():
            return []
        return sorted([d.name for d in self.results_dir.iterdir() if d.is_dir()])

    def _safe_json_parse(self, response_content: str) -> Dict:
        """Safely parse JSON response with error handling"""
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            return {
                "overall_score": 5.0,
                "reasoning": "Failed to parse LLM response",
                "grade": "C",
                "normalized_score": 0.5
            }

    def _load_dataset_artifacts(self, dataset: str) -> Dict[str, Any]:
        """Load all artifacts for a dataset"""
        dataset_dir = self.results_dir / dataset

        artifacts = {
            "dataset": dataset,
            "scores": None,
            "best_model_code": None
        }

        scores_path = dataset_dir / "scores.json"
        if scores_path.exists():
            with open(scores_path) as f:
                artifacts["scores"] = json.load(f)

        best_model_path = dataset_dir / "best_model.py"
        if best_model_path.exists():
            with open(best_model_path) as f:
                artifacts["best_model_code"] = f.read()

        return artifacts

    # ========== METRIC 1: GOAL FULFILLMENT (GF) ==========
    def evaluate_goal_fulfillment(self, artifacts: Dict) -> Dict[str, Any]:
        """
        PAPER METRIC 1: Goal Fulfillment (GF)

        Did the agent successfully accomplish the user's stated goal?

        Evaluation criteria:
        - Task completion (did it create a working classifier?)
        - Performance quality (is accuracy reasonable?)
        - Goal relevance (does output match user's intent?)
        """
        print(f"\n  [GF] Evaluating Goal Fulfillment...")

        dataset = artifacts["dataset"]
        scores = artifacts["scores"]
        code = artifacts["best_model_code"]

        if not scores or not code:
            return {"score": 0.0, "reasoning": "Missing artifacts", "grade": "F", "normalized_score": 0.0}

        accuracies = [m.get("accuracy", 0) for m in scores.values()]
        best_accuracy = max(accuracies) if accuracies else 0

        prompt = f"""You are an expert evaluator assessing an AutoML agent's Goal Fulfillment.

**USER GOAL**: Build an image classification model for the {dataset} dataset

**AGENT OUTPUT**:
- Best Accuracy Achieved: {best_accuracy:.2f}%
- Number of iterations: {len(scores)}
- Model code exists: {'Yes' if code else 'No'}

**CODE SNIPPET** (first 50 lines):
{chr(10).join(code.split(chr(10))[:50])}

**EVALUATION CRITERIA**:
1. Task Completion (0-4): Did it create a working classifier?
2. Performance Quality (0-3): Is the accuracy reasonable for this dataset?
3. Goal Relevance (0-3): Does the output match the user's intent?

**INSTRUCTIONS**:
Score each criterion, then provide:
1. Overall score (0-10)
2. Brief reasoning (2-3 sentences)
3. Grade (A/B/C/D/F)

Respond in JSON format:
{{"task_completion": X, "performance_quality": X, "goal_relevance": X, "overall_score": X, "reasoning": "...", "grade": "X"}}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._safe_json_parse(response.content)
        result["normalized_score"] = result["overall_score"] / 10.0

        print(f"     Score: {result['overall_score']}/10 ({result['grade']}) - {result['reasoning'][:80]}...")
        return result

    # ========== METRIC 2: PLAN QUALITY (PQ) - INCLUDES TOOL SELECTION ==========
    def evaluate_plan_quality(self, artifacts: Dict) -> Dict[str, Any]:
        """
        PAPER METRIC 2: Plan Quality (PQ)

        Was the agent's plan well-designed and appropriate?
        INCLUDES: Tool selection (choosing right frameworks/architectures)

        Evaluation criteria:
        - Tool/framework selection (PyTorch vs TensorFlow, etc.)
        - Architecture selection (ResNet vs custom CNN, etc.)
        - Hyperparameter choices (learning rate, batch size, etc.)
        - Training strategy (augmentation, regularization, etc.)
        """
        print(f"\n  [PQ] Evaluating Plan Quality (includes Tool Selection)...")

        dataset = artifacts["dataset"]
        code = artifacts["best_model_code"]
        scores = artifacts["scores"]

        if not code:
            return {"score": 0.0, "reasoning": "No model code found", "grade": "F", "normalized_score": 0.0}

        prompt = f"""You are an expert evaluator assessing an AutoML agent's Plan Quality.

**TASK**: Design a classifier for {dataset} dataset

**AGENT'S PLAN** (model code):
{code[:1500]}

**PERFORMANCE TRAJECTORY**:
{json.dumps(scores, indent=2)}

**EVALUATION CRITERIA**:
1. Framework Selection (0-2): Is PyTorch/TensorFlow choice appropriate?
2. Architecture Selection (0-3): Is the model architecture suitable (ResNet, CNN, etc.)?
3. Hyperparameter Choices (0-3): Are hyperparameters well-chosen (LR, batch size, etc.)?
4. Training Strategy (0-2): Appropriate techniques (augmentation, regularization)?

**INSTRUCTIONS**:
Evaluate the overall quality of the agent's plan INCLUDING tool/framework choices.

Respond in JSON format:
{{"framework_selection": X, "architecture_selection": X, "hyperparameters": X, "training_strategy": X, "overall_score": X, "reasoning": "...", "grade": "X"}}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._safe_json_parse(response.content)
        result["normalized_score"] = result["overall_score"] / 10.0

        print(f"     Score: {result['overall_score']}/10 ({result['grade']}) - {result['reasoning'][:80]}...")
        return result

    # ========== METRIC 3: PLAN ADHERENCE (PA) ==========
    def evaluate_plan_adherence(self, artifacts: Dict) -> Dict[str, Any]:
        """
        PAPER METRIC 3: Plan Adherence (PA)

        Did the agent follow through on its plan consistently?

        Evaluation criteria:
        - Consistency (maintained coherent strategy?)
        - Progressive refinement (logical changes across iterations?)
        - Stability (or erratic behavior?)
        """
        print(f"\n  [PA] Evaluating Plan Adherence...")

        scores = artifacts["scores"]

        if not scores or len(scores) < 2:
            return {"score": 5.0, "reasoning": "Only one iteration, adherence N/A", "grade": "C", "normalized_score": 0.5}

        accuracies = [scores[f"iteration{i}"]["accuracy"] for i in range(1, len(scores) + 1)]

        prompt = f"""You are an expert evaluator assessing an AutoML agent's Plan Adherence.

**PERFORMANCE ACROSS ITERATIONS**:
{json.dumps(scores, indent=2)}

**ACCURACY TRAJECTORY**: {accuracies}

**EVALUATION CRITERIA**:
1. Consistency (0-4): Did the agent maintain a coherent strategy?
2. Progressive Refinement (0-3): Do changes show logical progression?
3. Stability (0-3): Or is behavior erratic/contradictory?

**INSTRUCTIONS**:
Assess whether the agent followed a consistent plan or made random changes.

Respond in JSON format:
{{"consistency": X, "progressive_refinement": X, "stability": X, "overall_score": X, "reasoning": "...", "grade": "X"}}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._safe_json_parse(response.content)
        result["normalized_score"] = result["overall_score"] / 10.0

        print(f"     Score: {result['overall_score']}/10 ({result['grade']}) - {result['reasoning'][:80]}...")
        return result

    # ========== METRIC 4: EXECUTION EFFICIENCY (EE) - INCLUDES TOOL CALLING ==========
    def evaluate_execution_efficiency(self, artifacts: Dict) -> Dict[str, Any]:
        """
        PAPER METRIC 4: Execution Efficiency (EE)

        Was execution efficient and correct?
        INCLUDES: Tool calling correctness (proper API usage, syntax)

        Evaluation criteria:
        - Tool calling correctness (syntax, API usage, parameters)
        - Iteration efficiency (found good solution quickly?)
        - Code quality (clean, maintainable?)
        - Resource awareness (appropriate batch sizes, model size, etc.)
        """
        print(f"\n  [EE] Evaluating Execution Efficiency (includes Tool Calling)...")

        code = artifacts["best_model_code"]
        scores = artifacts["scores"]

        if not scores:
            return {"score": 5.0, "reasoning": "No execution data", "grade": "C", "normalized_score": 0.5}

        accuracies = [s.get("accuracy", 0) for s in scores.values()]
        best_iter = accuracies.index(max(accuracies)) + 1

        # Check execution success
        successful_runs = sum(1 for s in scores.values() if s.get("accuracy") is not None)
        success_rate = successful_runs / len(scores) if scores else 0

        prompt = f"""You are an expert evaluator assessing an AutoML agent's Execution Efficiency.

**EXECUTION TRACE**:
{json.dumps(scores, indent=2)}

**CODE**:
{code[:2000] if code else "No code available"}

**EXECUTION STATS**:
- Best model found at iteration: {best_iter}/{len(scores)}
- Success rate: {success_rate:.1%}
- Lines of code: {len(code.split(chr(10))) if code else 0}

**EVALUATION CRITERIA**:
1. Tool Calling Correctness (0-3): Syntax correct? APIs used properly? Valid parameters?
2. Iteration Efficiency (0-3): Found good solution quickly?
3. Code Quality (0-2): Clean and maintainable?
4. Resource Awareness (0-2): Appropriate batch sizes, model complexity?

**INSTRUCTIONS**:
Evaluate execution quality INCLUDING correctness of tool usage.

Respond in JSON format:
{{"tool_calling": X, "iteration_efficiency": X, "code_quality": X, "resource_awareness": X, "overall_score": X, "reasoning": "...", "grade": "X"}}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._safe_json_parse(response.content)
        result["normalized_score"] = result["overall_score"] / 10.0

        print(f"     Score: {result['overall_score']}/10 ({result['grade']}) - {result['reasoning'][:80]}...")
        return result

    # ========== METRIC 5: LOGICAL CONSISTENCY (LC) ==========
    def evaluate_logical_consistency(self, artifacts: Dict) -> Dict[str, Any]:
        """
        PAPER METRIC 5: Logical Consistency (LC)

        Are actions logically grounded in prior context and errors?
        THIS IS THE MOST IMPORTANT METRIC PER THE PAPER!

        Evaluation criteria:
        - Error awareness (does agent understand what went wrong?)
        - Grounded changes (are fixes related to actual problems?)
        - Logical justification (can we trace decisions to observations?)
        """
        print(f"\n  [LC] Evaluating Logical Consistency...")

        scores = artifacts["scores"]
        code = artifacts["best_model_code"]

        if not scores or len(scores) < 2:
            return {"score": 5.0, "reasoning": "Only one iteration, consistency N/A", "grade": "C", "normalized_score": 0.5}

        accuracies = [s["accuracy"] for s in scores.values()]
        improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]

        prompt = f"""You are an expert evaluator assessing an AutoML agent's Logical Consistency.

**CRITICAL QUESTION**: Are the agent's refinements logically grounded in observed errors?

**PERFORMANCE TRAJECTORY**:
{json.dumps(scores, indent=2)}

**ACCURACY CHANGES**: {improvements}
(Positive = improvement, Negative = regression)

**FINAL CODE**:
{code[:1500] if code else "No code"}

**EVALUATION CRITERIA**:
1. Error Awareness (0-3): Does agent show understanding of what failed?
2. Grounded Changes (0-4): Are changes clearly linked to observed problems?
3. Logical Justification (0-3): Can we trace decisions to evidence?

**RED FLAGS**:
- Declining accuracy without obvious error diagnosis
- Random architectural changes
- No clear connection between failures and fixes

**INSTRUCTIONS**:
Evaluate whether the agent's actions show logical consistency and grounding.

Respond in JSON format:
{{"error_awareness": X, "grounded_changes": X, "justification": X, "overall_score": X, "reasoning": "...", "grade": "X"}}
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self._safe_json_parse(response.content)
        result["normalized_score"] = result["overall_score"] / 10.0

        print(f"     Score: {result['overall_score']}/10 ({result['grade']}) - {result['reasoning'][:80]}...")
        return result

    # ========== OVERALL GPA CALCULATION (PAPER-ALIGNED) ==========
    def calculate_overall_gpa(self, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate overall GPA from 5 paper-aligned metrics

        PAPER-ALIGNED WEIGHTING:
        - Goal Fulfillment (GF): 25%
        - Plan Quality (PQ): 20% [includes tool selection]
        - Plan Adherence (PA): 15%
        - Execution Efficiency (EE): 20% [includes tool calling]
        - Logical Consistency (LC): 20%

        Total: 100%
        """
        weights = {
            "GF": 0.25,   # 25% - Goal achievement
            "PQ": 0.20,   # 20% - Planning (includes tool selection)
            "PA": 0.15,   # 15% - Plan adherence
            "EE": 0.20,   # 20% - Execution (includes tool calling correctness)
            "LC": 0.20    # 20% - Logical consistency (emphasized in paper)
        }

        weighted_score = sum(
            metrics[metric]["normalized_score"] * weights[metric]
            for metric in weights.keys()
        )

        # Convert to 4.0 scale
        gpa_4_scale = weighted_score * 4.0

        return {
            "gpa_score": round(weighted_score, 3),
            "gpa_4_scale": round(gpa_4_scale, 2),
            "letter_grade": self._score_to_grade(weighted_score),
            "breakdown": {
                metric: {
                    "score": metrics[metric]["overall_score"],
                    "normalized": metrics[metric]["normalized_score"],
                    "grade": metrics[metric]["grade"],
                    "weight": weights[metric],
                    "contribution": round(metrics[metric]["normalized_score"] * weights[metric], 3)
                }
                for metric in weights.keys()
            },
            "weighting_note": "Paper-aligned: GF=25%, PQ=20% (incl. tools), PA=15%, EE=20% (incl. calling), LC=20%"
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert 0-1 score to letter grade"""
        if score >= 0.93:
            return "A"
        elif score >= 0.90:
            return "A-"
        elif score >= 0.87:
            return "B+"
        elif score >= 0.83:
            return "B"
        elif score >= 0.80:
            return "B-"
        elif score >= 0.77:
            return "C+"
        elif score >= 0.73:
            return "C"
        elif score >= 0.70:
            return "C-"
        elif score >= 0.67:
            return "D+"
        elif score >= 0.60:
            return "D"
        else:
            return "F"

    # ========== MAIN EVALUATION ==========
    def evaluate_dataset(self, dataset: str) -> Dict[str, Any]:
        """Run full 5-metric paper-aligned GPA evaluation for a dataset"""
        print(f"\n{'='*70}")
        print(f"Evaluating: {dataset.upper()}")
        print(f"{'='*70}")

        artifacts = self._load_dataset_artifacts(dataset)

        if not artifacts["scores"]:
            print(f"  ⚠️  No scores.json found, skipping...")
            return None

        # Run all 5 paper-aligned metrics
        metrics = {
            "GF": self.evaluate_goal_fulfillment(artifacts),
            "PQ": self.evaluate_plan_quality(artifacts),          # Includes Tool Selection
            "PA": self.evaluate_plan_adherence(artifacts),
            "EE": self.evaluate_execution_efficiency(artifacts),  # Includes Tool Calling
            "LC": self.evaluate_logical_consistency(artifacts)
        }

        # Calculate overall GPA
        gpa = self.calculate_overall_gpa(metrics)

        print(f"\n  {'='*66}")
        print(f"  Final GPA: {gpa['gpa_4_scale']}/4.0 ({gpa['letter_grade']})")
        print(f"  {'='*66}")

        # Print breakdown
        print(f"\n  Metric Breakdown (Paper-Aligned 5 Metrics):")
        for metric_name, metric_data in gpa["breakdown"].items():
            metric_full_names = {
                "GF": "Goal Fulfillment",
                "PQ": "Plan Quality (incl. Tool Selection)",
                "PA": "Plan Adherence",
                "EE": "Execution Efficiency (incl. Tool Calling)",
                "LC": "Logical Consistency"
            }
            print(f"    [{metric_name}] {metric_full_names[metric_name]}")
            print(f"         Score: {metric_data['score']}/10 ({metric_data['grade']})")
            print(f"         Weight: {metric_data['weight']:.0%} → Contributes: {metric_data['contribution']:.3f}")

        return {
            "dataset": dataset,
            "metrics": metrics,
            "gpa": gpa
        }

    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all datasets and generate comprehensive report"""
        print("\n" + "="*70)
        print(" "*10 + "AGENT GPA EVALUATION (PAPER-ALIGNED 5 METRICS)")
        print("="*70)
        print("\nEvaluating AutoML Agent using EXACT paper framework")
        print("Reference: Agent GPA Research Paper (arXiv:2510.08847)")
        print(f"\nDatasets found: {len(self.datasets)}")
        print(f"LLM Judge: {self.llm.model_name}")
        print("\nMETRICS (5 from paper):")
        print("  1. Goal Fulfillment (GF) - 25%")
        print("  2. Plan Quality (PQ) - 20% [includes Tool Selection]")
        print("  3. Plan Adherence (PA) - 15%")
        print("  4. Execution Efficiency (EE) - 20% [includes Tool Calling]")
        print("  5. Logical Consistency (LC) - 20%")

        all_results = []

        for dataset in self.datasets:
            result = self.evaluate_dataset(dataset)
            if result:
                all_results.append(result)
                self.evaluation_results["datasets"][dataset] = result

        # Generate summary
        self._print_summary(all_results)

        # Save results
        output_file = "agent_gpa_paper_aligned.json"
        with open(output_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        print(f"\n💾 Saved detailed results to: {output_file}")

        return self.evaluation_results

    def _print_summary(self, results: List[Dict]):
        """Print comprehensive summary table"""
        print("\n" + "="*70)
        print(" "*15 + "GPA SUMMARY REPORT (PAPER-ALIGNED)")
        print("="*70)

        if not results:
            print("No results to display")
            return

        # Create summary table
        summary_data = []
        for r in results:
            metrics = r["metrics"]
            summary_data.append({
                "Dataset": r["dataset"],
                "GF": f"{metrics['GF']['overall_score']:.1f}",
                "PQ": f"{metrics['PQ']['overall_score']:.1f}",
                "PA": f"{metrics['PA']['overall_score']:.1f}",
                "EE": f"{metrics['EE']['overall_score']:.1f}",
                "LC": f"{metrics['LC']['overall_score']:.1f}",
                "GPA": f"{r['gpa']['gpa_4_scale']}/4.0",
                "Grade": r['gpa']['letter_grade']
            })

        df = pd.DataFrame(summary_data)
        print("\n" + df.to_string(index=False))
        print("\nLegend:")
        print("  GF = Goal Fulfillment (25%)")
        print("  PQ = Plan Quality incl. Tool Selection (20%)")
        print("  PA = Plan Adherence (15%)")
        print("  EE = Execution Efficiency incl. Tool Calling (20%)")
        print("  LC = Logical Consistency (20%)")

        # Overall statistics
        avg_gpa = sum(r['gpa']['gpa_4_scale'] for r in results) / len(results)
        best_dataset = max(results, key=lambda x: x['gpa']['gpa_4_scale'])
        worst_dataset = min(results, key=lambda x: x['gpa']['gpa_4_scale'])

        print(f"\n📊 STATISTICS")
        print(f"  Average GPA: {avg_gpa:.2f}/4.0")
        print(f"  Best Performer: {best_dataset['dataset']} ({best_dataset['gpa']['gpa_4_scale']}/4.0)")
        print(f"  Needs Improvement: {worst_dataset['dataset']} ({worst_dataset['gpa']['gpa_4_scale']}/4.0)")

        # Metric-specific insights
        print(f"\n📈 METRIC INSIGHTS")
        metric_names = ["GF", "PQ", "PA", "EE", "LC"]
        metric_avgs = {}

        for metric in metric_names:
            scores = [r["metrics"][metric]["overall_score"] for r in results]
            metric_avgs[metric] = sum(scores) / len(scores)

        # Sort by average score
        sorted_metrics = sorted(metric_avgs.items(), key=lambda x: x[1])

        print("\n  Average Scores by Metric:")
        for metric, avg_score in sorted_metrics:
            bar = "█" * int(avg_score)
            print(f"    {metric}: {avg_score:.1f}/10 {bar}")

        # Identify weakest dimension
        weakest_metric = sorted_metrics[0][0]
        metric_full_names = {
            "GF": "Goal Fulfillment",
            "PQ": "Plan Quality (incl. Tool Selection)",
            "PA": "Plan Adherence",
            "EE": "Execution Efficiency (incl. Tool Calling)",
            "LC": "Logical Consistency"
        }

        print(f"\n⚠️  WEAKEST METRIC: {metric_full_names[weakest_metric]} ({weakest_metric})")
        print(f"    This is the primary area for improvement.")

        # Grade distribution
        grades = [r['gpa']['letter_grade'] for r in results]
        grade_counts = pd.Series(grades).value_counts().sort_index()
        print(f"\n📈 GRADE DISTRIBUTION")
        for grade, count in grade_counts.items():
            print(f"  {grade}: {'█' * count} ({count})")

        print("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate AutoML Agent using Paper-Aligned GPA Framework (5 Metrics)"
    )
    parser.add_argument("--results-dir", default="results",
                       help="Directory containing agent results")
    parser.add_argument("--model", default="gpt-4o-mini",
                       help="LLM model for judging")
    args = parser.parse_args()

    evaluator = AgentGPAPaperAligned(results_dir=args.results_dir, model=args.model)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
