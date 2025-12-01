# Agent GPA Evaluation - Paper-Aligned Implementation

This directory contains the paper-aligned implementation of the Agent GPA evaluation framework from [arXiv:2510.08847](https://arxiv.org/abs/2510.08847).

## Quick Start

```bash
# Run paper-aligned evaluation
python evaluate_gpa_paper_aligned.py --results-dir results --model gpt-4o-mini
```

## File

**`evaluate_gpa_paper_aligned.py`**
- Paper-aligned 5-metric evaluation
- Matches arXiv:2510.08847 specification exactly
- LLM-as-a-judge (GPT-4o-mini)
- **Metrics**: GF (25%), PQ (20%), PA (15%), EE (20%), LC (20%)
- Cost: ~$0.10-0.50 per full evaluation

## The 5 Paper-Aligned Metrics

| Metric | Weight | What It Measures |
|--------|--------|------------------|
| **GF** - Goal Fulfillment | 25% | Did agent accomplish the task? |
| **PQ** - Plan Quality | 20% | Good architecture/tool choices? (includes Tool Selection) |
| **PA** - Plan Adherence | 15% | Consistent strategy across iterations? |
| **EE** - Execution Efficiency | 20% | Efficient execution & correct tool usage? (includes Tool Calling) |
| **LC** - Logical Consistency | 20% | Are changes error-grounded with clear reasoning? |

## Why 5 Metrics (Not 7)?

The paper (arXiv:2510.08847) specifies 5 core metrics. Previous implementations incorrectly used 7 judges by splitting:
- **Tool Selection (TS)** and **Tool Calling (TC)** as separate judges

This implementation correctly merges them:
- **Tool Selection** → part of **Plan Quality (PQ)** (choosing tools is planning)
- **Tool Calling** → part of **Execution Efficiency (EE)** (using tools correctly is execution)

### Weighting
```
GF (Goal Fulfillment):        25%
PQ (Plan Quality):            20%  [includes tool selection]
PA (Plan Adherence):          15%
EE (Execution Efficiency):    20%  [includes tool calling]
LC (Logical Consistency):     20%  [emphasized by paper]
```

## Your Current Results

**Average GPA**: 2.89/4.0 (C-)

### Strengths ✅
- **Goal Fulfillment**: 9.3/10 (A) - Creates working classifiers
- **Plan Quality**: 8.0/10 (B) - Good tool/architecture choices

### Critical Weakness ❌
- **Logical Consistency**: 4.7/10 (F) - No error diagnosis or grounded reasoning
- **Plan Adherence**: 6.0/10 (D+) - Performance often declines across iterations

### What This Means

You have a **"Lucky Agent"**:
- ✅ Achieves goals initially (9.3/10 GF)
- ❌ Cannot learn from mistakes (4.7/10 LC)
- ❌ Makes things worse in later iterations (6.0/10 PA)

**Example (CIFAR-10)**:
```
Iteration 1: 75.89% ← Good initial result!
Iteration 2: 72.31% ← Gets worse
Iteration 3: 68.45% ← Even worse!

Problem: No error diagnosis explaining WHY it failed
```

## How to Fix

### Priority 1: Add Error Diagnosis (Target: LC 8/10)

Modify your agent to log reasoning:

```python
# After each iteration, require:
{
  "iteration2": {
    "accuracy": 72.31,
    "diagnosis": "Loss increased 0.70→0.81, suggests LR too high",
    "changes": ["Reduced LR from 1e-3 to 1e-4"],
    "reasoning": "Lower LR should stabilize gradient updates",
    "expected": "Loss should decrease in next iteration"
  }
}
```

**Expected impact**: GPA 2.89 → 3.5 (C- → B+)

### Priority 2: Prevent Regressions

```python
if new_accuracy < best_accuracy:
    revert_to_best_model()
    try_different_approach()
```

### Priority 3: Add Data Augmentation

Add standard augmentations to all image datasets.

## Citation

If using this framework for research:

```bibtex
@article{jia2024agentgpa,
  title={What Is Your Agent's GPA? A Framework for Evaluating Agent Goal-Plan-Action Alignment},
  author={Jia, Allison Sihan and Huang, Daniel and Vytla, Nikhil and Choudhury, Nirvika and Mitchell, John C and Datta, Anupam},
  journal={arXiv preprint arXiv:2510.08847},
  year={2024}
}
```

## Reference

Paper: https://arxiv.org/abs/2510.08847
