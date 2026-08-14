---
name: eoh-problem
description: Define and configure EoH evolution problems. Use when setting up a new EoH problem, defining function signatures, or creating evaluation functions for heuristic evolution.
---

# EoH Problem Definition

Define the problem that EoH will evolve heuristics for. A well-defined problem is critical — the LLM's output quality depends heavily on how clearly the problem is specified.

## When to Use

- "Define an EoH problem for X"
- "Set up a new evolution problem"
- "Create an evaluation function for EoH"

## Problem Components

### 1. Name

A short, descriptive identifier for the problem.

```
name: "Online Bin Packing"
```

### 2. Description

A detailed description that will be fed to the LLM. Include:
- What the problem is
- What constraints exist
- What constitutes a good solution
- Any domain-specific knowledge

```
description: """
Given items of various sizes (0 < size ≤ 1) arriving online, pack them 
into bins of capacity 1.0 using a heuristic function. The heuristic 
selects which existing open bin to place the current item in, or opens 
a new bin.

Goal: Minimize the number of bins used.
"""
```

### 3. Function Signature

The exact function signature the LLM must produce. This is used for:
- Validation of generated code
- Template for the LLM's output

```
functionSignature: "def select_bin(item_size: float, bins: list[float]) -> int:"
```

### 4. Instances

Test cases used to evaluate heuristic fitness. These should be:
- Representative of real-world inputs
- Diverse enough to prevent overfitting
- Small enough for fast evaluation

```
instances: [
  [0.4, 0.3, 0.5, 0.2, 0.7],      // Small case
  [0.1, 0.9, 0.5, 0.3, 0.8, 0.2],  // Mixed sizes
  // ... more instances
]
```

### 5. Evaluation Function

A function that scores a heuristic on an instance. Returns a scalar:
- Higher = better (or lower, depending on direction)
- Must be deterministic and fast

```
evaluateInstance: (fnCode: string, instance: unknown) => Promise<number>
```

## File-Based Problems (Python)

For Python files, the problem is defined in the file itself:

```python
"""Maximize the score over a fixed integer dataset."""

# EOH-BEGIN
def heuristic(value: int) -> int:
    return value
# EOH-END

def evaluate(heuristic) -> float:
    return sum(heuristic(value) for value in [1, 2, 3])
```

The file between `# EOH-BEGIN` and `# EOH-END` is the evolvable region. The `evaluate(heuristic)` function after it computes fitness.

## Tips for Good Problem Definitions

1. **Be specific about constraints**: What are the allowed inputs/outputs?
2. **Include edge cases**: Empty inputs, single items, extreme values
3. **Provide context**: Why does this problem matter? What's the real-world use case?
4. **Define success clearly**: What makes one heuristic better than another?
5. **Keep instances small**: Evaluation speed matters — each heuristic is tested on all instances each generation
6. **Use realistic data**: Synthetic data that matches real distributions works best

## Common Problem Types

### Algorithmic Heuristics
- Bin packing, scheduling, routing
- Function signature: `def heuristic(inputs) -> solution`

### Parameter Tuning
- Hyperparameter optimization
- Function signature: `def heuristic(params) -> score`

### Pattern Recognition
- Classification, detection rules
- Function signature: `def heuristic(features) -> label`

## Integration with Evolution

Once defined, the problem is passed to `init_evolution` which:
1. Writes the config to `.eoh/log.jsonl`
2. Seeds the initial population via LLM
3. Begins the evolution loop

The problem description becomes part of every LLM prompt during evolution, so invest time making it thorough and clear.
