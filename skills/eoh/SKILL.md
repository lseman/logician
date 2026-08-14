---
name: eoh
description: Evolution of Heuristics (EoH) — arxiv 2401.02051. Autonomous heuristic evolution via LLM-driven operators. Use when asked to evolve a heuristic, run EoH, or optimize an algorithmic function through generations.
---

# Evolution of Heuristics (EoH)

Evolve algorithmic heuristics autonomously using the EoH framework (arxiv 2401.02051). An LLM acts as the evolutionary operator, generating candidate heuristics that are evaluated for fitness and selected across generations.

## Core Concepts

- **Heuristic**: A function (thought + code) that solves a problem instance. Each heuristic has a fitness score.
- **Population**: A set of N heuristics. After each generation, the best N survive.
- **Operators**: Five evolutionary operators from the EoH paper:
  - **E1 Diversity**: Generate diverse heuristics from multiple parents
  - **E2 Convergence**: Refine toward the best solutions
  - **M1 Improve**: Improve the single best heuristic
  - **M2 Tune**: Fine-tune parameters of the best heuristic
  - **M3 Simplify**: Simplify the best heuristic (reduce complexity)
- **Generation**: One full cycle of applying all 5 operators to produce candidates, then selecting the next population.
- **Problem**: Defines the task, function signature, instances, and evaluation function.

## Session Structure

EoH sessions live in `.eoh/` directory:

| File | Purpose |
|------|---------|
| `.eoh/log.jsonl` | Append-only log of all generations and heuristics |
| `.eoh/prompt.md` | Optional: custom rules and constraints for the LLM |
| `.eoh/config.json` | Optional: max generations override |

## Workflow

### 1. Define the Problem

The problem must specify:
- **name**: Short identifier
- **description**: Full problem description for the LLM
- **functionSignature**: Required function signature the LLM must produce
- **instances**: Problem instances for evaluation
- **evaluateInstance**: Function that scores a heuristic on an instance

For Python files with EoH regions (`# EOH-BEGIN` / `# EOH-END`), the file itself defines the problem — the heuristic function is in the marked region and `evaluate(heuristic)` computes fitness.

### 2. Initialize Evolution

Set up the evolution session with:
- Problem definition (name, description, function signature)
- Population size (default: 10)
- Max generations (0 = unlimited)
- Direction: "lower" or "higher" fitness

### 3. Run Generations

Each generation:
1. Applies all 5 operators × population size = 5N candidate heuristics
2. Evaluates each candidate on all problem instances
3. Selects the top N heuristics for the next generation

The LLM generates both a natural language **thought** (explaining the idea) and **code** (the actual function).

### 4. Monitor Progress

Track:
- Best fitness per generation
- Population statistics (best/worst/mean)
- LLM call count
- Convergence (no improvement for 3+ generations)

### 5. Extract Results

Get the best heuristic found, including its thought (explanation) and code.

## File-Based EoH (Python)

For Python files, EoH uses a special region format:

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

## Operators in Detail

| Operator | Parents | Purpose |
|----------|---------|---------|
| E1 Diversity | N parents | Generate diverse solutions exploring the search space |
| E2 Convergence | N parents | Refine toward promising regions |
| M1 Improve | 1 parent | Directly improve the best heuristic |
| M2 Tune | 1 parent | Fine-tune parameters/strategy |
| M3 Simplify | 1 parent | Reduce code complexity while maintaining fitness |

## Tips

- **Start with a good problem description**. The LLM's output quality depends heavily on how well the problem is specified.
- **Use more generations** for complex problems. The EoH paper shows improvement continues beyond 10 generations.
- **Larger populations** explore more of the search space but cost more LLM calls.
- **Watch for convergence**: when 3+ generations show no improvement, the population may have plateaued.
- **M3 Simplify** is valuable for producing clean, maintainable code — not just high fitness.

## Related Skills

- `/autoresearch` — General-purpose autonomous optimization loop (broader scope, not heuristic-specific)
