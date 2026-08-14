---
name: eoh-evolve
description: Run EoH evolution loops — initialize, run generations, and monitor heuristic evolution. Use when asked to evolve a heuristic, run EoH generations, or start an evolution loop.
---

# EoH Evolution Loop

Run autonomous heuristic evolution using the EoH framework (arxiv 2401.02051). This skill handles the evolution loop: initialize a problem, run generations of evolution, and track progress.

## When to Use

- "Run EoH on this problem"
- "Evolve a heuristic for X"
- "Start an evolution loop"
- "Run more generations of EoH"

## Workflow

### 1. Initialize Evolution

Call the `init_evolution` tool with:
- `name`: Problem name (e.g., "bin-packing")
- `description`: Detailed problem description for the LLM
- `functionSignature`: Required function signature
- `populationSize`: Number of heuristics in population (default: 10)
- `maxGenerations`: Max generations to run (0 = unlimited)
- `direction`: "lower" or "higher" (fitness direction)

This writes the config to `.eoh/log.jsonl` and activates EoH mode.

### 2. Run Generations

Call the `run_generation` tool repeatedly. Each generation:
- Applies all 5 operators (E1, E2, M1, M2, M3)
- Evaluates all candidate heuristics
- Selects the best N for the next population
- Logs results to `.eoh/log.jsonl`

The tool returns:
- Generation number
- Population statistics (best/mean/worst fitness)
- Number of candidates evaluated
- LLM call count

### 3. Monitor Convergence

After each generation, check:
- **Best fitness**: Is it improving?
- **Mean fitness**: Is the population improving overall?
- **Stale generations**: 3+ generations with no improvement suggests convergence

When converged, call `stop_evolution` to end the session.

### 4. Get Results

Call `get_best` to retrieve the best heuristic found:
- Its thought (natural language explanation)
- Its code (the actual function)
- Fitness score and generation

## Evolution Loop Pattern

```
1. init_evolution(problem, config)
2. Loop:
   a. run_generation()
   b. Check if converged (3+ generations no improvement)
   c. If not converged, go to step 2a
3. get_best() to retrieve the winner
4. stop_evolution()
```

## Tips

- **Start with a small population** (6-10) for quick iterations
- **Set maxGenerations** to avoid infinite loops (e.g., 20-50)
- **Monitor the dashboard** for visual progress tracking
- **Use higher temperature** (0.8) for more diverse exploration
- **If stuck**, try increasing population size or adjusting the problem description

## Operators Applied Each Generation

| Operator | Description | Parents Used |
|----------|-------------|--------------|
| E1 Diversity | Explore diverse solutions | N parents |
| E2 Convergence | Refine toward best solutions | N parents |
| M1 Improve | Direct improvement | 1 parent (best) |
| M2 Tune | Fine-tune parameters | 1 parent (best) |
| M3 Simplify | Reduce complexity | 1 parent (best) |

## File-Based EoH

For Python files with `# EOH-BEGIN` / `# EOH-END` markers, the file itself defines the problem. The heuristic in the marked region is evolved, and `evaluate(heuristic)` computes fitness.
