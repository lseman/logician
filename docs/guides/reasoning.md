---
title: Reasoning Modes
description: Tree of Thoughts, SSR, Reflexion, Auto-CoT, and other reasoning strategies.
---

# Reasoning Modes

Logician supports multiple structured reasoning strategies. Choose the mode that fits your task complexity.

## Available reasoners

| Reasoner | Best for | Description |
|---|---|---|
| `cot` | Simple tasks | Standard chain-of-thought |
| `auto_cot` | Moderate tasks | Automatic reasoning depth selection |
| `reflexion` | Complex debugging | Self-reflection and revision |
| `tot` | Architecture decisions | Tree of Thoughts exploration |
| `ssr` | Code generation | Self-subjective reasoning |
| `got` | Multi-step problems | Graph of Thoughts |
| `best_of_n` | Quality-critical tasks | Multiple attempts, best result |
| `in_context_cot` | Few-shot learning | Demonstrations in context |
| `self_consistency` | Ambiguous tasks | Majority vote across attempts |

## Configuration

```json
{
  "reasoning": {
    "mode": "reflexion",
    "maxIterations": 10,
    "temperature": 0.7
  }
}
```

## Reasoning depth

The `thinkingLevel` setting controls how deeply the agent reasons:

| Level | Description |
|---|---|
| `low` | Quick responses, minimal reasoning |
| `medium` | Balanced reasoning and speed |
| `high` | Deep analysis, more iterations |
| `full` | Maximum reasoning depth |

## Switching at runtime

Change reasoning mode during a session:

```
/reasoning mode tot
/reasoning depth high
```

## When to use each

- **Simple fixes** → `cot` or `auto_cot`
- **Bug investigation** → `reflexion`
- **Architecture changes** → `tot` or `got`
- **Critical code** → `best_of_n`
- **Ambiguous requirements** → `self_consistency`
