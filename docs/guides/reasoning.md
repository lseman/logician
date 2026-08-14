---
title: Reasoning and Inference
description: Provider defaults, thinking levels, inference presets, and structured reasoners.
---

# Reasoning and Inference

Logician exposes three separate controls. Keeping them distinct makes configuration predictable.

| Control | Purpose | Default |
|---|---|---|
| Inference mode | Sampling parameters sent to the provider | `none` (**Provider**) |
| Thinking level | Provider-specific reasoning effort | `off` |
| Structured reasoner | Optional advisory pre-reasoning strategy | `none` |

## Inference modes

`none` leaves provider sampling parameters untouched. Other presets include `auto`, `thinking-general`, `thinking-coding`, `instruct-general`, `instruct-reasoning`, `instruct-coding`, `deterministic`, `creative`, and `analytical`.

Open the selector with `/mode` or `Ctrl+M`/`Alt+M`. `/inference-mode-cycle` advances through presets.

## Thinking level

Supported values are `off`, `minimal`, `low`, `medium`, `high`, and `xhigh`. Availability depends on the selected model; Logician clamps unsupported levels when models change.

Use `/thinking <level>` or the settings overlay. Thinking level is independent of whether reasoning text is collapsed, summarized, or expanded in the transcript.

## Structured reasoners

Reasoners run before the ordinary tool-capable agent loop and provide advisory analysis. Available IDs are `ssr`, `tot`, `got`, `reflexion`, `self_consistency`, `best_of_n`, `auto_cot`, `in_context_cot`, and `cover`.

```json
{
  "reasoner": "reflexion",
  "reasonerConfig": {
    "maxTrials": 2
  }
}
```

Use `/reasoner <mode>` to select and persist a reasoner, or `/reasoner none` to disable it. Reasoners do not replace verification: the main loop still owns tools, edits, and completion.

## A practical starting point

Keep Provider mode, thinking off, and no structured reasoner until a task benefits from tighter sampling or additional deliberation. For an ambiguous design decision, try `tot`; for revision after a failed attempt, try `reflexion`; for output diversity with selection, try `best_of_n`.
