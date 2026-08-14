---
name: eoh-dashboard
description: Monitor and visualize EoH evolution progress. Use when checking evolution status, viewing the best heuristic, exporting the dashboard, or analyzing evolution results.
---

# EoH Dashboard & Monitoring

Monitor, visualize, and analyze EoH evolution progress. Provides status checks, best heuristic retrieval, dashboard export, and result analysis.

## When to Use

- "Check EoH status"
- "Show the best heuristic"
- "Export the EoH dashboard"
- "What's the current evolution progress?"

## Status Check

Call `get_status()` to see:
- Running state (true/false)
- Current generation number
- Total LLM calls made
- Population size
- Best/mean/worst fitness
- Session run count
- Mode status (ON/OFF)

## Best Heuristic

Call `get_best()` to retrieve:
- Fitness score
- Generation number
- Operator that produced it
- Natural language thought (explanation of the idea)
- Code (the actual function)

This is the primary output — the best heuristic found so far.

## Dashboard Export

Call `export_dashboard()` to:
1. Generate an HTML dashboard from the JSONL log
2. Start a local SSE-based server
3. Open the dashboard in the browser

The dashboard provides:
- Live updates via Server-Sent Events
- Full results table with all generations
- Fitness trend visualization
- Best heuristic highlighting
- Population statistics

## Result Analysis

The JSONL log at `.eoh/log.jsonl` contains all evolution data:

```jsonl
{"type": "eoh_config", "name": "bin-packing", "populationSize": 10, ...}
{"run": 1, "thought": "...", "code": "...", "fitness": 0.85, "generation": 1, ...}
{"run": 2, "thought": "...", "code": "...", "fitness": 0.92, "generation": 2, ...}
```

Each line is either:
- **Config entry**: `{type: "eoh_config", name, populationSize, maxGenerations, bestDirection}`
- **Run entry**: `{run, thought, code, fitness, generation, createdBy, parentIds, status, description, timestamp, segment}`

## Session Management

### Stop Evolution
Call `stop_evolution()` to:
- Send stop signal to the engine
- Turn off EoH mode
- Reset generation counter

### Clear Session
Call `clear()` to:
- Delete the session log
- Turn off EoH mode
- Reset all state

## Widget Summary

For persistent status display, use `get_widget_summary()`:

```typescript
interface EohWidgetSummary {
  active: boolean;
  name: string | null;
  populationSize: number;
  generation: number;
  bestFitness: number | null;
  meanFitness: number | null;
  worstFitness: number | null;
  totalLLMCalls: number;
  running: { description: string; elapsedMs: number } | null;
  maxGenerations: number | null;
}
```

## Dashboard Data

For full-screen display, use `get_dashboard_data()`:

```typescript
interface EohDashboardData {
  summary: EohWidgetSummary | null;
  rows: EohDashboardRow[];
}

interface EohDashboardRow {
  run: number;
  thought: string;
  fitness: number;
  fitnessFormatted: string;
  generation: number;
  createdBy: string;
  parentIds: string[];
  status: "keep" | "discard" | "crash";
  description: string;
  timestamp: number;
  isBest: boolean;
}
```

## Tips

- **Check status regularly** during long evolutions to track progress
- **Export the dashboard** when you want to share results or do deep analysis
- **Use the widget summary** for persistent status display in the UI
- **Monitor LLM calls** to manage costs — each generation makes 5N LLM calls
- **Save the best heuristic** when convergence is reached
