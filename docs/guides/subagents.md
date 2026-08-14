---
title: Subagents
description: Delegate bounded tasks to child agents and collect their results.
---

# Subagents

Subagents run self-contained tasks with isolated conversation context. They share the configured workspace unless a workflow explicitly creates a Git worktree, so delegation is not automatically file-isolated.

```mermaid
flowchart LR
    Parent[Parent agent] -->|spawn task| ChildA[Child agent A]
    Parent -->|spawn task| ChildB[Child agent B]
    ChildA -->|events and final result| Parent
    ChildB -->|events and final result| Parent
    Parent --> Workspace[(Shared workspace by default)]
    ChildA --> Workspace
    ChildB --> Workspace
```

## Good delegation boundaries

Delegate work that is concrete, independently verifiable, and unlikely to overlap another writer: a focused investigation, one test suite, a bounded component, or a source review. Keep tightly coupled edits in one agent.

## Configuration

Use the flat bridge settings rather than a nested `subagents` object:

```json
{
  "maxParallelAgents": 4
}
```

The parent controls concurrency. Individual agent definitions can set their own model and turn limits. Child results and lifecycle events flow back to the parent, which remains responsible for integration and final verification.

## Commands and tools

- `/spawn <task>` starts a child task from the TUI.
- `/agents` (when contributed by the active capability set) shows child state.
- Programmatic agents use the registered delegation tools to spawn, message, interrupt, or wait for children.

Do not assume a child committed changes or used a worktree unless the task explicitly required and verified that workflow.
