---
title: System Overview
description: High-level architecture of the Logician system.
---

# System Overview

Logician is a TypeScript monorepo whose packages form a layered runtime.

## Package layers

```mermaid
graph LR
  subgraph Presentation
    A["@logician/tui"]
  end
  subgraph Orchestration
    B["@logician/coding-agent"]
  end
  subgraph Runtime
    C["@logician/agent-core"]
    D["@logician/agent-capabilities"]
  end
  subgraph DataAndEvaluation["Data and evaluation"]
    F["memory + rag"]
    G["autoresearch + agent-eval"]
  end
  A --> B
  B --> C
  B --> D
  B --> F
  D --> G
  C --> E[LLM Backend]
  D --> E
```

## Package details

### agent-core

The foundation layer. Handles:
- LLM backend (OpenAI-compatible HTTP client)
- Agent loop execution
- Provider-facing runtime configuration
- Append-only agent session journal
- Compaction (context window management)
- Hook system
- Tool registry and execution

Execution durability is owned by the [Run Kernel](/architecture/run-kernel), a
versioned event ledger used for replay, task-spanning budgets, fencing, and tool
effect recovery.

The evidence and invariants behind the current runtime boundaries are recorded
in [Runtime Design Decisions](/architecture/modernization).

### coding-agent

The orchestration layer. Handles:
- System prompt construction
- Skills loading and activation
- Session management (bookmarks, branching)
- Trust model (permission modes)
- MCP server integration
- Prompt guidelines and context files

### agent-capabilities

The intelligence layer. Handles:
- Reasoners (ToT, SSR, Reflexion, etc.)
- Subagent delegation
- Todo/task management
- EOH (Evolution of Hints) engine
- Tool selection and execution strategies

### tui

The presentation layer. Handles:
- Terminal UI rendering
- Input handling
- Streaming output
- State management
- Layout and theming

### memory and rag

Workspace-scoped durable memory and document/repository retrieval, with hybrid
ranking, provenance, context budgets, and component evaluation.
Memory claims also use gated lifecycles, executable validity predicates, and
outcome-linked shadow learning; see [Evolving memory](./evolving-memory.md).

### autoresearch and agent-eval

Measured experiment loops and independently graded coding-task trials. Agent
evaluation treats repository state and executable checks as authoritative; an
agent's own completion claim is retained only as diagnostic evidence.

## Data flow

```mermaid
flowchart LR
    User --> TUI --> Coding["Coding agent"] --> Core["Agent core"] --> Provider
    Provider --> Core
    Core --> Tools
    Tools --> Core
    Core --> Coding --> TUI --> User
```
