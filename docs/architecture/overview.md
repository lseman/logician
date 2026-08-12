---
title: System Overview
description: High-level architecture of the Logician system.
---

# System Overview

Logician is built on four core packages that form a layered architecture.

## Package layers

```mermaid
graph LR
  subgraph "Presentation"
    A[TUI Package]
  end
  subgraph "Agent"
    B[Coding Agent Package]
  end
  subgraph "Core"
    C[Agent Core Package]
  end
  subgraph "Capabilities"
    D[Agent Capabilities Package]
  end

  A --> B
  B --> C
  B --> D
  C --> E[LLM Backend]
  D --> E
```

## Package details

### agent-core

The foundation layer. Handles:
- LLM backend (OpenAI-compatible HTTP client)
- Agent loop execution
- Configuration management
- Session persistence (SQLite)
- Compaction (context window management)
- Hook system
- Tool registry and execution

Execution durability is owned by the [Run Kernel](/architecture/run-kernel), a
versioned event ledger used for replay, task-spanning budgets, fencing, and tool
effect recovery.

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

## Data flow

```
User Input → TUI → Coding Agent → Agent Core → LLM Backend
                                                                ↓
User Display ← TUI ← Coding Agent ← Agent Core ← LLM Response ←
```
