---
title: Overview
description: Understand the architecture, components, and design philosophy of Logician.
---

# Overview

Logician is a local-first coding agent built on a modular architecture. This page explains how the pieces fit together.

## Design philosophy

- **Local-first runtime** — sessions and tool execution stay local; the model endpoint can be local or hosted.
- **Streaming-first** — provider text and tool progress are visible as they arrive.
- **Safe by default** — edits use strict exact-text matching with CRLF/BOM preservation. Permission modes control how aggressively the agent acts.
- **Extensible** — skills (`SKILL.md` files), plugins, and MCP servers extend capabilities without code changes.

## Component architecture

```mermaid
graph TB
  subgraph TUI
    A[Terminal UI]
  end
  subgraph Agent
    B[Coding Agent]
    C[Agent Core]
  end
  subgraph Capabilities
    D[Reasoners]
    E[Subagents]
    F[Todo & Task Mgmt]
    G[EOH Engine]
  end
  subgraph Storage
    H[Session DB]
    I[Memory Store]
  end
  subgraph External
    J[LLM Backend]
    K[MCP Servers]
  end

  A --> B
  B --> C
  C --> J
  B --> D
  B --> E
  B --> F
  B --> G
  B --> H
  B --> I
  B --> K
```

### Core packages

| Package | Responsibility |
|---|---|
| `agent-core` | Agent loop, backend, configuration, compaction, hooks, tools |
| `coding-agent` | System prompt, skills loading, sessions, trust model, TUI utilities |
| `agent-capabilities` | Reasoners (ToT, SSR, Reflexion), subagents, todo management, EOH |
| `tui` | Terminal UI components, engine, layers, state management |

### Key concepts

- **Agent loop** — the core cycle: receive input → build system prompt → call LLM → parse response → execute tools → repeat.
- **Skills** — `SKILL.md` files that inject specialized instructions into the system prompt when triggered.
- **Hooks** — lifecycle callbacks (before/after tool calls, before/after LLM requests, etc.) for plugins.
- **Sessions** — persistent conversation history stored in SQLite, with support for bookmarks, branching, and compaction.
- **Trust model** — permission modes (`acceptAll`, `acceptEdits`, `ask`, `plan`) control agent behavior.

## Why local-first?

Logician keeps orchestration, tools, and session storage on your machine. Network exposure depends on the model endpoint, MCP servers, web tools, and plugins you configure.

This means:
- Local model endpoints can keep model traffic on your machine
- Secrets can remain in `~/.logician/.env` rather than project files
- Core editing and session workflows do not require a hosted Logician service
- Full control over data retention and session history
