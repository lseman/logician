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
  subgraph TUI["apps/tui"]
    A[Terminal UI]
  end
  subgraph Agent
    C[Agent Core]
  end
  subgraph Capabilities["agent-blocks"]
    D[Reasoners]
    E[Subagents]
    F[Todo & Task Mgmt]
  end
  subgraph Evolution
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

  A --> C
  C --> J
  A --> D
  D --> C
  D --> G
  C --> G
  C --> H
  C --> I
  C --> K
```

### Core packages

| Package | Responsibility |
|---|---|
| `agent-protocol` | Dependency-free, versioned notifications for clients |
| `agent-core` | Agent loop, harness, context, configuration, sessions, skills, MCP, trust, and product orchestration |
| `agent-blocks` | Optional reasoners, subagents, todo management, interaction, and RAG tools |
| `eoh` | Evolution of Heuristics — standalone optimization engine, wired into `agent-core` and re-exported by `agent-blocks` |
| `memory` | SQLite-backed persistent memory: observation capture, consolidation, retrieval |
| `rag` | Hybrid dense + BM25 retrieval, chunking, reranking, context budgets |
| `autoresearch` | Measured experiment loops — run, evaluate, keep or discard |
| `agent-eval` | Outcome-grounded evaluation runner for agent trials |
| `tui` | Terminal UI components, engine, layers, state management (`apps/tui`) |
| `memory-mcp` | Stdio MCP adapter exposing `@logician/memory` to any MCP client (`apps/memory-mcp`) |

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
