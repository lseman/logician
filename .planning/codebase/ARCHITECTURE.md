---
title: "Architecture"
analysis_date: "2026-08-08"
---

# Architecture

**Analysis Date:** 2026-08-08

*Logician TUI — system design, layers, data flow, abstractions, and entry points.*

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Logician TUI                             │
│                    (tui/packages/tui)                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │  Terminal UI │  │ Goal Runner  │  │ Trust Prompt Overlay  │  │
│  │  (Curses/    │  │ (multi-turn  │  │ (interactive trust    │  │
│  │   ANSI)      │  │  execution)  │  │  decisions)           │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘  │
│         │                  │                      │              │
│         └──────────────────┼──────────────────────┘              │
│                            │                                     │
│              ┌─────────────▼─────────────┐                      │
│              │   AgentCoreBridge          │                      │
│              │   (coding-agent)           │                      │
│              └─────────────┬─────────────┘                      │
└────────────────────────────┼─────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                   │
   ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼───────┐
   │ Agent Core  │   │ Agent Cap.  │   │  Coding Agent │
   │ (loop,      │   │ (tools,    │   │  (sessions,   │
   │  harness,   │   │  subagents,│   │   config,     │
   │  hooks)     │   │  reasoners)│   │   skills,     │
   └──────┬──────┘   └────────────┘   │   MCP, trust) │
          │                            └───────────────┘
          │
   ┌──────▼──────┐   ┌──────────────┐   ┌──────────────────┐
   │  Backend    │   │  Extensions  │   │  Built-in Hooks  │
   │ (OpenAI     │   │  (event bus, │   │  (compaction,    │
   │  compatible)│   │   Pi adapter)│   │   loop-detect,   │
   └─────────────┘   └──────────────┘   │   budget, guards)│
                                        └──────────────────┘
```

---

## Layer Overview

### Layer 1: Terminal UI (`@logician/tui`)

The user-facing terminal interface.

| Component | File | Purpose |
|---|---|---|
| TUI shell | `tui/packages/tui/src/app/tui.ts` | Main terminal UI class |
| Goal runner | `tui/packages/tui/src/app/goal-runner.ts` | Multi-turn goal execution |
| Trust prompt | `tui/packages/tui/src/overlays/trust-prompt-overlay.ts` | Interactive trust decisions |
| Terminal core | `tui/packages/tui/src/terminal/core.ts` | Terminal detection, width |
| Theme system | `tui/packages/tui/src/terminal/theme.ts` | Color themes, truecolor |
| Headless exec | `tui/packages/tui/src/app/headless-exec.ts` | Non-interactive mode |

**Entry point:** `tui/packages/tui/src/index.ts` — loads `~/.logician/.env`, shows trust prompt, then launches TUI.

### Layer 2: Coding Agent (`@logician/coding-agent`)

The orchestration layer connecting the TUI to agent-core.

| Component | File | Purpose |
|---|---|---|
| AgentCoreBridge | `src/application/agent-bridge.ts` | Translates agent-core events to TUI shapes |
| Session store | `src/sessions/session-store.ts` | JSONL session persistence |
| Transcript | `src/sessions/transcript.ts` | Session message history |
| Config | `src/configuration/config.ts` | Logician config loading/saving |
| Skills | `src/skills/activation.ts` | Skill discovery and activation |
| MCP | `src/mcp/manager.ts` | MCP server management |
| Trust | `src/trust/index.ts` | Trust decision persistence |
| Prompts | `src/prompts/index.ts` | System prompt templates |
| Context | `src/context/system-prompt.ts` | System prompt builder |
| Runtime | `src/runtime/` | Event mapping, plugin result formatting |
| Developer tools | `src/developer-tools/` | Doctor, LSP, post-edit diagnostics |
| Slash commands | `src/commands/slash-commands.ts` | `/gsd-*` command definitions |
| Tools | `src/tools/` | Web fetch, shell, default tools |

### Layer 3: Agent Capabilities (`@logician/agent-capabilities`)

Built-in agent tools and capabilities.

| Component | Sub-path | Purpose |
|---|---|---|
| Tasks | `src/tasks/` | `todo`, `task_status` tools |
| Interaction | `src/interaction/` | `ask_user` tool |
| Delegation | `src/delegation/` | `spawn_agent`, `spawn_agents` tools |
| Reasoning | `src/reasoning/` | 10+ reasoners (AutoCoT, CoVe, GoT, ToT, etc.) |
| RAG | `src/rag/` | `rag_ingest_pdf`, `rag_search_docs`, etc. |
| Tools | `src/tools.ts` | Built-in tools registry |

**Reasoners available:** AutoCoT, BestOfN, CoVe, GoT, InContextCoT, Reflexion, SelfConsistency, SSR, ToT

### Layer 4: Agent Core (`@logician/agent-core`)

The core agent engine — the heart of the system.

| Component | Sub-path | Purpose |
|---|---|---|
| Loop runner | `src/agent/agent-loop-runner.ts` | Functional agent loop (context + prompts + config + emit => messages) |
| Harness | `src/agent/harness.ts` | Phase management, queues, steering, compaction, branching |
| Session | `src/agent/session.ts` | JSONL session persistence |
| Backend | `src/agent/backend.ts` | OpenAI-compatible HTTP client with error classification |
| Messages | `src/agent/messages.ts` | Message conversion, token estimation |
| Execution policy | `src/agent/execution-policy.ts` | Stop policies, execution profiles |
| File checkpoints | `src/agent/file-checkpoints.ts` | Workspace snapshots before writes |
| Compaction | `src/compaction/` | Context window compaction engine |
| Hooks | `src/hooks/` | Hook bus, builtin hooks, extensions |
| Extensions | `src/extensions/` | Typed event system, Pi adapter |
| Tools | `src/tools/` | Tool registry, JSON utils, permissions |
| Queue | `src/queue/` | Message delivery manager |
| Tasks | `src/agent/tasks/` | Task state controller, todo state |
| Guards | `src/agent/guards/` | Loop detector, response patterns, acceptance contract, output guard, thinking loop detector |
| Plugins | `src/plugins/claude-code/` | Claude Code hook layer adapter |

### Layer 5: RAG (`@logician/rag`)

Retrieval-Augmented Generation pipeline.

| Component | File | Purpose |
|---|---|---|
| Embedder | `src/embedder.ts` | HuggingFace Transformers ONNX inference |
| Ingestion | `src/ingestion.ts` | Python Docling subprocess for document extraction |
| Pipeline | `src/pipeline/index.ts` | Orchestration layer |
| Store | `src/store/sqlite-store.ts` | SQLite-backed vector storage |
| Types | `src/types.ts` | Search hit, pipeline config types |

### Layer 6: Memory (`@logician/memory`)

Persistent memory system with SQLite backend.

| Component | File | Purpose |
|---|---|---|
| Hooks | `src/hooks/` | Memory hooks for capture/injection |
| Embeddings | `src/embeddings/local-embedder.js` | Local ONNX embeddings |
| Store | `src/store/` | SQLite persistence |
| Episodes | `src/episodes/` | Semantic extraction |
| Viewer | `src/viewer/` | HTTP viewer server |

### Layer 7: Autoresearch (`@logician/autoresearch`)

Autonomous experiment loop.

| Component | File | Purpose |
|---|---|---|
| Session | `src/index.ts` | `AutoresearchSession` — run/measure/keep-discard |
| Hooks | `src/hooks.ts` | Experiment hooks |
| Paths | `src/paths.ts` | Experiment path management |
| JSONL | `src/jsonl.ts` | JSONL read/write |
| Compaction | `src/compaction.ts` | Experiment log compaction |
| Shortcuts | `src/shortcuts.ts` | Experiment shortcuts |

---

## Data Flow

### Agent Loop Flow

```
User Input → TUI → AgentCoreBridge → AgentHarness → AgentLoopRunner
                                                    ↓
                                              LLM Backend (OpenAI-compatible)
                                                    ↓
                                              New Messages
                                                    ↓
                                              Harness Queues (steering, follow-up)
                                                    ↓
                                              Tool Execution (via ToolRegistry)
                                                    ↓
                                              Hook Bus (before/after tool, before/after response)
                                                    ↓
                                              Extension Event Bus
                                                    ↓
                                              Back to TUI (streaming)
```

### Tool Execution Flow

```
Agent decides to call tool
  → Pre-tool hooks fire
  → ToolRegistry resolves tool
  → Permission check (PermissionMode)
  → Tool execution (sync or async)
  → Post-tool hooks fire
  → ToolResultCache stores result
  → Extension events emitted
  → Result returned to agent loop
```

### Session Persistence Flow

```
AgentHarness turn end
  → Session.append(msg) — JSONL line written
  → File checkpoint saved (before write)
  → Harness queues drained
  → Next turn begins
```

### Compaction Flow

```
Context approaches 80% of window
  → Proactive compaction hook fires
  → compactToFit() called
  → Session compact event emitted
  → Shortened history returned to loop
```

---

## Key Abstractions

### AgentHarness

The central orchestration class. Manages:
- Phases (idle, running, paused, etc.)
- Runtime state transitions
- Steering/follow-up/nextTurn queues
- Compaction triggers
- Branching (fork, navigate, summarize)
- Session lifecycle (start, switch, shutdown)

### AgentLoopRunner

The functional loop contract. Takes:
- Context (messages)
- Prompts
- Config
- Emit function
Returns: new messages

### HookBus

Typed event bus for hooks. Supports:
- Before/after tool call
- Before/after provider response
- Before/after compact
- Session lifecycle events
- Custom extension events

### ToolRegistry

Central tool registry with:
- Tool discovery
- Permission modes
- JSON parsing with comments
- Plugin integration

### ExtensionEventBus

Typed event system for extensions:
- Turn start/end
- Message start/end/update
- Tool execution start/end
- Session events
- Pi adapter for cross-compatibility

---

## Design Patterns

| Pattern | Usage |
|---|---|
| **Functional loop** | `agent-loop-runner.ts` — pure function contract |
| **Event bus** | `ExtensionEventBus` — typed events for extensions |
| **Hook composition** | `composeHooks()` — layered hook execution |
| **JSONL persistence** | Sessions, transcripts, experiment logs |
| **Result types** | `Result<T, E>` with `ok()`/`err()` |
| **Dependency injection** | `AgentCoreBridge` wires all layers |
| **Strategy** | Reasoner registry — swap reasoning strategies |
| **Observer** | Extension event handlers |
| **Factory** | `createMcpClient()` — stdio/HTTP clients |

---

*Architecture analysis: 2026-08-08*
