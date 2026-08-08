---
title: "Structure"
analysis_date: "2026-08-08"
---

# Structure

**Analysis Date:** 2026-08-08

*Logician TUI — directory layout, key locations, and naming conventions.*

---

## Top-Level Layout

```
/home/seman/logician/
├── .git/                          — Git repository
├── .github/workflows/             — CI/CD: release.yml, deploy-pages.yml
├── .logician/                     — User config directory (~/.logician/)
├── .playwright-mcp/               — Playwright MCP test recordings
├── .planning/                     — GSD planning artifacts (being created)
│   ├── codebase/                  — Codebase map documents
│   ├── onboarding/                — Onboarding summary
│   ├── PROJECT.md                 — Project context
│   ├── REQUIREMENTS.md            — Requirements
│   ├── ROADMAP.md                 — Phase roadmap
│   └── STATE.md                   — Project state
├── docs/                          — Documentation site
│   ├── getting-started.md
│   ├── index.md
│   ├── overview.md
│   ├── architecture/              — Architecture docs
│   ├── guides/                    — User guides
│   ├── reference/                 — API/config reference
│   └── tutorials/                 — Tutorial content
├── logo/                          — Brand assets
├── notes/                         — Working notes
├── plugins/                       — Plugin definitions
├── rag-python/                    — Python RAG extraction package
├── repos/                         — Related external repos
│   ├── claude-mem/                — Claude Code memory plugin
│   ├── gsd-core/                  — GSD workflow core
│   ├── pi/                        — Pi coding agent
│   ├── pi-autoresearch/           — Pi autoresearch
│   └── agentmemory/               — Agent memory service
├── site/                          — Website source
├── skills/                        — Skill definitions
├── tui/                           — Main TUI monorepo
│   ├── package.json               — Workspace root
│   ├── tsconfig.json              — Project references (6 packages)
│   └── packages/                  — Sub-packages
│       ├── agent-core/            — Core agent engine
│       ├── agent-capabilities/    — Built-in tools & capabilities
│       ├── coding-agent/          — Orchestration layer
│       ├── tui/                   — Terminal UI layer
│       ├── rag/                   — RAG pipeline
│       ├── memory/                — Persistent memory
│       └── autoresearch/          — Autonomous experiment loop
├── biome.json                     — Root lint/format config
├── bun.lock                       — Bun lockfile
├── Makefile                       — Build targets
├── package.json                   — Root shim
└── README.md                      — Project readme
```

---

## Package Layouts

### `tui/packages/agent-core/` — Core Agent Engine

```
src/
├── index.ts                       — Barrel: loop, harness, types, tools
├── agent/
│   ├── agent-loop-runner.ts       — Functional agent loop
│   ├── backend.ts                 — OpenAI-compatible HTTP client
│   ├── execution-policy.ts        — Stop policies
│   ├── file-checkpoints.ts        — Workspace snapshots
│   ├── harness.ts                 — Phase management, queues, compaction
│   ├── messages.ts                — Message conversion, token estimation
│   ├── runtime-state.ts           — Runtime state types
│   ├── session.ts                 — JSONL session persistence
│   ├── tool-cache.ts              — Tool result caching
│   ├── types.ts                   — Core types barrel
│   ├── configuration/             — Config validation, inference modes
│   ├── guards/                    — Loop detector, response patterns, acceptance
│   ├── harness/                   — Branching, compaction, phase, queue ops
│   ├── loop/                      — Callbacks, reflection
│   ├── summaries/                 — Branch summarization
│   └── tasks/                     — Task state, todo state
├── compaction/                    — Context compaction engine
├── extensions/                    — Extension system (event bus, Pi adapter)
├── hooks/
│   ├── builtin/                   — Built-in hooks (budget, compaction, guards)
│   ├── extensions/                — Extension event types
│   └── native/                    — Hook bus, metrics
├── plugins/
│   └── claude-code/               — Claude Code plugin adapter
├── queue/                         — Message delivery manager
├── runtime/                       — Conclusion policy, tool batch controller
└── tools/
    └── shared/                    — JSON utils, permissions, registry, plugins
```

**Key files:**
- `src/agent/harness.ts` — Central orchestration (phases, queues, compaction)
- `src/agent/agent-loop-runner.ts` — Functional loop contract
- `src/agent/backend.ts` — LLM backend with error classification
- `src/hooks/builtin/builtin-hooks.ts` — Default safeguard hooks
- `src/extensions/` — Typed event system for extensions

### `tui/packages/agent-capabilities/` — Agent Capabilities

```
src/
├── index.ts                       — Barrel: tasks, interaction, delegation, reasoning
├── tools.ts                       — Built-in tools registry
├── tasks/                         — todo, task_status tools
├── interaction/
│   └── ask-user/                  — ask_user tool
├── delegation/                    — spawn_agent, spawn_agents tools
├── reasoning/                     — 10+ reasoners (AutoCoT, CoVe, GoT, ToT, etc.)
├── rag/                           — RAG tools (rag_ingest_pdf, rag_search_docs)
└── eoh/                           — End-of-Head demo (engine, evaluator, LLM)
```

**Key files:**
- `src/tools.ts` — `getBuiltInTools()`, `getBuiltInSubagentTools()`
- `src/reasoning/registry.ts` — Reasoner registry with metadata
- `src/delegation/definitions.ts` — Subagent tool definitions

### `tui/packages/coding-agent/` — Orchestration Layer

```
src/
├── index.ts                       — Barrel: commands, config, context, MCP, skills, tools, trust
├── application/
│   ├── agent-bridge.ts            — Main bridge: wires all layers
│   ├── loop-manager.ts            — Loop management
│   ├── goal-manager.ts            — Goal execution
│   ├── subagent-coordinator.ts    — Subagent coordination
│   ├── interaction-coordinator.ts — User interaction
│   ├── tool-router.ts             — Tool routing
│   └── eoh/                       — EoH controller
├── commands/
│   └── slash-commands.ts          — /gsd-* command definitions
├── configuration/
│   └── config.ts                  — Logician config loading/saving
├── context/
│   ├── system-prompt.ts           — System prompt builder
│   └── files/                     — File mention handling
├── developer-tools/
│   ├── doctor.ts                  — Health diagnostics
│   ├── lsp-manager.ts             — Language server integration
│   └── post-edit-diagnostics.ts   — Post-edit diagnostics
├── mcp/
│   ├── client.ts                  — StdioMcpClient, HttpMcpClient
│   └── manager.ts                 — McpManager
├── prompts/                       — System prompt templates
├── runtime/
│   ├── event-mapping.ts           — Agent event → bridge event
│   ├── events.ts                  — Bridge event types
│   ├── runtime-config.ts          — Runtime configuration
│   └── plugin-result-formatter.ts — Plugin result formatting
├── sessions/
│   ├── session-store.ts           — JSONL session persistence
│   └── transcript.ts              — Session message history
├── skills/
│   ├── activation.ts              — Skill discovery and activation
│   └── loader.ts                  — Skill loading
├── tools/                         — All agent tools
│   ├── bash.ts, shell.ts          — Shell execution
│   ├── edit-file.ts, write-file.ts, write-file-append.ts
│   ├── read-file.ts, list-files.ts, find.ts, search.ts
│   ├── git.ts, file-diff.ts       — Git operations
│   ├── web-fetch.ts, web-search.ts — Web tools
│   ├── sandbox.ts                 — Sandbox execution
│   ├── diff-utils.ts              — Diff utilities
│   ├── memory-tools.ts            — Memory tools
│   ├── autoresearch.ts            — Autoresearch tool
│   └── shared/                    — Atomic write, file mutation queue, tools manager
├── trust/
│   ├── checker.ts                 — Trust decision checker
│   ├── manager.ts                 — Trust management
│   └── store.ts                   — Trust store persistence
└── tui-utils.ts                   — Utility functions
```

**Key files:**
- `src/application/agent-bridge.ts` — Main integration point (1000+ lines)
- `src/tools/` — 30+ tool implementations
- `src/trust/` — Trust decision system
- `src/mcp/` — MCP server management

### `tui/packages/tui/` — Terminal UI Layer

```
src/
├── index.ts                       — Entry point (env loading, trust prompt, TUI launch)
├── app/
│   ├── tui.ts                     — Main TUI class
│   ├── goal-runner.ts             — Multi-turn goal execution
│   ├── headless-exec.ts           — Non-interactive execution
│   ├── input-controller.ts        — Keyboard input handling
│   ├── inference-settings.ts      — Inference mode settings
│   ├── git-status.ts              — Git status display
│   ├── tui-helpers.ts             — UI helper functions
│   ├── bridge-event-handler.ts    — Event handling from bridge
│   ├── commands/                  — Command handlers
│   │   ├── async-handlers.ts      — Async command handlers
│   │   ├── local-handlers.ts      — Local command handlers
│   │   ├── submit-handler.ts      — Submit handling
│   │   └── context.ts             — Command context
│   ├── overlay-controllers/       — Overlay management
│   ├── session/                   — Session management
│   └── startup/                   — Startup initialization
├── input/
│   ├── input-bar.ts               — Input bar rendering
│   ├── kill-ring.ts               — Kill ring (clipboard)
│   ├── undo-stack.ts              — Undo/redo
│   └── word-navigation.ts         — Word-level navigation
├── overlays/                      — Interactive overlays
│   ├── trust-prompt-overlay.ts    — Trust decisions
│   ├── choice-popup.ts            — Choice selection
│   ├── file-mention-popup.ts      — File mention popup
│   ├── inference-mode-selector.ts — Inference mode selection
│   ├── mcp-manager.ts             — MCP server management
│   ├── model-selector.ts          — Model selection
│   ├── permission-popup.ts        — Permission requests
│   ├── plugin-manager.ts          — Plugin management
│   ├── reasoner-selector.ts       — Reasoner selection
│   ├── selector-controller.ts     — Selector base class
│   ├── session-manager.ts         — Session management
│   ├── settings-overlay.ts        — Settings overlay
│   ├── slash-popup.ts             — Slash command popup
│   ├── theme-selector.ts          — Theme selection
│   ├── autoresearch-dashboard.ts  — Research dashboard
│   └── popup-utils.ts             — Popup utilities
├── rendering/
│   ├── layout.ts, layout-node.ts  — Flex layout engine
│   ├── scroll-view.ts             — Scrollable views
│   ├── separator.ts               — Visual separators
│   ├── terminal-sanitize.ts       — Output sanitization
│   ├── flex.ts                    — Flexbox utilities
│   └── transcript/                — Transcript rendering
│       ├── display.ts             — Main transcript display
│       ├── layout.ts              — Transcript layout
│       ├── text-utils.ts          — Text utilities
│       ├── new-output-indicator.ts — New output indicator
│       ├── file-language.ts       — File language detection
│       └── render/                — Renderers
│           ├── content.ts         — Content rendering
│           ├── markdown-table.ts   — Table rendering
│           ├── subagent.ts        — Subagent rendering
│           ├── thinking.ts        — Thinking block rendering
│           ├── tool.ts            — Tool call rendering
│           └── tool-details.ts    — Tool detail rendering
├── state/
│   └── turn-state.ts              — Turn state management
├── status/
│   ├── status-bar.ts              — Status bar
│   ├── todo-bar.ts                — Todo/task bar
│   ├── steer-queue.ts             — Steering queue display
│   ├── notification-center.ts     — Notifications
│   ├── research-widget.ts         — Research widget
│   └── work-surface.ts            — Work surface display
├── terminal/
│   ├── core.ts                    — Terminal detection, width
│   ├── theme.ts                   — Color themes
│   ├── primitives.ts              — Terminal primitives
│   ├── utils.ts                   — Terminal utilities
│   └── hyperlinks.ts              — Terminal hyperlinks
├── testing/
│   ├── pty-app-home.ts            — PTY app for testing
│   ├── pty-harness.ts             — PTY harness
│   └── terminal-screen.ts         — Terminal screen mock
└── __tests__/                     — TUI tests
```

**Key files:**
- `src/index.ts` — Entry point
- `src/app/tui.ts` — Main TUI class
- `src/overlays/` — 15+ overlay components
- `src/rendering/transcript/` — Transcript rendering system

### `tui/packages/rag/` — RAG Pipeline

```
src/
├── index.ts                       — Barrel export
├── embedder.ts                    — HuggingFace Transformers ONNX embedder
├── ingestion.ts                   — Python Docling subprocess ingestion
├── types.ts                       — RAG types
├── pipeline/
│   └── index.ts                   — Pipeline orchestration
└── store/
    ├── index.ts                   — Store barrel
    └── sqlite-store.ts            — SQLite-backed vector store
```

### `tui/packages/memory/` — Persistent Memory

```
src/
├── index.ts                       — Barrel export
├── types.ts                       — Memory types
├── bun-sqlite.d.ts                — Bun SQLite type declarations
├── embeddings/
│   └── local-embedder.ts          — Local ONNX embeddings
├── episodes/
│   ├── episode-synthesizer.ts     — Episode synthesis
│   └── semantic-extractor.ts      — Semantic extraction
├── hooks/
│   ├── hook-adapter.ts            — Hook adapter
│   └── memory-hooks.ts            — Memory hooks
├── store/
│   └── index.ts                   — Store implementation
└── viewer/
    ├── viewer-server.ts           — HTTP viewer server
    └── viewer-document.ts         — Document rendering
```

### `tui/packages/autoresearch/` — Autonomous Experiment Loop

```
src/
├── index.ts                       — AutoresearchSession class
├── hooks.ts                       — Experiment hooks
├── paths.ts                       — Experiment path management
├── jsonl.ts                       — JSONL read/write
├── compaction.ts                  — Experiment log compaction
├── shortcuts.ts                   — Experiment shortcuts
└── assets/                        — Static assets
```

---

## Naming Conventions

| Convention | Examples |
|---|---|
| **Package names** | `@logician/<name>` — scoped npm packages |
| **File names** | `snake_case.ts` — all source files |
| **Class names** | `PascalCase` — `AgentHarness`, `McpManager`, `LogicianTUI` |
| **Function names** | `camelCase` — `runAgentLoop`, `createMcpClient` |
| **Type names** | `PascalCase` — `AgentConfig`, `McpServerConfig` |
| **Interface names** | `PascalCase` — `McpClient`, `Tool` |
| **Constants** | `UPPER_SNAKE_CASE` — `MCP_PROTOCOL_VERSION`, `DEFAULT_COMPACTION_FRACTION` |
| **Test files** | `*.test.ts` — colocated with source |
| **Barrel files** | `index.ts` — re-exports from sub-modules |
| **Private files** | No special convention; TypeScript `private` keyword used |

---

## Key Patterns

| Pattern | Location | Description |
|---|---|---|
| **Barrel exports** | All packages | `index.ts` re-exports from sub-modules |
| **Sub-path exports** | All packages | `exports` field in package.json maps sub-paths |
| **TypeBox schemas** | agent-core | `@sinclair/typebox` for config validation |
| **JSONL persistence** | session.ts, transcript.ts | Line-delimited JSON for sessions |
| **Result types** | types-errors.ts | `Result<T, E>` with `ok()`/`err()` |
| **Hook composition** | builtin-hooks.ts | `composeHooks()` layers hooks |
| **Event bus** | extensions/event-bus.ts | Typed event system |
| **Dependency injection** | agent-bridge.ts | Wires all layers together |

---

*Structure analysis: 2026-08-08*
