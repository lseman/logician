# Logician

[![Node.js](https://img.shields.io/badge/node-%3E%3D22.19.0-brightgreen)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/typescript-5.x-blue)](https://typescriptlang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A local-first coding agent with a streaming terminal UI. SSH-ready, thinking-visible, and built for real code editing workflows.

Logician turns natural-language instructions into verified code changes — with full reasoning trace, session persistence, and skill-based extensibility. No cloud dependency, no black-box prompts.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/logician-banner.svg">
  <img src="logo/logician-banner-light.svg" alt="Logician" width="800">
</picture>

## Features

- **Terminal-native** — works over SSH, in tmux, in any VT100-compatible terminal
- **Streaming responses** — see reasoning, tool progress, and results as they happen
- **Safe edits** — strict text matching, CRLF/BOM preservation, path normalization
- **Skills & plugins** — `SKILL.md`-driven capabilities, Claude-style lifecycle hooks
- **Subagents** — delegate to child agents with isolated worktrees
- **Structured reasoning** — SSR, Tree of Thoughts, Reflexion, and more
- **Session management** — persistence, bookmarks, branching, rewind checkpoints, compaction
- **Cross-session memory** — persistent observations, lessons, and action tracking
- **MCP support** — stdio and streamable HTTP MCP servers
- **Permission modes** — `acceptAll`, `acceptEdits`, `ask`, `plan`

## Quick Start

```bash
cd tui
npm install
npm start
```

The TUI connects to an OpenAI-compatible backend at `http://127.0.0.1:8080` by default. Configure it in `.logician.json` or via `LOGICIAN_LLM_URL`.

### Requirements

| Requirement | Details |
|---|---|
| Node.js | `>=22.19.0` |
| LLM backend | OpenAI-compatible API |
| `rg`, `fd` | Optional — speeds up search tools |
| SearXNG | Optional — powers `web_search` |
| MCP servers | Optional — extend tool surface |

### Development

```bash
cd tui
npm run dev        # start dev server
npm run typecheck  # TypeScript check
npm test           # run tests
npm start -- doctor --json  # read-only readiness report
npm start -- exec --jsonl "fix the failing test"  # headless JSONL stream
```

`doctor` inspects configuration, local dependencies, MCP declarations, skills,
permissions, and sandbox capability without contacting the configured model or
starting MCP servers. Omit `--json` for a compact human-readable report.

`exec --jsonl` keeps stdout machine-readable with versioned content, tool,
error, terminal metadata, and `done` records. Diagnostics go to stderr, and
reasoning tokens are not included in the stream.

## Architecture

Logician is a monorepo with four packages under `tui/packages/`:

```
tui/packages/
├── agent-core/              Agent engine: loop, harness, hooks, types
├── agent-capabilities/      Capabilities: todo, ask-user, subagents, reasoners
├── coding-agent/            Orchestration: sessions, config, skills, MCP, prompts
└── tui/                     Terminal rendering, input, themes, overlays
```

| Package | Exports |
|---|---|
| `@logician/agent-core` | `core/*`, `hooks/*`, `tools/*`, `compaction/*`, `message-queue/*` |
| `@logician/agent-capabilities` | `todo/*`, `ask-user/*`, `subagents/*`, `reasoners/*`, `eoh/*` |
| `@logician/coding-agent` | `tools`, `skills`, `mcp`, `context-files`, `prompts`, `trust`, `sessions` |
| `@logician/tui` | `components/*`, `engine/*`, `layers/*`, `state/*` |

The `@logician/observational-memory` npm package (published separately) provides
the cross-session memory store used by the agent at runtime.

## Tools

**File operations**
| Tool | Purpose |
|---|---|
| `read_file` | Read files with path normalization |
| `write_file` | Create/replace files (auto-creates dirs) |
| `edit_file` | Strict text edits with unique-match guarantee |
| `file_diff` | Show file diffs |
| `list_files` | Safe directory listing |

**Search**
| Tool | Purpose |
|---|---|
| `grep` | Content search via ripgrep |
| `find` | File location via fd/find |

**System**
| Tool | Purpose |
|---|---|
| `bash` | Run shell commands with timeout/abort |
| `git` | Git status/diff/log |
| `sandbox` | Execute commands in Bubblewrap-isolated sandbox |

**Agent primitives**
| Tool | Purpose |
|---|---|
| `todo` | Task tracking with status transitions |
| `task_status` | Structured completion/delegation |
| `ask_user` | User input prompts |
| `spawn_agent` | Child agent runner (isolated context) |
| `spawn_agents` | Parallel child agent runners |

**Web & docs**
| Tool | Purpose |
|---|---|
| `web_search` | Search via SearXNG |
| `web_fetch` | Fetch web content |
| `read_skill` | Load skill instructions |

**MCP tools**
| Tool | Purpose |
|---|---|
| `mcp_*` | External MCP server tools (configured in `.logician.json`) |

## Configuration

Config is read in order: `LOGICIAN_CONFIG` → `.logician.json` → `~/.logician/settings.json` → env vars.

```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "model": "local-model",
  "theme": "dark",
  "executionProfile": "minimal",
	"webSearch": { "baseUrl": "http://127.0.0.1:8090", "maxResults": 10 },
  "permissionMode": "acceptEdits",
  "compaction": { "enabled": true, "reserveTokens": 16384 },
  "mcpServers": {
    "context7": { "type": "streamable-http", "url": "https://mcp.context7.com/mcp" }
  }
}
```

### Execution profiles

`executionProfile` controls who owns continuation and termination policy. It
does not change tool permissions or sandboxing; those remain controlled by
`permissionMode` and the sandbox profile.

| Capability | `autonomous` (default) | `minimal` |
|---|---|---|
| Provider calls and streaming | Enabled | Enabled |
| Tool execution and result delivery | Enabled | Enabled |
| Steering and follow-up queues | Enabled | Enabled |
| Retries, cancellation, checkpoints, and compaction | Enabled | Enabled |
| Caller hooks and SDK `stopPolicies` | Enabled | Enabled |
| Built-in continuation and todo nudges | Enabled | Disabled |
| Acceptance contracts and reflection | Enabled | Disabled |
| Duplicate/failure, budget, and thinking-loop guards | Enabled | Disabled |
| Heuristic completion, question, and non-committal stops | Enabled | Disabled |
| Built-in `task_status` termination | Enabled | Disabled |

Use `autonomous` for the interactive coding agent: Logician can notice
unfinished work, inject corrective turns, validate acceptance criteria, and
produce a structured conclusion.

Use `minimal` when embedding the loop as a mechanism inside another agent or
SDK. The loop still calls the provider, executes tools, drains queues, retries,
compacts context, and honors cancellation, but it becomes idle after the
provider has no pending work. The caller can then use `stopPolicies` to inject
another turn or return a structured `completed`, `needs_input`, `blocked`,
`failed`, or `cancelled` outcome.

In the TUI, press **Ctrl+I** to toggle `autonomous ↔ minimal`. The status bar
shows `exec: auto` or `exec: minimal`; the selected profile is persisted and
takes effect on the next turn. It can also be changed through the settings
overlay or `/settings execution-policy <autonomous|minimal>`.

`web_search` uses SearXNG's JSON endpoint. Override the default local endpoint
with `webSearch.baseUrl` or `LOGICIAN_SEARXNG_URL`; use
`LOGICIAN_SEARXNG_MAX_RESULTS` to change the default result count.

Successful JavaScript, TypeScript, and JSON edits receive a fast syntax check
before the next model turn. Set `LOGICIAN_POST_EDIT_DIAGNOSTICS=0` to disable
this advisory check.

### Permission modes

| Mode | Behavior |
|---|---|
| `acceptAll` | Execute everything without asking |
| `acceptEdits` | Auto-accept reads; ask before writes |
| `ask` | Ask before every tool call |
| `plan` | Plan only, no execution |

### Slash commands

| Area | Commands |
|---|---|
| Sessions | `/new`, `/sessions`, `/save`, `/rename`, `/bookmark`, `/load`, `/export` |
| Agent | `/status`, `/agents`, `/agent`, `/pipeline`, `/reload` |
| Context | `/context`, `/compact`, `/fork`, `/reset`, `/changes` |
| RAG/docs | `/mount`, `/upload`, `/docs`, `/rag` |
| Tools | `/skills-health`, `/plugins`, `/mcp`, `/theme`, `/reasoner`, `/settings` |
| Permissions | `/permissions`, `/plan`, `/rewind` |

Run `/help` in the TUI for the full live list.

## Skills

Skills are `SKILL.md` files with YAML frontmatter. They define capabilities the agent can activate on demand.

```md
---
name: code-review
description: Review code changes for correctness, risks, and missing tests.
allowed-tools: [read_file, grep, git]
argument-hint: <diff-or-topic>
---

Inspect the change as a reviewer. Prioritize bugs and regressions before style.
```

## Themes

Built-in themes live in `tui/packages/tui/src/layers/theme`. Custom themes go in `~/.logician/themes/`.

```json
{
  "name": "my-theme",
  "mode": "truecolor",
  "colors": {
    "accent": "#58a6ff",
    "text": "#c9d1d9",
    "bg": "#0d1117",
    "success": "#56d364",
    "error": "#f85149",
    "warning": "#d29922"
  }
}
```

Switch with `/theme dark` or `/theme light`.

## Architecture

Logician is a monorepo with five packages under `tui/packages/`:

```
tui/packages/
├── agent-core/              Agent engine: loop, harness, hooks, types
├── agent-capabilities/      Capabilities: todo, ask-user, subagents, reasoners
├── coding-agent/            Orchestration: sessions, config, skills, MCP, prompts
├── legacy-observational-memory/  Structured observations with file-based persistence
└── tui/                     Terminal rendering, input, themes, overlays
```

### Agent loop
The agent runs a single harness loop: receive user input → call backend → parse response → execute tools → repeat. Each iteration is a "turn" with full lifecycle events (`turn_start`, `tool_call`, `tool_result`, `turn_end`). The `AgentHarness` orchestrates the loop via `runAgentLoop()`, which manages budget, compaction triggers, and guardrails. The `OutputGuard` watches for degenerate patterns (context-full errors, empty responses, provider errors) and triggers recovery (auto-compact, retry with backoff, turn abortion). The `LoopDetector` tracks tool-call-level patterns (duplicate calls, failure loops, circling) to detect when the agent is stuck.

### Hook system
Hooks are lifecycle callbacks registered per event type on a `HookBus`. The bus unifies single-handler contracts into a multi-handler system with per-event reducer semantics:

| Event | Reducer | Behavior |
|---|---|---|
| `beforeToolCall` | early-block | First `{content}` short-circuits; `{args}` rewrites thread |
| `afterToolCall` | patch-accumulate | Each handler sees prior patch; later non-undefined fields win |
| `prepareNextTurn` | transform | Messages thread through all handlers |
| `shouldStopAfterTurn` | first-true | Any handler returning `true` stops the loop |

Handlers have `priority` (higher first), `timeoutMs` (per-handler, 0 = no timeout), and `source` metadata for diagnostics. The `errorMode` (default `continue`) controls whether a thrown handler aborts the chain or is skipped. Built-in hooks include: duplicate-call guard, failure-loop guard, thinking-loop detector, budget tracker, proactive compaction, and file checkpointing. Extensions (skills, plugins) register hooks via `AgentHooks` objects.

### Permission model
Four modes control tool execution: `acceptAll` (no prompts), `acceptEdits` (auto-read, ask-write), `ask` (every tool prompts), `plan` (no execution). The permission resolver checks the mode, tool name, and configured allow/deny lists before each call. Blocked tools emit a `permission_denied` event. The `permissions` config object maps tool names to `allowed`/`denied` arrays for fine-grained control. When `permissionMode` is `ask`, a popup overlay appears with accept/reject options.

### Session lifecycle
Sessions persist to disk as JSONL transcript files. They support:
- **Bookmarks** — named checkpoints at any turn
- **Branching** — fork a session at any point; branches merge back via summarization
- **Rewind** — restore to a bookmark checkpoint
- **Compaction** — summarize old turns to free context. Triggers at `proactiveCompactionFraction` (default 0.8) of the context window. Preserves tool call results as summaries while collapsing content chunks.
- **Cross-session memory** — structured observations (decisions, errors, plans) indexed in a BM25 knowledge base retrievable via `ctx_search`

### Trust model
At startup, the TUI scans the working directory for trust-requiring resources (`.logician/`, skills, extensions). A prompt overlay appears with five choices: trust this folder, trust parent, session-only trust, deny, or exit. Trust decisions are persisted to `~/.logician/` and scoped to the workspace path. Skills with `allowed-tools` lists are gated by trust — untrusted workspaces cannot activate skill capabilities.

### Subagent isolation
Child agents run in isolated Bun worktrees (`worktree`) with their own `node_modules` and file system scope. They receive a subset of the parent's tools, a truncated context window, and their own session. Results flow back as structured reports with optional streaming transcripts and child tool call traces. The `spawn_agent` and `spawn_agents` capabilities handle delegation with configurable timeouts and validation retries.

### Response guardrails
The `response-patterns` module contains regex patterns for detecting degenerate model behavior:
- **Non-committal patterns** — hedging language ("I need to check", "let me think")
- **Completion patterns** — task-complete declarations
- **Meta-reasoning patterns** — reasoning about reasoning without action
- **Circling patterns** — retry intent without progress
These patterns feed into the `OutputGuard` and `LoopDetector` for early intervention.

### MCP integration
Logician supports MCP (Model Context Protocol) servers via stdio and streamable HTTP. MCP tools are discovered at startup, registered in the `ToolRegistry`, and surfaced to the agent as native tools. Load failures are injected into the system prompt for transparency. Configuration lives in `.logician.json` under `mcpServers` or `mcp` (new format).

## License

MIT
