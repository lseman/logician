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

Logician is a monorepo with five packages under `tui/packages/`:

```
tui/packages/
├── agent-core/              Agent engine: loop, harness, hooks, types
├── agent-capabilities/      Capabilities: todo, ask-user, subagents, reasoners
├── coding-agent/            Orchestration: sessions, config, skills, MCP, prompts
├── legacy-observational-memory/  Structured observations with file-based persistence
└── tui/                     Terminal rendering, input, themes, overlays
```

| Package | Exports |
|---|---|
| `@logician/agent-core` | `core/*`, `hooks/*`, `tools/*`, `compaction/*`, `message-queue/*` |
| `@logician/agent-capabilities` | `todo/*`, `ask-user/*`, `subagents/*`, `reasoners/*`, `eoh/*` |
| `@logician/coding-agent` | `tools`, `skills`, `mcp`, `context-files`, `prompts`, `trust`, `sessions` |
| `@logician/legacy-observational-memory` | Structured observations with file-based persistence |
| `@logician/tui` | Terminal UI layer |

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
	"webSearch": { "baseUrl": "http://127.0.0.1:8090", "maxResults": 10 },
  "permissionMode": "acceptEdits",
  "compaction": { "enabled": true, "reserveTokens": 16384 },
  "mcpServers": {
    "context7": { "type": "streamable-http", "url": "https://mcp.context7.com/mcp" }
  }
}
```

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

## Architecture Decisions

### Agent loop
The agent runs a single harness loop: receive user input → call backend → parse response → execute tools → repeat. Each iteration is a "turn" with full lifecycle events (turn_start, tool_call, tool_result, turn_end). The loop runner manages budget, compaction triggers, and guardrails.

### Hook system
Hooks are lifecycle callbacks registered per event type (turn_start, tool_call, turn_end, etc.). They run in order within a hook bus. Each hook can read/write turn state, emit events, or short-circuit the loop. Extensions register hooks via `SKILL.md` or plugin manifests.

### Permission model
Four modes control tool execution: `acceptAll` (no prompts), `acceptEdits` (auto-read, ask-write), `ask` (every tool prompts), `plan` (no execution). The permission resolver checks the mode, tool name, and configured allow/deny lists before each call. Blocked tools emit a `permission_denied` event.

### Session lifecycle
Sessions persist to disk as JSONL transcript files. They support bookmarks (named checkpoints), branching (fork a session at any point), rewind (restore to a bookmark), and compaction (summarize old turns to free context). Compaction triggers at configurable token thresholds and preserves tool call results as summaries.

### Trust model
At startup, the TUI scans the working directory for trust-requiring resources (`.logician/`, skills, extensions). A prompt overlay appears with five choices: trust this folder, trust parent, session-only trust, deny, or exit. Trust decisions are persisted to `~/.logician/` and scoped to the workspace path.

### Subagent isolation
Child agents run in isolated Bun worktrees with their own `node_modules` and file system scope. They receive a subset of the parent's tools, a truncated context window, and their own session. Results flow back as structured reports with optional streaming transcripts and child tool call traces.

## License

MIT
