# Logician

[![Node.js](https://img.shields.io/badge/node-%3E%3D22.19.0-brightgreen)](https://nodejs.org)
[![TypeScript](https://img.shields.io/badge/typescript-5.x-blue)](https://typescriptlang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A local-first coding agent with a streaming terminal UI. SSH-ready, thinking-visible, and built for real code editing workflows.

![Logician TUI](logo/logo.png)

## At a glance

- **Terminal-native** — works over SSH, in tmux, in any VT100-compatible terminal
- **Streaming responses** — see reasoning, tool progress, and results as they happen
- **Safe edits** — strict text matching, CRLF/BOM preservation, path normalization
- **Skills & plugins** — `SKILL.md`-driven capabilities, Claude-style lifecycle hooks
- **Subagents** — delegate to child agents with isolated worktrees
- **Structured reasoning** — SSR, Tree of Thoughts, Reflexion, and more
- **Session management** — persistence, bookmarks, branching, rewind checkpoints, compaction
- **MCP support** — stdio and streamable HTTP MCP servers

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
```

## Architecture

Logician is a monorepo with four packages:

```
tui/
├── packages/agent-core/       Generic agent engine: loop, harness, hooks, types
├── packages/agent-capabilities/  Capabilities: todo, ask-user, subagents, reasoners
├── packages/coding-agent/     Coding tools, skills, MCP, prompts, session store
└── packages/tui/              Terminal rendering, input, themes, overlays
```

| Package | Exports |
|---|---|
| `@logician/agent-core` | `core/*`, `hooks/*`, `tools/*`, `compaction/*`, `message-queue/*` |
| `@logician/agent-capabilities` | `todo/*`, `ask-user/*`, `subagents/*`, `reasoners/*`, `tools` |
| `@logician/coding-agent` | `tools`, `skills`, `mcp`, `context-files`, `prompts`, `trust` |
| `@logician/tui` | Terminal UI layer |

## Tools

| Tool | Purpose |
|---|---|
| `bash` | Run shell commands with timeout/abort |
| `read_file` | Read files with path normalization |
| `write_file` | Create/replace files |
| `edit_file` | Strict text edits with fuzzy fallback |
| `file_diff` | Show file diffs |
| `grep` | Content search via ripgrep |
| `find` | File location via fd/find |
| `list_files` | Safe directory listing |
| `git` | Git status/diff/log |
| `web_search` | Search via SearXNG |
| `web_fetch` | Fetch web content |
| `read_skill` | Load skill instructions |
| `todo` | Task tracking |
| `task_status` | Structured completion |
| `ask_user` | User input prompts |
| `spawn_agent` | Child agent runner |
| `spawn_agent_parallel` | Parallel child agents |
| `coordinate_subagents` | Multi-agent coordination |

## Configuration

Config is read in order: `LOGICIAN_CONFIG` → `.logician.json` → `~/.logician/settings.json` → env vars.

```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "model": "local-model",
  "theme": "dark",
  "permissionMode": "acceptEdits",
  "compaction": { "enabled": true, "reserveTokens": 16384 },
  "mcpServers": {
    "context7": { "type": "streamable-http", "url": "https://mcp.context7.com/mcp" }
  }
}
```

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

## License

MIT
