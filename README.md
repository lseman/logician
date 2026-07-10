# Logician

Logician is a TypeScript coding-agent workspace with a terminal UI, a lean reusable agent core, and a coding-agent application layer. It is designed for local OpenAI-compatible backends, SSH-friendly terminal use, MCP tools, plugin hooks, structured reasoning, skills, subagents, and real code editing workflows.

The current architecture follows the same broad split as Pi:

- `agent-core` owns the generic agent engine: backend contracts, messages, the ReAct loop, harness state, hooks, sessions, core types, compaction, and small shared primitives.
- `coding-agent` owns the coding-agent application layer: default tools, system prompt assembly, skills, MCP, subagents, reasoners, trust/config/context loading, slash-command logic, session storage, and the bridge used by the UI.
- `tui` owns terminal rendering, input handling, themes, overlays, transcript display, and interactive command UX.

## Workspace

```text
tui/
  package.json
  packages/
    agent-core/       Lean agent loop, harness, hooks, core types
    coding-agent/     Coding tools, skills, MCP, prompts, reasoners, bridge
    tui/              Terminal UI layer
```

Important package exports:

```ts
@logician/agent-core
@logician/agent-core/core/*
@logician/agent-core/hooks/*
@logician/agent-core/tools/*

@logician/coding-agent
@logician/coding-agent/tools
@logician/coding-agent/skills
@logician/coding-agent/mcp
@logician/coding-agent/reasoners/*
@logician/coding-agent/context-files
@logician/coding-agent/prompts
@logician/coding-agent/trust

@logician/tui
```

## Features

- Streaming terminal chat with live assistant text, reasoning text, tool progress, and structured tool results.
- Coding tools: `bash`, `read_file`, `write_file`, `edit_file`, `file_diff`, `grep`, `find`, `list_files`, `git`, `web_search`, `web_fetch`, and `read_skill`.
- Pi-style robust tool behavior: safer path normalization, stricter edit matching, CRLF/BOM preservation, better shell exit reporting, and clearer grep failures.
- Configurable permission modes: `acceptAll`, `acceptEdits`, `ask`, and `plan`.
- Skills loaded from `SKILL.md` files with YAML frontmatter and on-demand `read_skill` access.
- MCP support for stdio and streamable HTTP servers.
- Subagents with built-in `general` and `explorer` definitions plus user/plugin-defined agents.
- Optional structured reasoners including SSR, Tree of Thoughts, Reflexion, Self-Consistency, Best-of-N, Auto-CoT, and In-Context CoT.
- Session persistence, bookmarks, branch/fork workflows, rewind checkpoints, and manual/proactive compaction.
- Plugin hooks compatible with Claude-style lifecycle events.
- Themeable terminal UI with built-in and user themes.

## Quick Start

```bash
cd tui
npm install
npm start
```

For development:

```bash
cd tui
npm run dev
npm run typecheck
npm test
```

The default backend URL is `http://127.0.0.1:8080`, expected to expose an OpenAI-compatible chat API.

## Requirements

- Node.js `>=22.19.0`
- An OpenAI-compatible model backend
- Optional: `rg` and `fd` for faster search/list tools
- Optional: SearXNG for `web_search`
- Optional: MCP server configs

## Configuration

Logician reads config in this order:

1. `LOGICIAN_CONFIG`
2. nearest `.logician.json`, walking upward from the current directory
3. `~/.logician/settings.json`
4. environment variable overrides

Example `.logician.json`:

```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "model": "local-model",
  "theme": "dark",
  "systemPrompt": "Project-specific instructions.",
  "contextWindowTokens": 131072,
  "permissionMode": "acceptEdits",
  "mcpEager": true,
  "webSearch": {
    "baseUrl": "http://127.0.0.1:8090",
    "maxResults": 10
  },
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  },
  "mcpServers": {
    "context7": {
      "type": "streamable-http",
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

Common config keys:

| Key | Purpose |
| --- | --- |
| `baseUrl` / `llmUrl` | OpenAI-compatible backend URL |
| `model` | Model name sent to the backend |
| `theme` | Initial TUI theme |
| `systemPrompt` | Extra system instructions |
| `temperature` | Sampling temperature |
| `maxTokens` | Per-response token cap |
| `maxIterations` | Max loop iterations for a prompt |
| `contextWindowTokens` / `contextWindow` | Context-window size for estimates/status |
| `permissionMode` | `acceptAll`, `acceptEdits`, `ask`, or `plan` |
| `permissions` | Allow/deny tool rules |
| `toolExecution` | `sequential` or `parallel` |
| `hooks` | Enable plugin hooks |
| `mcpServers` / `mcp` | MCP server configuration |
| `mcpEager` | Load MCP tools at startup |
| `webSearch` | SearXNG web search settings |
| `compaction` | Proactive compaction settings |
| `plugins` | Plugin enabled/disabled state |

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `LOGICIAN_CONFIG` | Explicit config path |
| `LOGICIAN_LLM_URL` | Backend URL override |
| `LOGICIAN_MODEL` | Model override |
| `LOGICIAN_SYSTEM_PROMPT` | Extra system prompt override |
| `LOGICIAN_THEME` | Theme override |
| `LOGICIAN_CONTEXT_WINDOW` / `LOGICIAN_CTX_SIZE` | Context-window override |
| `LOGICIAN_MCP` | Set `0` to disable MCP |
| `LOGICIAN_MCP_CONFIG` / `MCP_CONFIG` | MCP config path |
| `LOGICIAN_MCP_EAGER` | Set `0` to defer MCP loading |
| `LOGICIAN_HOOKS` | Set `0` to disable runtime hooks |
| `LOGICIAN_SEARXNG_URL` | SearXNG URL for `web_search` |
| `LOGICIAN_SEARXNG_MAX_RESULTS` | Default web-search result count |
| `LOGICIAN_AGENTS_FILE` | Explicit AGENTS-style instruction file |
| `LOGICIAN_STARTUP_HOOK_TIMEOUT_MS` | Startup hook timeout |

## Slash Commands

Use `/help` in the TUI for the live command list. Current command groups include:

| Area | Commands |
| --- | --- |
| Sessions | `/new`, `/sessions`, `/save`, `/rename`, `/name`, `/bookmark`, `/bookmarks`, `/load`, `/export` |
| Agent | `/status`, `/agents`, `/agent`, `/pipeline`, `/reload` |
| Context | `/context`, `/compact`, `/fork`, `/branch-summary`, `/discard-branch`, `/reset`, `/changes` |
| RAG/docs | `/mount`, `/mount-code`, `/upload`, `/upload-dir`, `/docs`, `/rag` |
| Tools/config | `/skills-health`, `/plugins`, `/mcp`, `/theme`, `/reasoner`, `/thinking`, `/mode`, `/thinking-steps`, `/cache`, `/trace`, `/settings` |
| Permissions | `/permissions`, `/plan`, `/rewind` |
| Misc | `/loop`, `/version`, `/login`, `/jb`, `/clear`, `/quit`, `/q`, `/exit` |

## Themes

Built-in themes live in `tui/packages/tui/src/layers/theme` and custom themes can be placed under `~/.logician/themes/`.

Example:

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

Switch themes with:

```text
/theme dark
/theme light
/theme github-dark
```

## Skills

Skills are `SKILL.md` files with frontmatter. They are loaded from user, project, plugin, or explicit skill paths and exposed to the model through the system prompt and `read_skill`.

Example:

```md
---
name: code-review
description: Review code changes for correctness, risks, and missing tests.
allowed-tools: [read_file, grep, git]
argument-hint: <diff-or-topic>
---

Inspect the change as a reviewer. Prioritize bugs and regressions before style.
```

## Tools

The default coding-agent tool set is owned by `@logician/coding-agent/tools`, not `agent-core`.

| Tool | Purpose |
| --- | --- |
| `bash` | Run shell commands with timeout/abort handling |
| `read_file` | Read files with path normalization and CWD safety |
| `write_file` | Create/replace files with serialized file mutation |
| `edit_file` | Apply strict text edits with fuzzy normalization fallback |
| `file_diff` | Show file diffs |
| `grep` | Search content using ripgrep |
| `find` | Locate files using fd/find fallback |
| `list_files` | List directory entries safely |
| `git` | Read git status/diff/log data |
| `web_search` | Search via SearXNG |
| `web_fetch` | Fetch web content |
| `read_skill` | Load full instructions for a named skill |

## Development

Run checks from `tui/`:

```bash
npm run typecheck
npm test
```

Current test coverage is split across:

```text
tui/packages/agent-core/src/__tests__
tui/packages/coding-agent/src/__tests__
```

The root test script runs both packages:

```bash
tsx --test packages/agent-core/src/__tests__/*.test.ts packages/coding-agent/src/__tests__/*.test.ts
```

## Architecture Notes

The current split is intentionally moving toward a lean `agent-core`.

Already application-owned by `coding-agent`:

- coding tools
- skills
- MCP
- system prompt construction
- context file loading
- prompt templates
- project trust
- subagents
- reasoners
- session store and UI bridge

Still in `agent-core` because the harness directly coordinates them:

- compaction
- message queue
- built-in hook policy
- todo/workflow state
- extension primitives

Those remaining pieces are the next candidates for dependency inversion if `agent-core` needs to become even closer to Pi's minimal loop-plus-harness package.

## License

MIT
