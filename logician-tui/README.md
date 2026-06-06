# Logician TUI

Terminal TUI for [Logician](https://github.com/logician-ai/logician) — streaming-first, thinking-visible, SSH-ready.

[![npm version](https://img.shields.io/npm/v/@earendil-works/logician-tui.svg)](https://www.npmjs.com/package/@earendil-works/logician-tui)
[![Node.js >= 22](https://img.shields.io/badge/node-%3E%3D22.19-brightgreen.svg)](https://nodejs.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Streaming responses** — tokens render in real-time as they arrive
- **Thinking visibility** — collapsible thinking blocks with configurable levels and display modes
- **pi-style input** — grapheme-aware editing with undo/redo, kill ring, history navigation
- **Slash commands** — 30+ commands with fuzzy matching, Tab completion, and arrow navigation
- **Session management** — list, search, switch, and export sessions
- **Markdown rendering** — bold, italic, code blocks, lists, blockquotes, headings, JSON formatting
- **Animated status bar** — phase indicators, thinking level, cache status, context size
- **Todo bar** — inline task tracking with `/todo_write` integration
- **Plugin system** — manage and toggle plugins from the TUI
- **MCP support** — loads local stdio and HTTP MCP tools from project config
- **Agent-core** — built-in agent loop with tool registry, budget tracking, and guard rails

## Quick Start

```bash
# Run (requires running bridge or LLM server)
npm start

# Hot reload during development
npm run dev

# Build standalone binary
make build
```

### Prerequisites

- **Node.js >= 22.19.0**
- Running Logician bridge or LLM server (default: `http://127.0.0.1:8080`)

### Environment Variables

| Variable                  | Default                           | Description                        |
| ------------------------- | --------------------------------- | ---------------------------------- |
| `LOGICIAN_LLM_URL`        | `http://127.0.0.1:8080`           | LLM bridge URL                     |
| `LOGICIAN_MODEL`          | _(empty)_                         | Model name to use                  |
| `LOGICIAN_SYSTEM_PROMPT`  | _(empty)_                         | Custom system prompt               |
| `LOGICIAN_CONFIG`         | _(empty)_                         | Path to config file                |
| `LOGICIAN_MCP_CONFIG`     | `.mcp.json` / `agent_config.json` | MCP config file                    |
| `LOGICIAN_MCP`            | `1`                               | Set to `0` to disable MCP loading  |
| `LOGICIAN_MCP_EAGER`      | `1`                               | Set to `0` to defer MCP discovery  |
| `LOGICIAN_HOOKS`          | `1`                               | Set to `0` to disable plugin hooks |
| `LOGICIAN_CONTEXT_WINDOW` | _(empty)_                         | Context window size for status     |

### MCP

`logician-tui` discovers MCP servers from `LOGICIAN_MCP_CONFIG`, then walks upward from the current directory looking for `.mcp.json` or `agent_config.json`.
MCP discovery runs during startup by default. Set `LOGICIAN_MCP_EAGER=0` to defer discovery until the first agent turn or `/status`.

```json
{
    "mcpServers": {
        "ariadne": {
            "type": "stdio",
            "command": "/path/to/server",
            "args": ["mcp-server"]
        },
        "context7": {
            "type": "streamable-http",
            "url": "https://mcp.context7.com/mcp",
            "headers": {
                "Authorization": "Bearer ${CONTEXT7_API_KEY}"
            }
        }
    }
}
```

Discovered tools are exposed as `mcp__<server>__<tool>`, with unsafe characters converted to underscores.

### Plugin Hooks

`logician-tui` loads Claude-style plugin hooks from the installed plugin registry and passes the active `session_id`, `cwd`, and a JSONL `transcript_path` to hook commands. The hook transcript is written under `~/.logician/tui/sessions/...` so plugins can inspect the current session after a turn or during shutdown.

The TUI fires these hook events during normal agent work:

| Event              | When it runs                                                     |
| ------------------ | ---------------------------------------------------------------- |
| `SessionStart`     | Once on startup, with source `startup`                           |
| `UserPromptSubmit` | Before a user prompt is sent to the model                        |
| `PreToolUse`       | Immediately before a tool executes                               |
| `PostToolUse`      | Immediately after a tool returns                                 |
| `Stop`             | After an agent turn finishes, before the UI returns to idle      |
| `SessionEnd`       | During reset or shutdown, including `/quit`, SIGINT, and SIGTERM |

Tool hooks receive `tool_name`, `tool_input`, and a matcher value. The matcher includes both Logician tool names and Claude-style aliases where available, such as `read_file|Read`, `write_file|Write`, `edit_file|Edit`, `bash|Bash`, `rg_search|Grep`, `list_files|LS`, and `todo_write|TodoWrite`.

Hook commands may return plain text, which becomes additional context, or JSON hook output. `hookSpecificOutput.additionalContext`, `additional_context`, and `additionalContext` are supported. Control JSON such as `{"continue":true,"suppressOutput":true}` is accepted without being shown as context.

Use `/plugins hooks [startup|clear|compact|UserPromptSubmit|PreToolUse|PostToolUse|Stop|SessionEnd]` to inspect enabled hooks. Use `/plugins run-hooks [startup|clear|compact]` to manually refresh `SessionStart` hook context.

## Usage

```bash
# Pipe from bridge
logician-tui

# With Python bridge
python logician_bridge.py --tui

# With custom config
LOGICIAN_CONFIG=/path/to/config.json logician-tui

# Standalone binary after make build
./dist/logician
```

## Input Shortcuts

| Shortcut               | Action                     |
| ---------------------- | -------------------------- |
| `Enter`                | Submit message             |
| `Shift+Enter`          | New line                   |
| `Ctrl+Z` / `Ctrl+Y`    | Undo / Redo                |
| `Ctrl+O`               | Toggle tool details        |
| `Ctrl+W`               | Delete word backward       |
| `Ctrl+K`               | Delete to line end         |
| `Ctrl+U`               | Delete to line start       |
| `Ctrl+Left` / `Alt+b`  | Move word backward         |
| `Ctrl+Right` / `Alt+f` | Move word forward          |
| `Up` / `Down`          | History navigation         |
| `Tab`                  | Autocomplete slash command |
| `/`                    | Open slash command popup   |
| `Esc` / `Ctrl+C`       | Cancel input               |

## Slash Commands

| Category | Commands                                                                       |
| -------- | ------------------------------------------------------------------------------ |
| Help     | `/help`, `/?`, `/version`                                                      |
| Sessions | `/new`, `/sessions`, `/load`, `/export`                                        |
| Agent    | `/status`, `/agents`, `/agent`, `/pipeline`, `/reload`                         |
| Context  | `/context`, `/compact`, `/reset`, `/changes`                                   |
| RAG      | `/mount`, `/mount-code`, `/upload`, `/upload-dir`, `/docs`, `/rag`             |
| Skills   | `/skills-health`, `/plugins`                                                   |
| Display  | `/thinking [level]`, `/thinking-steps [mode]`, `/mode`, `/cache [on\|off]`, `/trace [on\|off]`, `/clear` |
| Auth     | `/login [provider]`                                                            |
| Exit     | `/q`, `/quit`, `/exit`                                                         |

Slash popup supports **fuzzy matching**, **Tab completion**, **arrow navigation**, and **usage hints**.

### Context Compaction

The status bar shows estimated context size as `ctx <tokens>` or `ctx <tokens>/<window>` when `LOGICIAN_CONTEXT_WINDOW` or `LOGICIAN_CTX_SIZE` is set. If a llama.cpp/OpenAI-compatible backend returns a context-full error, `logician-tui` automatically compacts older in-flight messages and retries the model call once.

## Architecture

```
┌─────────────────────────────────────┐
│         Logician TUI                │
│  ┌───────────┐  ┌───────────────┐  │
│  │ Status Bar│  │ Transcript    │  │
│  │ (top)     │  │ Display       │  │
│  └───────────┘  └───────────────┘  │
│  ┌───────────┐  ┌───────────────┐  │
│  │ Thinking  │  │ Input Bar     │  │
│  │ Panel     │  │ (bottom)      │  │
│  └───────────┘  └───────────────┘  │
│  ┌───────────┐                     │
│  │ Slash/    │ (overlay)           │
│  │ Session   │                     │
│  └───────────┘                     │
└─────────────────────────────────────┘
           ↕ stdin/stdout JSON
┌─────────────────────────────────────┐
│       Bridge Layer                  │
│  - Reads events from stdin          │
│  - Writes commands to stdout        │
└─────────────────────────────────────┘
           ↕ Python bridge
┌─────────────────────────────────────┐
│     logician_bridge.py              │
│  - Event emission                   │
│  - Agent execution                  │
│  - Tool execution                  │
└─────────────────────────────────────┘
```

## Source Layout

```
src/
├── index.ts              # Entry point
├── tui.ts                # Main TUI orchestrator
├── tui-core.ts           # Differential rendering engine
├── bridge.ts             # JSON-RPC bridge to Python
├── agent-bridge.ts       # Bridge to Logician agent-core
├── transcript.ts         # Conversation state management
├── events.ts             # Event type definitions
├── slash-commands.ts     # Command definitions + fuzzy filter
├── undo-stack.ts         # Undo/redo support
├── kill-ring.ts          # Emacs-style kill ring
├── utils.ts              # Text wrapping, visible width, fuzzy match
├── word-navigation.ts    # Grapheme-aware word boundaries
├── components/
│   ├── input-bar.ts      # Full-featured input with history
│   ├── slash-popup.ts    # Slash command overlay with fuzzy search
│   ├── session-manager.ts # Session listing and switching
│   ├── status-bar.ts     # Phase indicator with animation
│   ├── thinking-panel.ts # Collapsible thinking blocks
│   ├── transcript-display.ts # Message rendering with markdown
│   ├── plugin-manager.ts # Plugin management overlay
│   └── todo-bar.ts       # Inline task tracking
└── agent-core/
    ├── index.ts          # Agent core entry
    ├── loop.ts           # Main agent loop
    ├── parser.ts         # Response parsing
    ├── backend.ts        # Backend abstraction
    ├── messages.ts       # Message types
    ├── events.ts         # Agent events
    ├── types.ts          # Shared types
    ├── plugins.ts        # Plugin system
    ├── system-prompt.ts  # System prompt builder
    ├── default-tools.ts  # Default tool definitions
    ├── budget.ts         # Token budget tracking
    ├── guards.ts         # Agent guard rails
    ├── mcp.ts            # MCP server integration
    ├── builtin-hooks.ts  # Built-in hook definitions
    └── tools/
        ├── registry.ts   # Tool registry
        ├── bash.ts       # Shell execution
        ├── git.ts        # Git operations
        ├── edit-file.ts  # File editing
        ├── write-file.ts # File writing
        ├── file-diff.ts  # File diffing
        ├── read-file.ts  # File reading
        ├── list-files.ts # Directory listing
        ├── search.ts     # Content search
        ├── find.ts       # File search (ripgrep)
        ├── todo-write.ts # Todo management
        ├── read-tracker.ts  # File read tracking
        ├── file-mutation-queue.ts # Coordinated file writes
        ├── truncate.ts   # Context truncation
        └── helpers.ts    # Tool utilities
```

## Key Design Decisions

### Differential Rendering

Components render only what changed — battle-tested from the pi TUI. Full redraws are rare.

### Grapheme-Aware Input

Handles emoji, CJK, and combining characters correctly. Word navigation respects Unicode boundaries.

### Event-Driven Architecture

Bridge emits structured JSON events. Components subscribe and react. Transcript is the single source of truth.

### Streaming-First

Tokens stream in real-time via `token` events. Thinking blocks stream separately via `thinking_token`. Phase indicators show current agent state.

## License

MIT
