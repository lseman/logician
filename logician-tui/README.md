# Logician TUI

Terminal agent UI for Logician: a TypeScript agent-core wrapped in a compact, SSH-friendly TUI.

[![npm version](https://img.shields.io/npm/v/@earendil-works/logician-tui.svg)](https://www.npmjs.com/package/@earendil-works/logician-tui)
[![Node.js >= 22](https://img.shields.io/badge/node-%3E%3D22.19-brightgreen.svg)](https://nodejs.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`logician-tui` is no longer just a renderer in front of a Python bridge. The app owns the interactive loop: it streams model output, routes tool calls, runs MCP tools, applies Claude-style plugin hooks, tracks context, and renders thinking, response, and tool activity in the order it actually happened.

## What It Does

- **Ordered transcript**: thinking, response, and tool chunks render chronologically instead of being grouped after the fact.
- **Streaming answers**: assistant tokens and reasoning tokens update live.
- **Thinking controls**: choose the thinking budget and display thinking as collapsed, summary, or expanded.
- **Useful tool display**: tools are compact by default, with `Ctrl+O` for args, output, command logs, diffs, and write/edit details.
- **Markdown for terminal output**: headings, lists, code, JSON, markdown tables, and compact memory-summary tables render cleanly.
- **Plugin hooks**: Claude-style hooks can inject startup context, react to prompts/tools/stops, and inspect a JSONL transcript.
- **MCP support**: stdio and streamable HTTP MCP servers are discovered from local config and exposed as tools.
- **Pi-style input**: Unicode-aware editing, undo/redo, kill ring, word navigation, history, and slash autocomplete.
- **Status + todo bars**: phase, model, branch, cache, context size, and active todos stay visible without stealing space.

## Quick Start

```bash
cd logician-tui
npm install
npm start
```

For development:

```bash
npm run dev
npm run typecheck
npx eslint src/components/transcript-display.ts
```

Build a standalone binary:

```bash
make build
./dist/logician
```

## Requirements

- Node.js `>=22.19.0`
- An OpenAI-compatible chat endpoint, usually local llama.cpp/vLLM/etc.
- Optional: MCP configs, Claude plugin registry, and `claude-mem`.

By default the TUI points at `http://127.0.0.1:8080`.

## Configuration

Project config lives in `.logician.json`. The TUI searches upward from the current directory, or you can point `LOGICIAN_CONFIG` at an explicit file. Environment variables still win over file values.

```json
{
    "baseUrl": "http://127.0.0.1:8080",
    "model": "local-model",
    "systemPrompt": "Extra project instructions.",
    "contextWindowTokens": 131072,
    "hooks": true,
    "mcpEager": true,
    "webSearch": {
        "baseUrl": "http://127.0.0.1:8090",
        "maxResults": 10
    },
    "mcpServers": {
        "context7": {
            "type": "streamable-http",
            "url": "https://mcp.context7.com/mcp"
        }
    }
}
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOGICIAN_LLM_URL` | `http://127.0.0.1:8080` | OpenAI-compatible backend URL |
| `LOGICIAN_MODEL` | empty | Model name sent to the backend |
| `LOGICIAN_SYSTEM_PROMPT` | empty | Extra system instructions appended to the default prompt |
| `LOGICIAN_CONFIG` | auto `.logician.json` | Explicit config file path |
| `LOGICIAN_CONTEXT_WINDOW` | empty | Context window shown in the status bar |
| `LOGICIAN_CTX_SIZE` | empty | Alias used for context-window status |
| `LOGICIAN_MCP` | `1` | Set `0` to disable MCP loading |
| `LOGICIAN_MCP_CONFIG` | auto | MCP config file path |
| `LOGICIAN_MCP_EAGER` | `1` | Set `0` to defer MCP discovery |
| `LOGICIAN_HOOKS` | `1` | Set `0` to disable runtime plugin hooks |
| `LOGICIAN_STARTUP_HOOK_TIMEOUT_MS` | `1200` | Startup hook command timeout |

## Backend

The built-in backend expects OpenAI-compatible chat completions and supports:

- token streaming through `delta.content`
- reasoning streaming through `delta.reasoning_content`
- tool calls from the agent-core tool registry
- one retry after context compaction when the backend reports a context-full error

The default tool set includes file read/write/edit, diff, list/search/find, bash, git, web fetch/search, todo writing, and MCP tools when configured.

## MCP

MCP discovery checks `LOGICIAN_MCP_CONFIG`, then `LOGICIAN_CONFIG`, then walks upward from the current directory looking for `.logician.json`, `.mcp.json`, or `agent_config.json`.

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

MCP tools are registered as `mcp__<server>__<tool>`, with unsafe characters normalized to underscores.

## Plugin Hooks

`logician-tui` loads Claude-style plugins from the local Claude plugin registry. Hook commands receive:

- `session_id`
- `cwd`
- `transcript_path`
- hook-specific fields such as prompt, tool name, tool input, or stop state

The hook transcript is JSONL under `~/.logician/tui/sessions/...`.

Supported events:

| Event | When |
| --- | --- |
| `SessionStart` | Startup, `/clear`, and compaction refresh sources |
| `UserPromptSubmit` | Before a user prompt reaches the model |
| `PreToolUse` | Before a tool executes |
| `PostToolUse` | After a tool returns |
| `Stop` | After an agent turn finishes |
| `SessionEnd` | Shutdown, reset, `/quit`, SIGINT, SIGTERM |
| `getSteeringMessages` | Before each assistant response — injects queued steering messages |
| `getFollowUpMessages` | When the loop would otherwise stop — injects queued follow-up messages |

Loop hooks route through `AgentHarness`: `steer()` pushes into the steering queue (drained via `getSteeringMessages` at each assistant response), and `followUp()` pushes into the follow-up queue (drained via `getFollowUpMessages` when the loop would stop).

Hook output may be plain text or JSON. Supported context fields include `hookSpecificOutput.additionalContext`, `additional_context`, and `additionalContext`. Control JSON such as `{"continue":true,"suppressOutput":true}` is consumed without being rendered as startup text.

Useful commands:

```text
/plugins
/plugins list
/plugins hooks startup
/plugins run-hooks startup
```

## Controls

| Shortcut | Action |
| --- | --- |
| `Enter` | Submit |
| `Shift+Enter` | Insert newline |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |
| `Ctrl+O` | Expand or collapse tool details |
| `Ctrl+L` | Force redraw |
| `Ctrl+T` | Cycle thinking display mode |
| `Ctrl+W` | Delete word backward |
| `Ctrl+K` | Delete to line end |
| `Ctrl+U` | Delete to line start |
| `Ctrl+Left` / `Alt+b` | Move word backward |
| `Ctrl+Right` / `Alt+f` | Move word forward |
| `Up` / `Down` | History navigation |
| `/` | Open slash command popup |
| `Tab` | Accept slash completion |
| `Esc` / `Ctrl+C` | Cancel input |

## Slash Commands

| Area | Commands |
| --- | --- |
| Help | `/help`, `/?`, `/version` |
| Sessions | `/new`, `/sessions`, `/load`, `/export` |
| Agent | `/status`, `/agents`, `/agent`, `/pipeline`, `/reload` |
| Context | `/context`, `/compact`, `/reset`, `/changes` |
| RAG/docs | `/mount`, `/mount-code`, `/upload`, `/upload-dir`, `/docs`, `/rag` |
| Skills/plugins | `/skills-health`, `/plugins` |
| Display | `/thinking`, `/thinking-steps`, `/mode`, `/cache`, `/trace`, `/clear` |
| Auth | `/login` |
| Exit | `/q`, `/quit`, `/exit` |

The slash popup supports fuzzy matching, arrow navigation, usage hints, and Tab completion.

## Transcript Model

Assistant messages are stored as ordered chunks:

```ts
type AssistantChunk =
    | { type: "thinking"; contentText: string }
    | { type: "content"; contentText: string }
    | { type: "tool"; tool: ToolExecution };
```

This lets the UI show the real sequence:

```text
THINK reasoning...
── response ──
I will inspect the file.
TOOL read_file (done)
The bug is here...
```

Content chunks are buffered before block rendering so markdown tables and code fences still render correctly even when streamed in pieces.

## Source Layout

```text
src/
├── index.ts                  # Entry point
├── tui.ts                    # Main app orchestration
├── tui-core.ts               # Differential terminal renderer
├── agent-bridge.ts           # UI facade over agent-core
├── transcript.ts             # Ordered conversation state
├── events.ts                 # Bridge event types
├── slash-commands.ts         # Command definitions and fuzzy filter
├── input helpers             # undo-stack, kill-ring, word-navigation, utils
├── components/
│   ├── input-bar.ts
│   ├── transcript-display.ts
│   ├── status-bar.ts
│   ├── todo-bar.ts
│   ├── slash-popup.ts
│   ├── plugin-manager.ts
│   └── session-manager.ts
└── agent-core/
    ├── loop.ts               # Agent turn loop
    ├── backend.ts            # OpenAI-compatible backend
    ├── default-tools.ts      # Built-in tool list
    ├── plugins.ts            # Claude-style plugin hooks
    ├── mcp.ts                # MCP loading and registration
    ├── budget.ts             # Context tracking
    ├── guards.ts             # Guardrails
    └── tools/                # Tool implementations
```

## Development Notes

- Keep the terminal interaction dense and practical.
- Avoid broad rewrites in the renderer; small changes can affect scrolling and redraw stability.
- Run `npm run typecheck` after TypeScript changes.
- Use targeted smoke checks for rendering behavior instead of launching the interactive TUI in tests.
- The worktree may contain local experiments; keep patches scoped.

## License

MIT
