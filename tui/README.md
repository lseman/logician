# Logician TUI

Terminal agent UI for Logician: a TypeScript agent-core wrapped in a compact, SSH-friendly TUI.

[![npm version](https://img.shields.io/npm/v/@earendil-works/logician-tui.svg)](https://www.npmjs.com/package/@earendil-works/logician-tui)
[![Node.js >= 22](https://img.shields.io/badge/node-%3E%3D22.19-brightgreen.svg)](https://nodejs.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`logician-tui` owns the interactive agent loop: it streams model output, routes tool calls, applies plugin hooks, tracks context, and renders thinking, response, and tool activity in chronological order.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  TUI Layer                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ │
│  │ Input Bar│ │Transcript│ │ Status Bar│ │ Todo Bar   │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘ │
│         ▲              ▲              ▲                 │
│         │              │              │                 │
│  ┌──────┴──────────────┴──────────────┴──────────────┐  │
│  │              AgentBridge (UI facade)              │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                                │
├─────────────────────────┼────────────────────────────────┤
│                         │                                │
│  ┌──────────────────────┴────────────────────────────┐  │
│  │              AgentHarness (orchestration)         │  │
│  │  ┌────────────┐ ┌────────────┐ ┌───────────────┐ │  │
│  │  │ Steering Q │ │ FollowUp Q │ │ NextTurn Q    │ │  │
│  │  └────────────┘ └────────────┘ └───────────────┘ │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  prompt() │ compact() │ cycleModel()        │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                                │
├─────────────────────────┼────────────────────────────────┤
│                         │                                │
│  ┌──────────────────────┴────────────────────────────┐  │
│  │              AgentLoop (ReAct loop)               │  │
│  │  ┌────────────┐ ┌────────────┐ ┌───────────────┐ │  │
│  │  │  Backend   │ │  Tools     │ │  Hooks Bus    │ │  │
│  │  │ (OpenAI    │ │ (Registry) │ │ (typed bus)   │ │  │
│  │  │  compat)   │ │            │ │               │ │  │
│  │  └────────────┘ └────────────┘ └───────────────┘ │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## What It Does

- **Ordered transcript**: thinking, response, and tool chunks render chronologically instead of being grouped after the fact.
- **Streaming**: assistant tokens, reasoning tokens, and tool output update live.
- **Thinking controls**: choose the thinking budget and display thinking as collapsed, summary, or expanded.
- **Tool display**: tools are compact by default, with `Ctrl+O` for args, output, command logs, diffs, and write/edit details.
- **Markdown rendering**: headings, lists, code, JSON, markdown tables, and compact memory-summary tables render cleanly in the terminal.
- **Plugin hooks**: Claude-style hooks inject startup context, react to prompts/tools/stops, and inspect a JSONL transcript.
- **Typed hook bus**: multiple extensions register handlers per event with deterministic reducer semantics (short-circuit, patch-accumulate, transform, first-true).
- **MCP support**: stdio and streamable HTTP MCP servers are discovered from local config and exposed as tools.
- **Pi-style input**: Unicode-aware editing, undo/redo, kill ring, word navigation, history, and slash autocomplete.
- **Status + todo bars**: phase, model, branch, cache, context size, and active todos stay visible without stealing space.
- **Context management**: proactive compaction, micro-compaction, budget-based early stop, and duplicate/failure-loop guards.
- **AgentMessage abstraction**: union of standard LLM messages + custom app messages (notifications, status updates, UI-only artifacts). `convertToLlm` filters non-LLM messages before sending to the model. Apps extend via declaration merging.

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

## Agent Loop

The loop runs a ReAct cycle:

1. **Drain queues** — steering messages (mid-turn guidance), follow-up messages (post-stop continuation), next-turn messages (persist across runs).
2. **Estimate context** — emit token count for the status bar.
3. **Call backend** — stream response with callbacks for text deltas, thinking deltas, tool call start/delta, and message updates.
4. **Execute tools** — sequential or parallel, with `beforeToolCall`/`afterToolCall` hooks for short-circuiting, argument rewriting, and result patching.
5. **Run safeguards** — duplicate tool guard, failure-loop guard, budget-based early stop, proactive compaction.
6. **Contract hooks** — `prepareNextTurn` (rewrite messages before next call), `shouldStopAfterTurn` (decide to end the loop), `getFollowUpMessages` (auto-continue on unfinished todos).
7. **Auto-retry** — on provider errors (429, 500, 502, 503, 504) with exponential backoff.

### Context Management

- **Proactive compaction** — runs before hitting the context wall (default: at 80% of window).
- **Micro-compaction** — truncates oversized message bodies without LLM summarization.
- **Full compaction** — LLM-generated summary of older messages, replaces them with a compacted summary node.
- **Budget stop** — ends the loop when per-turn token growth shows diminishing returns.
- **Guards** — duplicate tool call detection and failure-loop detection.

### AgentMessage Abstraction

Standard messages use `MessageRole = "system" | "user" | "assistant" | "tool"`. Custom app messages extend via declaration merging:

```ts
declare module "@earendil-works/logician-tui/agent-core/types" {
    interface CustomAgentMessages {
        notification: { role: "notification"; content: string; level: "info" | "warn" | "error" };
        status: { role: "status"; content: string };
    }
}
```

`convertToLlm()` filters non-standard-role messages before sending to the model. Override via `AgentConfig.convertToLlm` for custom filtering logic.

## Plugin Hooks

`logician-tui` loads Claude-style plugins from the local Claude plugin registry. Hook commands receive:

- `session_id`
- `cwd`
- `transcript_path`
- hook-specific fields such as prompt, tool name, tool input, or stop state

The hook transcript is JSONL under `~/.logician/tui/sessions/...`.

### Plugin Events

| Event | When |
| --- | --- |
| `SessionStart` | Startup, `/clear`, and compaction refresh sources |
| `UserPromptSubmit` | Before a user prompt reaches the model |
| `PreToolUse` | Before a tool executes |
| `PostToolUse` | After a tool returns |
| `Stop` | After an agent turn finishes |
| `SessionEnd` | Shutdown, reset, `/quit`, SIGINT, SIGTERM |

### Contract Hooks

First-class extension points on `AgentLoopHooks`. Multiple extensions register handlers per event; the typed hook bus composes them deterministically:

| Hook | When | Reducer |
| --- | --- | --- |
| `beforeToolCall` | Before tool execution | Short-circuit: first `{content}` blocks; `{args}` rewrites input |
| `afterToolCall` | After tool returns | Patch-accumulate: each handler sees the prior patch |
| `prepareNextTurn` | Before next model call | Transform: messages thread through all handlers |
| `shouldStopAfterTurn` | After turn completes | First-true wins |
| `getSteeringMessages` | Before assistant response | Accumulate: collect all steering messages |
| `getFollowUpMessages` | When loop would stop | Accumulate: collect follow-up messages for continuation |
| `getNextTurnMessages` | Before user's next explicit prompt | Accumulate: persist across turns |

#### Reducer semantics

- **Short-circuit** (`beforeToolCall`): first handler returning `{content}` short-circuits — the tool is NOT run; the content is used as the result. `{args}` rewrites the input for downstream handlers.
- **Patch-accumulate** (`afterToolCall`): each handler sees the prior patch; non-undefined fields overwrite.
- **Transform** (`prepareNextTurn`): messages thread through all handlers; the last `{messages}` wins.
- **First-true** (`shouldStopAfterTurn`): first handler returning `true` stops the loop.
- **Accumulate** (`getSteeringMessages`, `getFollowUpMessages`): all handlers' results are concatenated.

#### Hook interface types

```ts
interface BeforeToolCallContext {
    toolCall: ToolCall;     // { id, name, arguments }
    args: Record<string, unknown>;
    iteration: number;
}
interface BeforeToolCallResult {
    content?: string;       // short-circuit: tool NOT run
    isError?: boolean;
    args?: Record<string, unknown>;  // rewrite tool input
}

interface AfterToolCallContext {
    toolCall: ToolCall;
    args: Record<string, unknown>;
    result: string;
    isError: boolean;
    iteration: number;
}
interface AfterToolCallResult {
    content?: string;       // rewrite recorded tool result
    isError?: boolean;
    terminate?: boolean;    // stop after current tool batch (all must set true)
}

interface PrepareNextTurnContext {
    messages: Message[];
    iteration: number;
    hadToolCalls: boolean;
    continuationCount: number;
    isContinuation: boolean;
}
interface PrepareNextTurnResult {
    messages: Message[];    // rewrite working history
}

interface ShouldStopAfterTurnContext {
    messages: Message[];
    iteration: number;
    hadToolCalls: boolean;
    continuationCount: number;
    isContinuation: boolean;
}

interface GetSteeringMessagesContext {
    messages: Message[];
    iteration: number;
}
interface GetFollowUpMessagesContext {
    messages: Message[];
    iteration: number;
    assistantText: string;
    continuationCount: number;
    maxContinuations: number;
}
```

All hooks are async-compatible and return `undefined` to pass through unchanged.

#### Hook output format

Hook output may be plain text or JSON. Supported context fields:

| Field | Source | Purpose |
| --- | --- | --- |
| `hookSpecificOutput.additionalContext` | Plugin response | Inject into session context |
| `hookSpecificOutput.additional_context` | Plugin response | Alias for above |
| `hookSpecificOutput.additionalContext` | Plugin response | CamelCase variant |
| `additionalContext` / `additional_context` | Top-level | Context injection |
| `initialUserMessage` / `initial_user_message` | SessionStart | Override first user message |
| `watchPaths` / `watch_paths` | SessionStart | File-watch paths for live reload |

Control JSON such as `{"continue":true,"suppressOutput":true}` is consumed without being rendered as startup text.

#### Error handling

The hook bus has an `errorMode` setting:
- `"continue"` (default): a thrown handler is skipped; the chain proceeds.
- `"throw"`: a thrown handler aborts the entire chain and the turn.

Observers (read-only firehose via `hookBus.observe()`) never affect a turn even if they throw.

#### Registering hooks programmatically

```ts
import { HookBus } from "@earendil-works/logician-tui/agent-core/hook-bus";

const bus = new HookBus({ errorMode: "continue" });
const unsubscribe = bus.register(
    {
        beforeToolCall: async (ctx) => {
            if (ctx.toolCall.name === "dangerous_tool") {
                return { content: "Blocked by policy.", isError: true };
            }
            return undefined;
        },
        shouldStopAfterTurn: (ctx) => ctx.iteration > 10,
    },
    { source: "my-extension" }
);
// Later: unsubscribe();
```

#### Plugin manifest hooks field

Plugins declare hooks in `.claude-plugin/plugin.json`:

```json
{
    "name": "my-plugin",
    "hooks": {
        "SessionStart": [
            {
                "matcher": "*",
                "hooks": [
                    { "type": "command", "command": "node start-context.js" }
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "read_file|write_file",
                "hooks": [
                    { "type": "prompt", "prompt": "Reading file: {tool_input}" }
                ]
            }
        ]
    }
}
```

Hook commands have four types:

| Type | Trigger | Config |
| --- | --- | --- |
| `command` (default) | Shell exec | `command`, `timeout` (seconds) |
| `prompt` | Inject text | `prompt` (string) |
| `http` | HTTP GET | `url`, `headers`, `timeout` |
| `agent` | Reserved | `agent` (string) |

### Hook payload shapes

Each event sends a JSONL payload to the hook command:

| Event | Payload fields |
| --- | --- |
| `SessionStart` | `{ session_id, cwd, transcript_path, source }` |
| `UserPromptSubmit` | `{ session_id, cwd, transcript_path, prompt }` |
| `PreToolUse` | `{ session_id, cwd, transcript_path, tool_name, tool_input }` |
| `PostToolUse` | `{ session_id, cwd, transcript_path, tool_name, tool_input, tool_response }` |
| `Stop` | `{ session_id, cwd, transcript_path, stop_hook_active }` |
| `SessionEnd` | `{ session_id, cwd, transcript_path, reason }` |

Source values: `startup`, `clear`, `compact`, `model-select`, etc.

Useful commands:

```text
/plugins
/plugins list
/plugins hooks startup
/plugins run-hooks startup
```

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
    ├── types.ts              # Core types: Message, AgentMessage, AgentEvent, hooks
    ├── loop.ts               # Agent turn loop
    ├── harness.ts            # Orchestration layer (queues, config, compaction)
    ├── hook-bus.ts           # Typed multi-handler hook bus
    ├── backend.ts            # OpenAI-compatible backend
    ├── messages.ts           # Message creation, chat format, compaction
    ├── default-tools.ts      # Built-in tool list
    ├── plugins.ts            # Claude-style plugin hooks
    ├── builtin-hooks.ts      # Safeguard hooks (guards, budget, compaction)
    ├── guards.ts             # Duplicate + failure-loop guards
    ├── budget.ts             # Context tracking
    ├── parser.ts             # Tool call parsing
    ├── mcp.ts                # MCP server loading and registration
    ├── system-prompt.ts      # Default system prompt generation
    ├── skills.ts             # Skills management
    ├── syntax-highlighter.ts # Terminal syntax highlighting
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
