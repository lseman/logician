# Logician TUI

Terminal agent UI for Logician: a TypeScript agent-core wrapped in a compact, SSH-friendly TUI.

[![npm version](https://img.shields.io/npm/v/@earendil-works/tui.svg)](https://www.npmjs.com/package/@earendil-works/tui)
[![Node.js >= 22](https://img.shields.io/badge/node-%3E%3D22.19-brightgreen.svg)](https://nodejs.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`tui` owns the interactive agent loop: it streams model output, routes tool calls, applies plugin hooks, tracks context, and renders thinking, response, and tool activity in chronological order.

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
│  │     AgentHarness (orchestration · phase machine)  │  │
│  │  ┌────────────┐ ┌────────────┐ ┌───────────────┐ │  │
│  │  │ Steering Q │ │ FollowUp Q │ │ NextTurn Q    │ │  │
│  │  └────────────┘ └────────────┘ └───────────────┘ │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │ prompt() compact() cycleModel() fork()       │  │  │
│  │  │ idle ↔ turn / compaction / branch_summary    │  │  │
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
- **Context management**: proactive compaction, micro-compaction, budget-based early stop, duplicate/failure-loop guards, and real provider token counts when available.
- **Resilient turns**: per-turn timeout that cancels the in-flight request, context-full compaction-retry, transient-error auto-retry, empty-response recovery, and well-formed transcripts on mid-batch abort.
- **Conversation branching**: fork the conversation, explore, then summarize the branch back into the parent or discard it.
- **AgentMessage abstraction**: union of standard LLM messages + custom app messages (notifications, status updates, UI-only artifacts). `convertToLlm` filters non-LLM messages before sending to the model. Apps extend via declaration merging.

## Quick Start

```bash
cd tui
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

Project config lives in `.logician.json`. The TUI resolves it in order: `LOGICIAN_CONFIG` (explicit file) → nearest `.logician.json` walking upward from the current directory → the per-user global `~/.logician/logician.json`. Environment variables still win over file values.

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

Two layers cooperate: the **AgentHarness** (outer, orchestration) drives prompts, queues, branching, and compaction; the **AgentLoop** (inner, ReAct) runs one prompt to completion.

### Inner loop (one turn)

1. **Drain steering** — `getSteeringMessages` injects queued mid-turn guidance before the model call.
2. **Transform context** — `transformContext` prunes / injects / drains the nextTurn queue; `convertToLlm` then filters non-LLM messages.
3. **Call backend** — streams the response under a **per-turn timeout that aborts the request on expiry** (no orphaned stream). Callbacks fire for text deltas, thinking deltas, tool-call start/delta, and partial message updates. The provider `finish_reason` and `usage` are captured.
4. **Recover** — context-full → compact once and retry; transient/rate-limit → exponential-backoff auto-retry (429/500/502/503/504); empty response → retry once, then nudge-and-continue or stop; truncated (`finish_reason: length`) is surfaced as the turn's stop reason.
5. **Execute tools** — sequential or parallel (read-only tools opt into parallel). `beforeToolCall` can short-circuit or rewrite args; `afterToolCall` can patch the result or request `terminate`. Pure tools (`cacheable: true`) may be served from the result cache. On abort mid-batch every requested tool call still gets a result so the transcript stays well-formed.
6. **Safeguards** — duplicate-call guard, failure-loop guard (signature/path/category buckets), budget-based early stop, proactive compaction — all ride contract hooks.
7. **Decide** — `prepareNextTurn` (rewrite history), `shouldStopAfterTurn` (end the loop), else `getFollowUpMessages` (auto-continue on unfinished todos). The unproductive-turn cap counts only turns that called no tools, so a long tool-using task is never truncated mid-work.

### Outer loop (the harness)

- **Phase state machine** — `idle → turn | compaction | branch_summary → idle`. Structural ops (prompt, compact, fork) are gated on `idle`; steering is gated on an active `turn`. One operation at a time.
- **Three queues** — `steer()` (into a running turn), `followUp()` (after the current turn), `nextTurn()` (before the next prompt, survives abort). The harness is the single source of truth; the UI reads snapshots via `onQueueChange`.
- **Runtime config setters** — system prompt, temperature, max tokens, tools, thinking level, model take effect on the **next** turn, never mutating an in-flight request.
- **Branching** — `fork()` snapshots the conversation; `branchSummary()` collapses the diverged tail into a summary merged back into the parent; `discardBranch()` drops it.
- **Reasoner pre-phase** — an optional structured reasoner (SSR / ToT / Reflexion / …) runs on the prompt before ReAct; its output is injected as a synthetic assistant message.

### Context Management

- **Compaction ladder** — one `compactToFit()` path shared by proactive (builtin hook, fires at 80% of the window) and context-full recovery: estimate → micro-compact → full summarizing pass if still over (targets 65% of the window).
- **Micro-compaction** — truncates oversized message bodies without LLM summarization (cheap, frequent).
- **Full compaction** — LLM-generated summary of older messages, replacing them with a single compacted summary node; manual `/compact` and `branchSummary()` use the same skeleton.
- **Real token counts** — `context_update` prefers the provider's reported `usage.total_tokens` over the local estimate when the backend streams it (`stream_options.include_usage`).
- **Budget stop** — ends the loop when per-turn token growth shows diminishing returns (opt-in).
- **Guards** — duplicate-call detection plus a failure-loop guard bucketed by call signature, target path, and error category.
- **Tool result cache** — opt-in per tool (`cacheable: true`). Off by default because most tools observe mutable state the agent itself changes between calls.

## Events

The loop emits a single typed `AgentEvent` stream (`agent-core/types.ts`). The TUI subscribes through `AgentBridge`; any consumer can subscribe via `loop.events.on(handler)`. The `EventEmitter` also keeps a bounded history (last 1000 events) for replay.

### Lifecycle / turn

| Event | Payload | Meaning |
| --- | --- | --- |
| `agent_start` | — | A prompt run began |
| `agent_end` | `messages` | Run finished; carries the final conversation |
| `turn_start` | `turnId` | A turn began |
| `turn_end` | `turnId, stopReason?, message?, toolResults?` | Turn completed — one event renders the whole turn |
| `phase` | `phase: "thinking" \| "tool" \| "idle"` | Loop activity sub-state |
| `max_iterations` | `iterations, limit` | Stopped on the unproductive-turn safety cap (not a clean finish) |

### Streaming / message

| Event | Payload | Meaning |
| --- | --- | --- |
| `message_start` | `turnId, role` | Assistant message starting |
| `text_start` / `text_end` | `turnId` | Text block boundaries |
| `text_delta` | `turnId, delta` | Streamed assistant token(s) |
| `thinking_delta` | `delta` | Streamed reasoning token(s) |
| `message_update` | `turnId, message` | Full partial message (for live re-render) |
| `message_end` | `turnId` | Assistant message complete |

### Tools

| Event | Payload | Meaning |
| --- | --- | --- |
| `tool_call_start` | `toolName, toolCallId, args` | Tool call known (fired early during streaming, then authoritatively before execution; UI dedups by id) |
| `tool_call_delta` | `toolCallId, delta` | Streamed tool-argument chunk |
| `tool_call_update` | `toolName, toolCallId, partialResult` | Long-running tool progress |
| `tool_call_end` | `toolName, toolCallId, result, isError?, details?` | Tool finished; `details` carries structured metadata (diffs, line counts) |

### Context / recovery

| Event | Payload | Meaning |
| --- | --- | --- |
| `context_update` | `tokens, maxTokens?, compacted?` | Context size — provider `usage` when reported, else estimate |
| `compaction` | `reason: "context_full" \| "manual", tokensBefore, tokensAfter` | History was compacted |
| `repair_nudge` | `repairStage, toolName?, message, turnId?` | Recovered from a malformed tool call / arg parse |
| `auto_retry_start` | `attempt, maxRetries, delayMs, error` | Retrying a transient provider error |
| `auto_retry_end` | `attempt, success` | Retry resolved |
| `model_select` | `model, index` | Active model cycled |
| `error` | `message, error?` | Recoverable error surfaced to the UI |

### Typical emit order (one tool-using turn)

```text
agent_start → phase(idle)
  turn_start → phase(thinking)
    context_update
    message_start(assistant)
      text_start → text_delta* → text_end
      tool_call_start → tool_call_delta*
    message_end
    phase(tool)
      tool_call_start (authoritative) → tool_call_update* → tool_call_end
  (loop continues: next turn_start …)
  turn_end(stopReason)            ← emitted once when the loop stops
phase(idle) → agent_end(messages)
```

## AgentMessage Abstraction

Standard messages use `MessageRole = "system" | "user" | "assistant" | "tool"`. Custom app messages extend via declaration merging:

```ts
declare module "@earendil-works/tui/agent-core/types" {
    interface CustomAgentMessages {
        notification: { role: "notification"; content: string; level: "info" | "warn" | "error" };
        status: { role: "status"; content: string };
    }
}
```

`convertToLlm()` filters non-standard-role messages before sending to the model. Override via `AgentConfig.convertToLlm` for custom filtering logic.

## Plugin Hooks

`tui` loads Claude-style plugins from the local Claude plugin registry. Hook commands receive:

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
| `afterToolCall` | After tool returns | Patch-accumulate: each handler sees the prior patch; `terminate` from any wins |
| `transformContext` | Before each model call (after steering injection, before `convertToLlm`) | Transform: messages thread through all handlers |
| `prepareNextTurn` | After a turn, before the next model call | Transform: messages thread through all handlers |
| `shouldStopAfterTurn` | After turn completes | First-true wins |
| `beforeProviderRequest` | Just before each provider request | Merge: per-request `headers` accumulate, last `timeoutMs` wins |
| `beforeProviderPayload` | With the fully-built request body | Transform: payload threads through all handlers |
| `getSteeringMessages` | Before each assistant response | Accumulate: collect all steering messages |
| `getFollowUpMessages` | When the loop would otherwise stop | Accumulate: collect follow-up messages for continuation |

> The `nextTurn` queue (messages inserted before the next user prompt, surviving abort) is drained by the harness through `transformContext` — there is no separate `getNextTurnMessages` hook.

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

interface TransformContext {
    messages: AgentMessage[];
    iteration: number;
    signal?: AbortSignal;
}                                   // returns { messages } to rewrite working context

interface BeforeProviderRequestContext {
    model: string;
    sessionId: string;
    iteration: number;
}                                   // returns { headers?, timeoutMs? }

interface BeforeProviderPayloadContext {
    model: string;
    payload: Record<string, unknown>;
}                                   // returns { payload } to rewrite the raw body
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
import { HookBus } from "@earendil-works/tui/agent-core/hook-bus";

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

MCP discovery checks `LOGICIAN_MCP_CONFIG`, then `LOGICIAN_CONFIG`, then walks upward from the current directory looking for `.logician.json`, `.mcp.json`, or `agent_config.json`, and finally falls back to the per-user global `~/.logician/logician.json`.

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
    ├── loop.ts               # Inner ReAct turn loop
    ├── harness.ts            # Outer orchestration: phase machine, queues, branching, compaction
    ├── events.ts             # AgentEvent emitter (bounded history)
    ├── hook-bus.ts           # Typed multi-handler hook bus
    ├── backend.ts            # OpenAI-compatible streaming backend (finish_reason, usage)
    ├── messages.ts           # Message creation, chat format, compaction ladder
    ├── default-tools.ts      # Built-in tool list
    ├── tool-cache.ts         # LRU+TTL result cache for opt-in pure tools
    ├── async-utils.ts        # withTimeout (cancelable) and shared async helpers
    ├── plugins.ts            # Claude-style plugin hooks
    ├── builtin-hooks.ts      # Safeguard hooks (guards, budget, compaction, follow-up)
    ├── guards.ts             # Duplicate + failure-loop guards
    ├── budget.ts             # Diminishing-returns budget tracker
    ├── parser.ts             # Tool call parsing
    ├── mcp.ts                # MCP server loading and registration
    ├── system-prompt.ts      # Default system prompt generation
    ├── skills.ts             # Skills catalog + on-demand read_skill
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
