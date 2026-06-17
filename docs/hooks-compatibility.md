# Hooks & Plugins: Claude Code Compatibility Reference

> **Logician's hook system is designed to be functionally equivalent to Claude Code's hook system.**
> Both systems support the same hook events and decision controls. This document maps
> between the two so you can port hooks between systems.

---

## Two Hook Layers

Logician has **two distinct hook layers**. Claude Code uses only the plugin event style; Logician adds a programmatic layer on top.

```
┌─────────────────────────────────────────────┐
│ Plugin Hook Events (JSON on stdin/stdout)   │
│ ─────────────────────────────────────────── │
│ 8 events: SessionStart, SessionEnd, Stop,   │
│   UserPromptSubmit, PreToolUse, PostToolUse,│
│   PreCompact, PostCompact                   │
│                                              │
│ Format: .claude-plugin/plugin.json +        │
│   hooks/hooks.json (same as Claude Code)    │
│                                              │
│ Executable types: command | prompt | http   │
│ Decision: exit 0 + JSON stdout, or exit 2   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ AgentLoopHooks (TypeScript callbacks)       │
│ ─────────────────────────────────────────── │
│ 9 methods: beforeToolCall, afterToolCall,   │
│   prepareNextTurn, transformContext,        │
│   shouldStopAfterTurn, beforeProviderRequest│
│   beforeProviderPayload, getSteeringMessages│
│   getFollowUpMessages                       │
│                                              │
│ Decision: return typed result or undefined  │
└─────────────────────────────────────────────┘
```

### When Each Layer Fires

| Layer | Trigger | Example Use |
|---|---|---|
| **Plugin events** | Session lifecycle, tool calls, compaction | Context injection, permission decisions, tool result rewriting |
| **AgentLoopHooks** | Model call cycle, provider boundary | Message pruning, steering injection, request header tuning |

---

## Plugin Events (8 events)

| Event | When |
|---|---|
| `SessionStart` | Startup, `/clear`, and compaction refresh sources |
| `UserPromptSubmit` | Before a user prompt reaches the model |
| `PreToolUse` | Before a tool executes |
| `PostToolUse` | After a tool returns |
| `Stop` | After an agent turn finishes |
| `SessionEnd` | Shutdown, reset, `/quit`, SIGINT, SIGTERM |
| `PreCompact` | Before context compaction |
| `PostCompact` | After context compaction |

### Plugin Hook Format

Plugins declare hooks in `.claude-plugin/plugin.json` and `hooks/hooks.json`:

**plugin.json:**
```json
{
  "name": "my-security-plugin",
  "version": "1.0.0",
  "hooks": "hooks/hooks.json"
}
```

**hooks/hooks.json:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "./block-rm.sh", "timeout": 10 }
        ]
      }
    ]
  }
}
```

### Plugin Hook Input (stdin JSON)

Each event receives a JSON object on stdin:

| Event | Fields |
|---|---|
| `SessionStart` | `session_id`, `transcript_path`, `cwd`, `source` |
| `SessionEnd` | `session_id`, `transcript_path`, `cwd`, `reason` |
| `Stop` | `session_id`, `transcript_path`, `cwd`, `stop_hook_active` |
| `UserPromptSubmit` | `session_id`, `prompt` |
| `PreToolUse` | `session_id`, `tool_name`, `tool_input` |
| `PostToolUse` | `session_id`, `tool_name`, `tool_input`, `tool_response` |
| `PreCompact` | `session_id`, `transcript_path`, `cwd` |
| `PostCompact` | `session_id`, `transcript_path`, `cwd` |

### Plugin Hook Output (stdout JSON)

| Output Field | Purpose |
|---|---|
| `additional_context` / `additionalContext` | Inject context for the model |
| `initial_user_message` / `initialUserMessage` | Rewrite the first user message |
| `watch_paths` / `watchPaths` | Declare file watch paths |
| `hookSpecificOutput.permissionDecision` | PreToolUse decision: `"allow"`, `"deny"`, `"ask"` |
| `hookSpecificOutput.permissionDecisionReason` | Reason for permission decision |
| `decision: "block"` | Generic block with reason |
| `exit code 2` | Block — stderr is the reason |

---

## AgentLoopHooks (9 programmatic methods)

These are typed TypeScript callbacks, not JSON-driven shell hooks.

### Method Signatures

```typescript
interface AgentLoopHooks {
  // Returns { content } to short-circuit (tool NOT run).
  // Returns { args } to rewrite tool input before it runs.
  beforeToolCall?: (
    ctx: BeforeToolCallContext,
  ) => Promise<BeforeToolCallResult | undefined>
    | BeforeToolCallResult
    | undefined;

  // Returns { content } / { isError } to rewrite the recorded result.
  // Returns { terminate: true } to stop the loop after this batch.
  afterToolCall?: (
    ctx: AfterToolCallContext,
  ) => Promise<AfterToolCallResult | undefined>
    | AfterToolCallResult
    | undefined;

  // Rewrite messages before the next model call.
  prepareNextTurn?: (
    ctx: PrepareNextTurnContext,
  ) => Promise<PrepareNextTurnResult | undefined>
    | PrepareNextTurnResult
    | undefined;

  // Rewrite the AgentMessage[] before each LLM call.
  transformContext?: (
    ctx: TransformContext,
  ) => Promise<TransformContextResult | undefined>
    | TransformContextResult
    | undefined;

  // Return true to stop after this turn.
  shouldStopAfterTurn?: (
    ctx: ShouldStopAfterTurnContext,
  ) => Promise<boolean | undefined> | boolean | undefined;

  // Inject per-request headers or tune timeout.
  beforeProviderRequest?: (
    ctx: BeforeProviderRequestContext,
  ) => Promise<BeforeProviderRequestResult | undefined>
    | BeforeProviderRequestResult
    | undefined;

  // Rewrite the raw request payload before serialization.
  beforeProviderPayload?: (
    ctx: BeforeProviderPayloadContext,
  ) => Promise<BeforeProviderPayloadResult | undefined>
    | BeforeProviderPayloadResult
    | undefined;

  // Inject queued steering `Message[]` before each assistant response.
  getSteeringMessages?: (
    ctx: GetSteeringMessagesContext,
  ) => Promise<Message[] | undefined> | Message[] | undefined;

  // Inject queued follow-up `Message[]` when the loop would stop.
  getFollowUpMessages?: (
    ctx: GetFollowUpMessagesContext,
  ) => Promise<Message[] | undefined> | Message[] | undefined;
}
```

### Decision Control Summary

| Method | Can Block | Can Rewrite | Can Stop |
|---|---|---|---|
| `beforeToolCall` | Yes — return `{ content }` | Tool args | No |
| `afterToolCall` | Yes — return `{ content }` | Tool result | Via `{ terminate: true }` |
| `prepareNextTurn` | — | Full message history | No |
| `transformContext` | — | Full AgentMessage[] | No |
| `shouldStopAfterTurn` | — | — | Yes — return `true` |
| `beforeProviderRequest` | — | Per-request headers/timeout | No |
| `beforeProviderPayload` | — | Raw request payload | No |
| `getSteeringMessages` | — | `Message[]` injected before each response | No |
| `getFollowUpMessages` | — | `Message[]` injected when loop would stop | No |

---

## Layer Mapping: Plugin Events ↔ AgentLoopHooks

| Plugin Event | Equivalent AgentLoopHook | Notes |
|---|---|---|
| `PreToolUse` | `beforeToolCall` | Both fire before tool execution |
| `PostToolUse` | `afterToolCall` | Both fire after tool returns |
| `Stop` | `shouldStopAfterTurn` | Both fire after turn completes |
| `UserPromptSubmit` | *(plugin events only)* | No AgentLoopHooks equivalent |
| `SessionStart` | *(plugin events only)* | No AgentLoopHooks equivalent |
| `SessionEnd` | *(plugin events only)* | No AgentLoopHooks equivalent |
| `PreCompact` | `prepareNextTurn` / `transformContext` | Both can rewrite messages |
| `PostCompact` | `transformContext` | Both fire before next LLM call |

**Key distinction:** `UserPromptSubmit`, `SessionStart`, and `SessionEnd` have no AgentLoopHooks equivalent — they are plugin events only, delivered via JSON on stdin. Use hooks.json for these.

---

## Decision Control: Claude Code vs Logician

### Blocking / Denying a Tool Call

**Claude Code (PreToolUse):**
```json
// stdout:
{ "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Destructive command blocked"
}}
// or stderr + exit 2:
echo "Destructive command blocked" >&2; exit 2
```

**Logician plugin hooks (PreToolUse):** Same JSON output on stdout. Identical contract.

**Logician AgentLoopHooks (beforeToolCall):**
```typescript
beforeToolCall: ({ toolCall, args }) => {
  if (toolCall.name === "Bash" && args.command?.includes("rm -rf")) {
    return { content: "Destructive command blocked", isError: true };
  }
  return undefined; // no change
}
```

### Rewriting Tool Arguments

**Claude Code:**
```json
{ "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "updatedInput": { "command": "npm run lint --fix" }
}}
```

**Logician AgentLoopHooks:**
```typescript
beforeToolCall: ({ toolCall, args }) => {
  if (toolCall.name === "Bash" && args.command?.startsWith("npm run lint")) {
    return { args: { ...args, command: args.command + " --fix" } };
  }
  return undefined;
}
```

### Blocking a Tool Call with Custom Result

**Claude Code:**
```json
{ "decision": "block", "reason": "Tool not allowed in plan mode" }
```

**Logician AgentLoopHooks:**
```typescript
beforeToolCall: ({ toolCall }) => {
  if (toolCall.name === "Bash" && planMode) {
    return { content: "Bash not allowed in plan mode", isError: true };
  }
  return undefined;
}
```

### Rewriting a Tool Result

**Claude Code (PostToolUse):**
```json
{ "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "updatedToolOutput": "Modified result text"
}}
```

**Logician AgentLoopHooks:**
```typescript
afterToolCall: ({ toolCall, result, isError }) => {
  if (toolCall.name === "Bash" && result.includes("SECRET_KEY")) {
    return { content: result.replace(/SECRET_KEY=\S+/g, "SECRET_KEY=***") };
  }
  return undefined;
}
```

### Injecting Context for the Model

**Claude Code (PostToolUse):**
```json
{ "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "This file is generated. Edit src/schema.ts instead."
}}
```

**Logician AgentLoopHooks (prepareNextTurn):**
```typescript
prepareNextTurn: ({ messages }) => {
  const lastResult = messages.at(-1);
  if (lastResult?.role === "tool" && lastResult.content?.includes("generated")) {
    return { messages: [
      ...messages,
      { role: "system", content: "This file is generated. Edit src/schema.ts instead." }
    ]};
  }
  return undefined;
}
```

### Blocking a Turn

**Claude Code (Stop):**
```json
{
  "decision": "block",
  "reason": "One more validation step is required.",
  "additionalContext": "Run the validation command before finishing."
}
```

When a `Stop` hook blocks, Logician injects the reason/context as a follow-up
user message and continues the loop. The next `Stop` hook invocation receives
`stop_hook_active: true`, and Logician will not recursively block again from
that active stop-hook continuation.

**Logician AgentLoopHooks (shouldStopAfterTurn):**
```typescript
shouldStopAfterTurn: ({ messages }) => {
  const hasErrors = messages.some(m => m.role === "tool" && m.isError);
  return hasErrors; // true = stop, false/undefined = continue
}
```

---

## Complete Example: Security Hook

### Claude Code version

**hooks/hooks.json:**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm -rf *)",
            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

**.claude/hooks/block-rm.sh:**
```bash
#!/bin/bash
if echo "$CLAUDE_PLUGIN_INPUT" | grep -q '"command":.*"rm -rf'; then
  echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Destructive rm -rf commands are not allowed"}}' >&2
  exit 2
fi
```

### Logician plugin version (identical format)

Same hooks.json. The shell script is identical — Logician uses the same JSON stdin/stdout contract.

### Logician AgentLoopHooks version (programmatic)

```typescript
import { AgentLoopHooks } from './agent-core/types';

const hooks: AgentLoopHooks = {
  beforeToolCall: ({ toolCall, args }) => {
    if (toolCall.name === "Bash" && args.command?.includes("rm -rf")) {
      return { content: "Destructive rm -rf commands are not allowed", isError: true };
    }
    return undefined;
  }
};

export default hooks;
```

---

## Plugin System Compatibility

### Directory Structure (Identical to Claude Code)

```
.plugin/
├── .claude-plugin/
│   ├── plugin.json          # name, version, hooks path, skills path
│   └── marketplace.json     # (optional) marketplace listing
├── hooks/
│   └── hooks.json           # hook definitions
├── skills/
│   └── my-skill/
│       └── SKILL.md         # skill prompt
└── agents/
    └── my-agent.yaml        # agent spec (optional)
```

### plugin.json Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Plugin identifier |
| `version` | string | No | Semantic version |
| `hooks` | string | No | Path to hooks JSON (relative to plugin dir) |
| `skills` | string | No | Path to skills directory |
| `owner` | string | No | Marketplace owner |
| `marketplace` | string | No | Marketplace name |
| `dependencies` | string[] | No | Plugin IDs this plugin depends on |
| `description` | string | No | Plugin description |

### Hook Types

| Type | Behavior |
|---|---|
| `command` | Execute a shell command, capture stdout/stderr/exit code |
| `prompt` | Inject a prompt string as additional context (no execution) |
| `agent` | Delegate to a named agent |
| `http` | Make an HTTP request, capture response |

### Hook Execution

- Commands run with `CLAUDE_PLUGIN_ROOT` env variable set to the plugin directory
- Timeout defaults to 30s (120s for startup hooks)
- Exit code 2 = blocking error (stderr = reason)
- Exit code 0 + JSON stdout = hook response
- JSON output is parsed for `additional_context`, `decision`, `hookSpecificOutput`, etc.

### Plugin Runtime Environment

Logician passes these environment variables to hook commands:

| Variable | Value |
|---|---|
| `CLAUDE_PLUGIN_ROOT` | Path to the plugin directory |
| `process.env` (inherited) | All parent process env vars |
| `pluginRuntimeEnv` | Configured runtime env via `configurePluginRuntimeEnv()` |

---

## Hook Bus: Multi-Handler Composition

Logician's `HookBus` allows multiple handlers per event with deterministic reducer semantics:

| Event | Reducer | Behavior |
|---|---|---|
| `beforeToolCall` | early-block | First `{content}` short-circuits; `{args}` rewrites thread |
| `afterToolCall` | patch-accumulate | Each handler sees the prior patch; non-undefined fields win |
| `prepareNextTurn` | transform | Messages thread through all handlers |
| `transformContext` | transform | Messages thread through all handlers |
| `shouldStopAfterTurn` | first-true | First `true` wins |
| `beforeProviderRequest` | merge | Headers merged, last timeout wins |
| `beforeProviderPayload` | transform | Payload threads through all handlers |
| `getSteeringMessages` | concat | Results concatenated |
| `getFollowUpMessages` | concat | Results concatenated |

### Error Handling

| Setting | Behavior |
|---|---|
| `errorMode: "continue"` | Failed handlers are skipped (default) |
| `errorMode: "throw"` | Failed handlers abort the chain |
| `timeoutMs` | Per-handler timeout; 0 = no timeout |
| `defaultTimeoutMs` | Bus-wide default timeout (default: 0 = no timeout) |

### Observer Pattern

The hook bus supports a read-only observer that sees every event:

```typescript
bus.observe((event, ctx) => {
  console.log(`${event}:`, ctx);
});
```

---

## Error & Timeout Configuration

### Bus-Level (AgentLoopHooks)

| Setting | Default | Description |
|---|---|---|
| `errorMode` | `"continue"` | How to handle handler errors |
| `defaultTimeoutMs` | `0` | Default per-handler timeout (ms) |
| `onError` | — | Callback for errors |

### Plugin-Level (hooks.json)

| Setting | Default | Description |
|---|---|---|
| `timeout` | `30s` (120s for startup) | Per-hook timeout in seconds |
| `if` | *(none)* | Condition pattern for filtering |
| `matcher` | *(none)* | Event matcher (e.g. `Bash`) |

---

## Quick Reference: Porting Hooks

### From Claude Code to Logician

1. **JSON hooks** → keep as-is. Logician reads the same `hooks/hooks.json` format.
2. **AgentLoopHooks** → write as TypeScript callbacks in your plugin's `hooks.ts`.
3. **Plugin skills** → copy `skills/` directory as-is.
4. **Plugin agents** → copy `agents/` directory as-is.

### From Logician AgentLoopHooks to Claude Code

| AgentLoopHooks method | Claude Code equivalent |
|---|---|
| `beforeToolCall` | `PreToolUse` hook with `permissionDecision` or exit 2 |
| `afterToolCall` | `PostToolUse` hook with `additionalContext` |
| `prepareNextTurn` | `PostCompact` hook with `additionalContext` |
| `shouldStopAfterTurn` | `Stop` hook with `decision: "block"` |
| `transformContext` | `PostCompact` hook with `additionalContext` |
| `beforeProviderRequest` | *(no equivalent — Logician-specific)* |
| `beforeProviderPayload` | *(no equivalent — Logician-specific)* |
| `getSteeringMessages` | *(no equivalent — Logician-specific)* |
| `getFollowUpMessages` | *(no equivalent — Logician-specific)* |
