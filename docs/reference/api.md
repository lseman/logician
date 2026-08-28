---
title: API Reference
description: Public API for the Logician agent.
---

# API Reference

Programmatic access to Logician's agent capabilities.

## Entry points

### TUI (interactive)

`@logician/tui` is a CLI, not a library — it has no programmatic entry
point. Launch it as a script; it reads its config from `.logician.json`
(searched upward from `cwd`), falling back to `~/.logician/settings.json`
(see [Config Schema](/reference/config)):

```sh
tsx apps/tui/src/index.ts
```

### Memory MCP server

`memoriam` ships its own stdio MCP server (`ecosystem/memoriam`,
`memoriam.mcp.server`) that exposes memory search, capture, and consolidation
to any MCP client, independent of the TUI or headless bridge. The database
path defaults to `~/.logician/memories.db` and can be overridden with the
`MEMORIAM_DB_PATH` environment variable.

```sh
ecosystem/memoriam/.venv/bin/python -m memoriam.mcp.server
```

See [ecosystem/memoriam/README.md](https://github.com/lseman/memoriam) for the
full tool list and MCP client configuration.

### Headless (programmatic)

The headless entry point is `AgentRuntime` from `@logician/log-runtime`'s
`application` export — the same bridge the TUI itself drives. It is
event-driven: subscribe through `bridge.events`, then call `sendMessage()`.

```typescript
import { AgentRuntime } from '@logician/log-runtime/application'

const bridge = new AgentRuntime({
  baseUrl: 'http://127.0.0.1:8080',
  model: 'gpt-4o',
  cwd: process.cwd(),
  permissions: { mode: 'acceptEdits' },
})

const unsubscribe = bridge.events.subscribe(notification => {
  if (notification.event.type === 'turn_end') console.log('done')
})
bridge.events.onError(err => console.error(err))

await bridge.sendMessage('fix the auth bug')
```

Other notable `AgentRuntime` methods: `steer()`, `steerQueue()`, `steerNow()`,
`followUp()`, `abort()`, `respondToQuestion()`, `respondToPermission()`,
`getSkills()` / `invokeSkill()`,
`getPrompts()` / `invokePrompt()`, `sendSlash()`, and
`setPermissionMode()` / `getPermissionMode()`.

## Event stream

```typescript
type ProtocolCallback = (notification: AgentProtocolNotification) => void
type ErrorCallback = (err: Error) => void
```

`RuntimeEvent` is a discriminated union keyed on `type`, exported from
`@logician/log-core/events`; `AgentProtocolNotification` is exported from
`@logician/log-core/protocol`. Subscribe through `bridge.events` to receive an
ordered envelope containing `protocolVersion`, `sequence`, `timestamp`,
optional `correlation`, and `event`. Correlation identifies the session, run,
turn, and tool call when available and is preserved during replay. Event
families include:

```typescript
type RuntimeEventType =
  | 'turn_start' | 'turn_end'
  | 'token' | 'thinking_token'
  | 'message_update' | 'agent_iteration_start'
  | 'tool_call_start' | 'tool_call_update' | 'tool_call_id_update'
  | 'tool_execution_start' | 'tool_execution_update' | 'tool_execution_end'
  | 'phase' | 'runtime_status' | 'context_update' | 'compaction'
  | 'queue_update' | 'repair_nudge'
  | 'question_request' | 'permission_request'
  | 'agent_retry_start' | 'agent_retry_end' | 'agent_error' | 'diagnostic'
  | 'model_select' | 'todos' | 'steered' | 'notice' | 'memory_update'
  | 'subagent_chunk' | 'subagent_lifecycle'
```

Each variant has its own payload shape (see
`packages/log-core/src/system/types/types-events.ts` for the full interfaces).
Most core events pass through `mapAgentEvent()`; bridge-owned features also
emit UI-facing events such as `todos`, `steered`, `notice`, and
`memory_update`. The [headless JSONL stream](/tutorials/headless) is a separate,
smaller versioned contract.

The event bus retains a bounded history for reconnecting or late clients:

```typescript
bridge.events.subscribe(handleNotification, {
  replay: { afterId: lastSeenSequence },
  onReplayGap: gap => resynchronizeFromSnapshot(gap),
})

const recent = bridge.events.snapshot({ types: ['notice', 'agent_error'] })
const cursor = bridge.events.latestSequence
bridge.events.clearHistory()
```

Replay cursors are protocol sequence numbers and remain monotonic after retained
history is cleared. `onReplayGap` reports the exact missing sequence range when
a reconnect cursor predates retained history. Configure retention with
`eventStream: { historyCapacity: 2_000 }` in `AgentBridgeOptions`.

## Configuration

`AgentRuntime` takes an `AgentBridgeOptions` object directly (as shown
above); the TUI CLI instead reads `.logician.json` (or the global
`~/.logician/settings.json` fallback) and maps it onto the same options via
`LogicianTuiConfig`. See [Config Schema](/reference/config) for the full
on-disk shape, including `permissionMode`, MCP, memory, and safeguard
options.
