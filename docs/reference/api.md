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

### Headless (programmatic)

The headless entry point is `AgentCoreBridge` from `@logician/agent-core`'s
`application` export—the same bridge the TUI itself drives. It is
event-driven: subscribe with `onNotification()`/`onError()`, then call
`sendMessage()`.

```typescript
import { AgentCoreBridge } from '@logician/agent-core/application'

const bridge = new AgentCoreBridge({
  baseUrl: 'http://127.0.0.1:8080',
  model: 'gpt-4o',
  cwd: process.cwd(),
  permissionMode: 'acceptEdits',
})

const unsubscribe = bridge.onNotification(notification => {
  if (notification.event.type === 'agent_end') console.log('done')
})
bridge.onError(err => console.error(err))

await bridge.sendMessage('fix the auth bug')
```

Other notable `AgentCoreBridge` methods: `steer()`, `followUp()`,
`abort()`, `respondToQuestion()`, `getSkills()` / `invokeSkill()`,
`getPrompts()` / `invokePrompt()`, `sendSlash()`, and
`setPermissionMode()` / `getPermissionMode()`.

## Event stream

```typescript
type ProtocolCallback = (notification: AgentProtocolNotification) => void
type ErrorCallback = (err: Error) => void
```

`RuntimeEvent` is a discriminated union keyed on `type`, exported from the
dependency-free `@logician/agent-protocol` package. Subscribe with
`onNotification()` to receive an ordered envelope containing `protocolVersion`,
`sequence`, `timestamp`, and `event`. Event families include:

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
  | 'agent_retry_start' | 'agent_retry_end' | 'agent_error'
  | 'model_select' | 'todos' | 'steered' | 'notice' | 'memory_update'
  | 'subagent_chunk' | 'subagent_lifecycle'
```

Each variant has its own payload shape (see
`packages/agent-protocol/src/events.ts` for the full interfaces).
Most core events pass through `mapAgentEvent()`; bridge-owned features also
emit UI-facing events such as `todos`, `steered`, `notice`, and
`memory_update`. The [headless JSONL stream](/tutorials/headless) is a separate,
smaller versioned contract.

## Configuration

`AgentCoreBridge` takes an `AgentBridgeOptions` object directly (as shown
above); the TUI CLI instead reads `.logician.json` (or the global
`~/.logician/settings.json` fallback) and maps it onto the same options via
`LogicianTuiConfig`. See [Config Schema](/reference/config) for the full
on-disk shape, including `permissionMode`, MCP, memory, and safeguard
options.
