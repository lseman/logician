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
tsx packages/tui/src/index.ts
```

### Headless (programmatic)

The headless entry point is `AgentCoreBridge` from `@logician/coding-agent`'s
`application` export — the same bridge the TUI itself drives. It's
event-driven: subscribe with `on()`/`onError()`, then call `sendMessage()`.

```typescript
import { AgentCoreBridge } from '@logician/coding-agent/application'

const bridge = new AgentCoreBridge({
  baseUrl: 'http://127.0.0.1:8080',
  model: 'gpt-4o',
  cwd: process.cwd(),
  permissionMode: 'acceptEdits',
})

const unsubscribe = bridge.on(event => {
  if (event.type === 'agent_end') console.log('done')
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
type EventCallback = (event: ParsedBridgeEvent) => void
type ErrorCallback = (err: Error) => void
```

`ParsedBridgeEvent` is a discriminated union keyed on `type`, exported from
`@logician/coding-agent` (re-exported from `./runtime/events.ts`). The real
`type` values are:

```typescript
type BridgeEventType =
  | 'agent_start' | 'agent_end' | 'agent_settled'
  | 'agent_retry_start' | 'agent_retry_end' | 'agent_error'
  | 'turn_start' | 'turn_end'
  | 'token' | 'thinking_token'
  | 'text_start' | 'text_end'
  | 'message_start' | 'message_update' | 'message_end'
  | 'queue_update'
  | 'tool_start' | 'tool_end'
  | 'tool_execution_start' | 'tool_execution_update' | 'tool_execution_end'
  | 'repair_nudge'
  | 'phase' | 'context_update' | 'compaction' | 'memory_update'
  | 'question_request' | 'permission_request'
  | 'session_delete'
  | 'model_select'
  | 'todos' | 'steered' | 'save_point' | 'notice'
  | 'subagent_chunk' | 'subagent_lifecycle'
```

Each variant has its own payload shape (see
`packages/coding-agent/src/runtime/events.ts` for the full interfaces).
Most mirror the core agent-loop's internal `AgentEventBody` union 1:1 via
`mapAgentEvent()`; a handful (`todos`, `steered`, `save_point`, `notice`,
`memory_update`) are synthesized directly by `AgentCoreBridge` for UI-only
signals that don't exist as core agent events.

## Configuration

`AgentCoreBridge` takes an `AgentBridgeOptions` object directly (as shown
above); the TUI CLI instead reads `.logician.json` (or the global
`~/.logician/settings.json` fallback) and maps it onto the same options via
`LogicianTuiConfig`. See [Config Schema](/reference/config) for the full
on-disk shape, including `permissionMode`, MCP, memory, and safeguard
options.
