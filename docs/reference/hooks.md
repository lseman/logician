---
title: Hook API
description: Programmatic API for registering and composing hooks on the HookBus.
---

# Hook API

Programmatic access to the hook system. For what each hook does and when it
fires, see [Hook System](/architecture/hooks). This page covers the
`HookBus` registration API itself.

## AgentHooks

Each hook point is an optional handler on the `AgentHooks` interface,
imported from `@logician/log-core`:

```typescript
import type { AgentHooks } from '@logician/log-core'

const hooks: AgentHooks = {
  beforeToolCall(ctx, signal) {
    console.log(`Calling ${ctx.toolCall.name}`)
  },
  afterToolCall(ctx, signal) {
    if (ctx.isError) console.error(`${ctx.toolCall.name} failed: ${ctx.result}`)
  },
}
```

Handlers may be synchronous or return a `Promise`, and receive an optional
`AbortSignal` as the second argument. See [Hook System](/architecture/hooks)
for the full list of hook points and their context/result shapes.

## Registering hooks

Hooks are registered on a `HookBus` instance, not via a global function.
`HookBus.register()` takes a whole `AgentHooks` object and wires up every
handler it defines in one call, returning a single unsubscribe function:

```typescript
import { HookBus } from '@logician/log-core/hooks/native'

const bus = new HookBus({ errorMode: 'continue' })

const unregister = bus.register(hooks, {
  id: 'my-plugin',       // stable identity for diagnostics/dedup
  source: 'my-plugin',   // used to attribute errors to a source
  priority: 0,            // higher runs first; ties keep registration order
  timeoutMs: 5000,        // per-handler timeout override
})

// Later, to remove all handlers registered above:
unregister()
```

Individual hook points can also be registered one at a time with `bus.on()`:

```typescript
bus.on('beforeToolCall', hooks.beforeToolCall!, { priority: 10 })
```

## Composition semantics

Multiple registrants can hook the same event. Each event type composes
handlers deterministically rather than just "last one wins":

- `beforeToolCall` — early-block: the first handler to return `{ content }`
  short-circuits tool execution; a returned `{ args }` rewrites arguments
  for later handlers.
- `afterToolCall` — patch-accumulate: each handler sees the prior patch;
  later non-`undefined` fields win.
- `prepareNextTurn` — transform: messages thread through every handler in
  order.
- `shouldStopAfterTurn` — first `true` wins.

## Priority and error isolation

- **Priority**: handlers with a higher `priority` run first; equal
  priorities preserve registration order.
- **Timeouts**: `HookBusOptions.defaultTimeoutMs` sets a default per-handler
  timeout (0 disables it); a per-registration `timeoutMs` overrides it. A
  timed-out handler is treated like a thrown error — skipped and reported.
- **Error mode**: `HookBusOptions.errorMode` controls whether a thrown
  handler aborts the rest of the chain (`"throw"`) or is skipped and
  reported via `onError` (`"continue"`, the default).

## Observing without hooking

`bus.observe(observer)` subscribes a read-only firehose over every event —
useful for logging or metrics without participating in the hook chain's
return-value semantics:

```typescript
bus.observe((event, ctx) => {
  console.log(`[hook] ${event} fired`)
})
```
