---
title: Hook API
description: Programmatic API for registering and managing hooks.
---

# Hook API

Programmatic access to the hook system.

## Hook interface

```typescript
interface HookPlugin {
  name: string
  hooks: {
    beforeLLMRequest?: (ctx: BeforeLLMRequestCtx) => void | Promise<void>
    afterLLMResponse?: (ctx: AfterLLMResponseCtx) => void | Promise<void>
    beforeToolCall?: (ctx: BeforeToolCallCtx) => void | Promise<void>
    afterToolCall?: (ctx: AfterToolCallCtx) => void | Promise<void>
    beforeSessionSave?: (ctx: BeforeSessionSaveCtx) => void | Promise<void>
    afterSessionSave?: (ctx: AfterSessionSaveCtx) => void | Promise<void>
    onError?: (ctx: ErrorCtx) => void | Promise<void>
  }
}
```

## Hook contexts

### BeforeLLMRequestCtx

```typescript
interface BeforeLLMRequestCtx {
  messages: Message[]
  tools: Tool[]
  config: Config
}
```

### AfterLLMResponseCtx

```typescript
interface AfterLLMResponseCtx {
  response: LLMResponse
  toolCalls: ToolCall[]
}
```

### BeforeToolCallCtx

```typescript
interface BeforeToolCallCtx {
  toolName: string
  args: Record<string, unknown>
}
```

### AfterToolCallCtx

```typescript
interface AfterToolCallCtx {
  toolName: string
  result: unknown
  error?: Error
}
```

### BeforeSessionSaveCtx

```typescript
interface BeforeSessionSaveCtx {
  sessionId: string
  messages: Message[]
}
```

### AfterSessionSaveCtx

```typescript
interface AfterSessionSaveCtx {
  sessionId: string
}
```

### ErrorCtx

```typescript
interface ErrorCtx {
  error: Error
  context: Record<string, unknown>
}
```

## Registering hooks

```typescript
import { registerHook } from '@logician/agent-core/hooks'

registerHook({
  name: 'my-plugin',
  hooks: {
    beforeToolCall({ toolName, args }) {
      console.log(`Calling ${toolName}`)
    },
  },
})
```

## Hook execution order

Hooks execute in registration order. No priority system.
