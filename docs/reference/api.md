---
title: API Reference
description: Public API for the Logician agent.
---

# API Reference

Programmatic access to Logician's agent capabilities.

## Entry points

### TUI (interactive)

```typescript
import { run } from '@logician/tui'

await run({
  config: {
    llm: {
      url: 'http://127.0.0.1:8080',
      model: 'gpt-4o',
    },
    permissions: 'ask',
  },
})
```

### Headless (programmatic)

```typescript
import { createAgent } from '@logician/coding-agent'

const agent = createAgent({
  llm: {
    url: 'http://127.0.0.1:8080',
    model: 'gpt-4o',
  },
  permissions: 'acceptEdits',
})

const result = await agent.execute('fix the auth bug')
console.log(result.output)
console.log(result.toolCalls)
console.log(result.sessionId)
```

## Agent interface

```typescript
interface Agent {
  execute(instruction: string): Promise<AgentResult>
  stream(instruction: string): AsyncIterable<StreamEvent>
  getSession(id: string): Promise<Session>
  listSessions(): Promise<Session[]>
  createBookmark(label: string): Promise<Bookmark>
  rewindTo(checkpointId: string): Promise<void>
  compact(): Promise<void>
  close(): Promise<void>
}

interface AgentResult {
  output: string
  toolCalls: ToolCall[]
  sessionId: string
  duration: number
  tokensUsed: number
}

interface StreamEvent {
  type: 'thinking' | 'tool_call' | 'tool_result' | 'response'
  content: string
  metadata?: Record<string, unknown>
}
```

## Configuration interface

```typescript
interface Config {
  llm: {
    url: string
    model: string
    apiKey?: string
    maxTokens?: number
    temperature?: number
  }
  permissions: 'plan' | 'ask' | 'acceptEdits' | 'acceptAll'
  thinkingLevel: 'low' | 'medium' | 'high' | 'full'
  reasoning?: {
    mode: string
    maxIterations?: number
  }
  compaction?: {
    enabled: boolean
    triggerFraction?: number
  }
  mcp?: {
    servers: Record<string, McpServerConfig>
  }
  plugins?: string[]
}
```
