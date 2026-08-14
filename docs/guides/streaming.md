---
title: Streaming
description: Understand interactive runtime events and the headless JSONL contract.
---

# Streaming

Logician renders provider text and tool progress as events arrive. Thinking output appears only when the selected provider and thinking settings supply it.

```mermaid
sequenceDiagram
    participant User
    participant TUI
    participant Agent
    participant Provider
    participant Tool
    User->>TUI: Submit or steer
    TUI->>Agent: Queue input
    Agent->>Provider: Streaming request
    Provider-->>Agent: Text, thinking, or tool call deltas
    Agent-->>TUI: Runtime events
    Agent->>Tool: Execute completed tool call
    Tool-->>Agent: Result
    Agent-->>TUI: Tool progress and result
    Agent->>Provider: Continue with tool result
    Agent-->>TUI: Settled
```

## Interactive event families

The bridge exposes turn, message, token, tool-execution, context, queue, permission, retry, compaction, and subagent lifecycle events. The TUI projects those events into transcript text, tool cards, status widgets, and overlays.

Text streaming and thinking display are not an audit log of hidden model reasoning. They are the content the provider exposes through its API.

## Control the display

- `Ctrl+O` expands or collapses tool details.
- `Alt+J`/`Alt+K` moves between tool cards.
- `Alt+Enter` toggles the focused card.
- `Ctrl+Shift+T` cycles thinking display mode.
- `Ctrl+Enter` sends or flushes steering during an active turn.

## Headless stream

```bash
logician exec --jsonl "analyze src/auth.ts"
```

Headless records use the versioned `logician.exec-stream` schema. This is intentionally smaller than the internal bridge event union. See the [Headless tutorial](/tutorials/headless) for records, exit status, and CI guidance.
