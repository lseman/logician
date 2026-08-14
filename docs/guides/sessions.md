---
title: Session Management
description: Persistence, labels, branches, rewind, and compaction.
---

# Session Management

Logician maintains two complementary records:

- the TUI session catalog in `<workspace>/.logician/tui/sessions/history.db`, used for browsing, names, labels, and completed turns;
- an append-only agent journal used for crash recovery, branches, settings changes, and compaction state.

## Everyday operations

| Command | Action |
|---|---|
| `/new` | Start a new session |
| `/session` | Open the session manager |
| `/sessions [clean]` | List sessions or clean stale entries |
| `/name <name>` | Set a short session name |
| `/rename <name>` | Rename the current session |
| `/bookmark <label> [note]` | Label the current position |
| `/bookmarks` | List labels in the current session |
| `/compact` | Summarize older context |
| `/fork` | Create a branch from the active journal leaf |
| `/discard-branch` | Return to the parent branch |
| `/branch-summary` | Summarize the current branch |
| `/rewind` | Open rewind/checkpoint selection |

Sessions are saved automatically. `/save` requests an explicit save but is not required after each turn.

## Branches and rewind

The agent journal is parent-linked and append-only. Forking creates a new active path without copying history. Discarding a branch selects its parent; merging records a branch summary rather than rewriting old entries. File checkpoints are separate from conversation branches and let the UI restore edited files when available.

## Compaction

Compaction replaces older provider context with a summary while retaining the durable transcript. Configure its token reserve and recent-context window:

```json
{
  "compaction": {
    "enabled": true,
    "reserveTokens": 16384,
    "keepRecentTokens": 20000
  }
}
```

Use `/context` to inspect current context state and `/compact` when you want to compact before the automatic threshold.

## Recovery

The SQLite catalog uses WAL mode and saves completed turns. The agent journal records incremental execution state. After an interruption, Logician can restore the selected session path and recover queued/session metadata without treating the displayed transcript as the only source of truth.
