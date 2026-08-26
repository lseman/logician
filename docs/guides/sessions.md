---
title: Session Management
description: Persistence, labels, branches, rewind, and compaction.
---

# Session Management

Logician stores each conversation as one append-only, parent-linked session
journal. The TUI session browser and the agent harness use the same canonical
record rather than maintaining separate catalog and recovery databases.

By default, sessions live under `.logician/sessions/sessions/` in the directory where
Logician was launched:

```text
.logician/sessions/sessions/<session-id>/
├── meta.json
└── messages.jsonl
```

`meta.json` contains the session name, workspace, activity time, selected leaf,
and format version. `messages.jsonl` contains messages, completed TUI turns,
settings changes, labels, branch summaries, and compaction entries. The files
are local, inspectable, and require no database server.

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

The journal records incremental execution state, not only the final transcript.
After an interruption, Logician rebuilds the selected parent-linked path and
restores its conversation, settings, branch state, and completed TUI turns.
`meta.json` identifies the active leaf, so selecting a branch does not rewrite or
duplicate earlier history.

Session history currently uses JSONL rather than SQLite. Session listing scans
the per-session metadata files, while previews are derived from the first user
message. A future search index may use SQLite as a rebuildable acceleration
layer, but it would not replace the append-only journal as the canonical record.
