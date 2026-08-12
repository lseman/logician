---
title: Session Management
description: Persistence, bookmarks, branching, rewind, and compaction.
---

# Session Management

Logician persists every session in SQLite, with support for bookmarks, branching, and compaction.

Agent execution state is persisted separately in the workspace Run Kernel
ledger. See [Run Kernel](/architecture/run-kernel) for replay and recovery
commands.

## Session structure

The application session catalog is stored in SQLite. Harness transcripts use
an append-only, parent-linked entry tree with an explicit active leaf. Forking
does not copy or truncate messages: discarding checks out the parent leaf and
merging appends a branch-summary entry. The selected path survives restart.

## Key operations

### Bookmarking

Create a named bookmark at any point:

```
Ctrl+B → "add checkpoint" → "review auth flow"
```

Bookmark the current state for later return.

### Branching

Start a new branch from any point:

```
/bookmark create review-auth
/branch from-review-auth
```

The branch inherits the conversation up to the bookmark, then diverges.

### Rewinding

Return to any checkpoint or bookmark:

```
Ctrl+R → select checkpoint
```

The active leaf moves to the selected point, and the agent continues on a new
path. Abandoned entries remain recoverable in the append-only journal.

### Compaction

When context grows large, the agent can compact the session:

```json
{
  "compaction": {
    "enabled": true,
    "triggerFraction": 0.75,
    "strategy": "summarize"
  }
}
```

Compaction summarizes older messages while preserving tool results and final answers.

## Session commands

| Command | Action |
|---|---|
| `/session list` | List all sessions |
| `/session switch <id>` | Switch to a session |
| `/session new` | Start a new session |
| `/session bookmark <name>` | Create a bookmark |
| `/session rewind <id>` | Rewind to a checkpoint |
| `/session branch <name>` | Branch from current point |
| `/session compact` | Compact current session |
| `/session export` | Export session as JSON |
