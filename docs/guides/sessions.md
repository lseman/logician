---
title: Session Management
description: Persistence, bookmarks, branching, rewind, and compaction.
---

# Session Management

Logician persists every session in SQLite, with support for bookmarks, branching, and compaction.

## Session structure

```
sessions/
├── default/
│   ├── history.db          # SQLite session database
│   ├── checkpoint-001.json # Rewind point
│   ├── checkpoint-002.json
│   └── bookmark-main.json  # Named bookmark
```

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

The session history is truncated to the selected point, and the agent continues from there.

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
