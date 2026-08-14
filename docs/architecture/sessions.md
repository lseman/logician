---
title: Session Persistence
description: How the TUI catalog and append-only agent journal work together.
---

# Session Persistence

Session persistence has two layers because browsing history and recovering an in-flight agent have different requirements.

```mermaid
flowchart LR
    Turn[Completed or interrupted turn] --> Catalog[(TUI SQLite catalog)]
    Turn --> Journal[Append-only agent journal]
    Catalog --> Browser[Session browser and labels]
    Journal --> ActivePath[Parent-linked active path]
    ActivePath --> Resume[Resume, fork, discard, compact]
```

## TUI catalog

Each workspace stores `history.db` under `.logician/tui/sessions/`. The schema tracks sessions, completed turns, labels, and settings changes. SQLite WAL mode makes incremental saves crash-resistant and supports fast session-browser queries.

## Agent journal

The core session journal is append-only. Entries can point to parents, allowing an active path to diverge without truncating or copying earlier history. Branch operations change the selected leaf or append summaries; they do not rewrite the journal.

## Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Active: start or resume
    Active --> Saved: turn completes
    Saved --> Active: next input
    Active --> Branched: fork
    Branched --> Active: merge summary or discard
    Active --> Compacted: context pressure or /compact
    Compacted --> Active: summary becomes provider context
    Active --> [*]: close
```

## Compaction and durability

Compaction changes the context sent to the model, not the historical record shown to the user. Summaries, usage metadata, and the active journal path are persisted so later recovery uses the same logical conversation state.

## File recovery

Conversation branches are distinct from file checkpoints. When checkpointing is enabled, Logician records recoverable file state around edits so rewind can restore both conversational position and affected files without destructive Git operations.
