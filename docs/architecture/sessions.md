---
title: Session Persistence
description: How sessions are stored, managed, and recovered.
---

# Session Persistence

Logician stores all session data in SQLite, providing fast queries and reliable recovery.

## Database schema

```mermaid
erDiagram
    SESSIONS {
        string id PK
        string title
        datetime created_at
        datetime updated_at
        int message_count
        string status
    }
    MESSAGES {
        int id PK
        string session_id FK
        string role
        string content
        datetime timestamp
        string tool_name
        boolean is_compacted
    }
    BOOKMARKS {
        int id PK
        string session_id FK
        string name
        int message_id
        datetime created_at
    }
    CHECKPOINTS {
        int id PK
        string session_id FK
        int message_id
        string label
        datetime created_at
    }

    SESSIONS ||--o{ MESSAGES : contains
    SESSIONS ||--o{ BOOKMARKS : has
    SESSIONS ||--o{ CHECKPOINTS : has
```

## Recovery

Sessions survive crashes and restarts:

1. **Auto-save** — messages are saved after each LLM response
2. **Checkpoint** — manual save points for rewind
3. **Bookmark** — named save points with labels
4. **Compaction** — automatic when context nears limit

## Compaction

When the context window approaches its limit:

```json
{
  "compaction": {
    "enabled": true,
    "triggerFraction": 0.75,
    "strategy": "summarize",
    "keepToolResults": true,
    "keepFinalAnswers": true
  }
}
```

Compaction preserves:
- Tool results and their outcomes
- Final answers to user questions
- Critical decisions and conclusions
- Error messages and their context

## Session lifecycle

```mermaid
stateDiagram-v2
    [*] --> Active
    Active --> Saving: After each response
    Saving --> Active
    Active --> Compacting: Context > 75%
    Compacting --> Active: Compacted
    Active --> Archived: User closes
    Archived --> [*]
```
