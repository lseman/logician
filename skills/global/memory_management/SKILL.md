---
name: Memory Management
description: Use for explicit context window management — summarizing accumulated history, dropping irrelevant context, creating checkpoints, and keeping the conversation focused when the session is long or context pressure is rising.
aliases:
  - context checkpoint
  - summarize session
  - compact history
  - drop context
  - context budget
triggers:
  - context getting long
  - summarize what we've done so far
  - create a checkpoint
  - forget the earlier context
  - we're running low on context
preferred_tools:
  - scratch_write
  - scratch_read
  - todo_write
example_queries:
  - summarize what we've accomplished so far
  - create a checkpoint before the next phase
  - what's the current state of this session
when_not_to_use:
  - early in a conversation (< 5 turns)
  - context pressure is low
next_skills:
  - global/think
  - global/orchestrator
---

## Why Explicit Memory Management Matters

Long sessions accumulate:
- Old file contents that have since changed
- Reasoning traces from resolved problems
- Intermediate outputs that are no longer relevant
- Exploratory reads that didn't lead anywhere

Without active management, the agent makes decisions based on stale context, wastes tokens re-reading known information, and eventually hits context limit without warning.

## Context Pressure Signals

Start memory management when you see:
- Session is ≥ 15 turns deep
- The same file has been read 3+ times in one session
- You are unsure what decisions were made earlier
- You feel compelled to re-read earlier context to answer a question
- The `auto_compact` threshold is approaching (see `agent_config.json`: `auto_compact_threshold`)

## Workflow

### 1. Create a Session Checkpoint

When starting a new phase of work (after completing a major feature or before a risky change):

```
scratch_write("
=== SESSION CHECKPOINT [2026-03-11] ===

Completed:
- Added parallel_dispatch skill
- Fixed CLAUDE.md (removed stale cli/ references)
- Expanded think + orchestrator SKILL.md

Current state:
- Working in: skills/coding/, skills/global/
- Last file edited: skills/coding/parallel_dispatch/SKILL.md
- All edits verified: ruff passing, no tests broken

Open threads:
- academic/SKILL.md not yet added
- memory_management SKILL.md in progress

Key decisions made:
- Not converting 10_superpowers to different format — already correct layout
- ralph SKILL.md already complete; SOUL.md reference is fine
")
```

### 2. Summarize Before Long Read-Phases

Before a large fan-out (reading 10+ files), write a brief state summary so you can recover context after:

```
scratch_write("
Before read phase:
  Goal: find all usages of class ForeForest
  Files likely affected: src/eoh/, skills/lazy_timeseries/
  Will ignore: tests/, rust-cli/
")
```

After the read phase, update the scratch with key findings.

### 3. Explicitly Drop Irrelevant History

When context contains large blocks of irrelevant information:
- State explicitly: "Dropping earlier X context — no longer relevant"
- Do NOT repeat it in your next response
- Rely on the checkpoint scratch instead

### 4. Recover from Auto-Compact

If `auto_compact` fires (summary injected into history):
1. Read the compact summary carefully before proceeding
2. Update scratch with any gaps the compact summary doesn't cover
3. Re-read only the files you actively need for the next edit — do not re-read everything

### 5. Use TODO for Multi-Session State

For tasks spanning multiple sessions, use `todo_write` to persist state across compaction events:

```
todo_write([
  {"task": "Expand firecrawl SKILL.md", "status": "done"},
  {"task": "Add academic/SKILL.md", "status": "pending"},
  {"task": "Run full quality pass", "status": "pending"}
])
```

`todo_read` at session start to recover where you left off.

## Context Budget Rules (from SOUL.md)

- Target: stay under 40–60% of available context window
- When context pressure rises:
  - Summarize previous turns in 1–2 sentences at start of response
  - Prefer targeted re-reads over relying on faded earlier context
  - Never dump entire large files unless asked

## Anti-Patterns

| Anti-pattern | Correct behavior |
|---|---|
| Re-reading the same large file twice in one session | Cache results in scratch; re-read only if file may have changed |
| Carrying full file content across many turns | Summarize what's relevant; discard the rest |
| Waiting for auto_compact to manage context | Be proactive; checkpoint before pressure builds |
| Not tracking what edits have been made | Use scratch or todo as an audit trail |
| Summarizing so aggressively that key decisions are lost | Keep facts, decisions, and open threads — drop ephemera only |
