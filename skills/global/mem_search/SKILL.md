---
name: mem_search
description: Search and manage cross-session project memory in .logician/memory/. Three-layer workflow (search → timeline → fetch) for token-efficient recall of past decisions, bugs, features, and discoveries.
aliases:
  - memory search
  - recall
  - remember
  - past sessions
  - cross-session memory
  - project memory
triggers:
  - did we solve this before
  - do you remember
  - what did we decide about
  - have we done this before
  - search memory
  - recall from past sessions
  - what was the reason we
  - check memory
  - look up in memory
preferred_tools:
  - mem_search
  - mem_get
  - mem_timeline
  - mem_record
  - mem_list
example_queries:
  - did we already fix this kind of bug before?
  - what did we decide about the auth middleware?
  - search memory for database mocking
  - do you remember what format we use for X?
when_not_to_use:
  - the user is asking about the current session only — use scratch instead
  - the question is about code structure — read the code directly
next_skills:
  - global/think
  - coding/explore
---

## What Project Memory Is

`.logician/memory/` captures knowledge that should survive across sessions.
`MEMORY.md` is a session-indexed table (auto-generated) injected at startup.
Individual observation files in `obs/` hold full content, loaded on demand.

## Observation Types

| Emoji | Type | When to use |
|-------|------|-------------|
| 🔴 | `bugfix`    | Something broken, now fixed |
| 🟣 | `feature`   | New capability added |
| 🔄 | `refactor`  | Code restructured, behaviour unchanged |
| ✅ | `change`    | Config, docs, or misc modification |
| 🔵 | `discovery` | Learned something about the existing system |
| ⚖️ | `decision`  | Architectural or design choice with rationale |

Legacy static facts (user, feedback, project, reference) are also supported.

## Three-Layer Search (Token-Efficient)

**Step 1 — Search (cheap, ~50 t/result):** Find relevant IDs.
```
mem_search("write_file CRLF")
```
Returns a compact table with IDs, timestamps, type emojis, and titles.
Does NOT load full content.

**Step 2 — Timeline (optional, context):** See what was happening around a result.
```
mem_timeline("#042", depth=3)
```
Shows the observations before and after #042. Useful for understanding
the session context without loading full details.

**Step 3 — Fetch (full content):** Load only the IDs you actually need.
```
mem_get(["#042", "#043"])
```
Returns complete observation content. Each is ~200-1000 tokens — be selective.

---

## Recording New Observations

Use `mem_record()` when something important was LEARNED, FIXED, BUILT, or DECIDED.

**Title rule:** Describe WHAT happened — not what *you* did.
- ✓ `"write_file now normalises CRLF line endings"`
- ✗ `"Investigated write_file and found CRLF handling"`

```
mem_record(
    obs_type="bugfix",
    title="write_file now normalises CRLF line endings",
    content="""
**Problem:** Files written with \\r\\n caused diff noise on Linux.
**Fix:** Added _normalize_agent_content() stripping \\r before write.
**How to apply:** No action needed — normalization is automatic.
    """,
  files=["src/tools/core/FileReadTool/tool.py", "src/tools/core/FileEditTool/tool.py"],
)
```

`mem_record()` automatically:
- Assigns the next sequential ID
- Groups by session
- Rebuilds `MEMORY.md` in session-table format

---

## What NOT to Record

- Ephemeral task state or in-progress work
- Things derivable from code, git log, or CLAUDE.md
- Straightforward implementations following obvious patterns
- Incremental progress — record the final result only

---

## Listing All Memory

```
mem_list()
```

Returns observation count, last 10 observations (compact), and any static facts.

## Legacy Static Facts

Pre-existing `TYPE_name.md` files (user/feedback/project/reference) in the memory
root are still supported. They appear under `## Facts` in `MEMORY.md` and are
searched as a fallback when `mem_search` finds no observations.

To add a static fact:
```
write_file(".logician/memory/feedback_prefer_uv.md", """
---
name: feedback_prefer_uv
description: Use uv instead of pip for Python package management
type: feedback
---

Always use `uv` for Python dependency management, not pip directly.
**Why:** User is on Gentoo with system Python; uv avoids conflicts.
**How to apply:** Any time installing packages or creating venvs.
""")
```
Then add it to the Facts section of MEMORY.md.
