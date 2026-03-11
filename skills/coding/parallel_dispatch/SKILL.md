---
name: Parallel Dispatch
description: Use when multiple independent reads, searches, or inspections can be fired simultaneously before a sequential write or decision phase. Teaches the fan-out → consolidate → serialize pattern for efficient, safe multi-tool execution.
aliases:
  - fan-out
  - parallel reads
  - batch exploration
  - concurrent tool calls
triggers:
  - read these files in parallel
  - search across these locations at once
  - fan out then consolidate
  - gather context from multiple sources
preferred_tools:
  - rg_search
  - get_file_outline
  - read_file_smart
  - fd_find
example_queries:
  - read all test files related to this module at once
  - search for usages and check config simultaneously
  - gather context from 5 files before editing
when_not_to_use:
  - only one file or source needs to be read
  - each read depends on the result of the previous (sequential dependency)
  - writes are involved (writes must always serialize)
next_skills:
  - coding/edit_block
  - coding/multi_edit
  - coding/quality
  - global/orchestrator
---

## The Core Rule

```
Reads:  PARALLELIZE  — fire all independent reads in one batch
Writes: SERIALIZE    — one file at a time, verify after each
```

Violating the write rule causes race conditions, partial edits, and hard-to-debug state corruption.

## When You Can Parallelize

Reads are parallelizable when:
- They touch different files (no shared state)
- The result of one is NOT needed to formulate another
- All must complete before the next phase begins anyway

Examples of safe parallel batches:
- `rg_search` for symbol A + `get_file_outline` for file B + `read_file_smart` for file C
- Three `rg_search` calls across different modules
- `fetch_url` + `web_search` + `read_file_smart` for background gathering

## Fan-Out → Consolidate → Write Pattern

```
Phase 1: Fan-Out (all parallel)
├── rg_search("class Foo")
├── rg_search("import Foo")
├── get_file_outline("src/core.py")
└── read_file_smart("tests/test_core.py")

Phase 2: Consolidate (sequential)
→ Analyze all results together
→ Decide what needs changing and where

Phase 3: Write (sequential, one at a time)
→ edit_block src/core.py  [verify ✓]
→ edit_block src/models.py  [verify ✓]
→ edit_block tests/test_core.py  [verify ✓]

Phase 4: Verify (parallel)
├── run_ruff
└── run_pytest (targeted)
```

## Implementation Guide

### Formulate the Parallel Batch

Before dispatching, list all needed information:
- What files contain the symbol/pattern I'm editing?
- What do the test files look like?
- Is there config that affects this?
- Is there an existing reference implementation?

Dispatch all of those reads in one turn.

### Consolidate Before Writing

After the batch returns:
1. Review all results together in one pass
2. Identify: which files need editing? in what order?
3. Detect unexpected findings (e.g., symbol has 12 usages, not 3) → adjust plan
4. Note dependencies between edits (file A change must precede file B change)

### Serialize Writes

For each write:
1. Apply the edit
2. Read back the changed region (`read_file_smart` or view the diff)
3. Confirm the edit landed correctly
4. Only then proceed to the next file

### Verify in Parallel

After all writes complete, quality checks can run in parallel:
- `run_ruff` + `run_mypy` + `run_pytest` simultaneously

## How Many Reads is Too Many?

Rule of thumb:
- **≤8 parallel reads**: fine, minimal overhead
- **9–20 parallel reads**: acceptable for large refactors; use `fd_find` first to identify the exact file set before reading
- **>20 parallel reads**: break into two rounds or use `rg_search` with broader pattern to narrow down first

## Anti-Patterns

| Anti-pattern | Correct behavior |
|---|---|
| Writing to multiple files in one parallel batch | Serialize writes — one at a time |
| Reading file B using a result from reading file A in the same batch | If B depends on A, they must be sequential |
| Skipping consolidation and writing immediately after reads | Always analyze the full batch before writing |
| Parallelizing reads after writes have started | Keep phases clean: all reads before any writes |
| Using parallel dispatch for a single-file task | Overkill; use direct read + edit |
