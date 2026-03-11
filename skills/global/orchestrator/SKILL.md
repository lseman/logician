---
name: Orchestrator
description: Use for coordinating multi-step workflows across several tools or skills, managing shared state, and ensuring coherent sequential or parallel execution.
aliases:
  - coordinate
  - multi-step workflow
  - pipeline
  - sequence skills
  - fan-out
triggers:
  - do these steps in order
  - coordinate across multiple files
  - run this pipeline
  - sequence these operations
  - fan out then consolidate
preferred_tools:
  - scratch_write
  - scratch_read
  - todo_write
  - todo_read
example_queries:
  - explore, then edit, then verify — coordinate the whole flow
  - fan out to read 5 files, then consolidate findings
  - run migrate, then test, then commit in order
when_not_to_use:
  - single-skill single-step task
  - the user has already given a complete ordered plan
next_skills:
  - coding/quality
  - coding/git
  - global/think
---

## Role of the Orchestrator

The orchestrator manages **execution order, shared state, and error recovery** across multiple skills. It does not do any coding itself — it sequences and monitors.

Use when a task requires **three or more distinct skill invocations** where:
- The output of one step feeds the next
- Some steps can be parallelized and some cannot
- Failure in one step should halt dependent steps

## Execution Model

### Read-Fan-Out → Consolidate → Write-Serialize

The most common pattern:

```
Phase 1 (parallel): Read all inputs
  ├── rg_search (find usages)
  ├── get_file_outline (understand structure)
  └── fetch_url (get external reference)

Phase 2 (sequential): Consolidate findings → form action plan

Phase 3 (sequential): Write / mutate (one file at a time)
  ├── edit_block file_A
  ├── edit_block file_B
  └── edit_block file_C

Phase 4 (parallel): Verify
  ├── run_ruff
  ├── run_mypy
  └── run_pytest (targeted)
```

**Rule: reads can parallelize; writes must serialize.**

### When to Parallelize

Parallelize when:
- Tasks have no shared output state (reading different files)
- Tasks are independent queries (search + fetch + outline)
- All must complete before the next step begins anyway

Do NOT parallelize when:
- One result feeds into another
- Writing to the same file or overlapping regions
- Order affects correctness (migration before seed, schema before UI)

## Workflow

### 1. Plan (use scratch)

Write the execution DAG before starting:

```
scratch_write("
Orchestration plan:
Phase 1 [parallel]:
  - explore: find all usages of foo()
  - explore: get outline of config.py

Phase 2 [sequential — depends on phase 1]:
  - think: decide which files need editing

Phase 3 [sequential]:
  - edit_block: update foo() in utils.py
  - edit_block: update call sites in core.py
  - edit_block: update tests in test_utils.py

Phase 4 [parallel]:
  - quality: ruff check
  - quality: pytest test/test_utils.py

Phase 5 [sequential — if phase 4 green]:
  - git: checkpoint commit
")
```

### 2. Execute with Status Tracking

Use `todo_write` for longer orchestrations:

```
todo_write([
  {"task": "Find all usages (rg_search)", "status": "pending"},
  {"task": "Get config.py outline", "status": "pending"},
  {"task": "Edit utils.py", "status": "pending"},
  {"task": "Edit core.py", "status": "pending"},
  {"task": "ruff + pytest verify", "status": "pending"},
  {"task": "Git checkpoint", "status": "pending"},
])
```

Mark items in-progress and done as you go.

### 3. Handle Failures

If a step fails:
- **Read failure** → retry once, then proceed with partial data (note the gap)
- **Write failure** → STOP. Do not continue to dependent writes. Report the blocker.
- **Verify failure** → STOP. Do not commit. Fix before proceeding to git.
- **Skill unavailable** → fall back to core tools (`shell`, `file_ops`)

### 4. Deliverables

After orchestration completes:
- Report final state: which steps succeeded, which were skipped
- Surface any open risks or partial completions
- If `git` step was reached: confirm checkpoint created

## Anti-Patterns

| Anti-pattern | Correct behavior |
|---|---|
| Writing to file B before confirming file A edit succeeded | Always verify each write before proceeding |
| Running all reads AND writes in one parallel batch | Reads parallel, writes sequential |
| Continuing after a write failure | STOP and report |
| Committing without a quality pass | Quality gate is mandatory before git |
| Losing track of which steps completed | Use scratch or todo to track state |
