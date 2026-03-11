---
name: Think
description: Use for deliberate reasoning, decomposition, hypothesis testing, and staged planning before taking action on ambiguous, multi-step, or high-risk tasks.
aliases:
  - reason
  - plan
  - deliberate
  - decompose
  - think step by step
triggers:
  - think through this
  - reason about this before acting
  - what's the best approach here
  - decompose this task
  - I'm not sure how to approach
preferred_tools:
  - scratch_write
  - scratch_read
example_queries:
  - think through the architecture before implementing
  - reason about the failure modes here
  - decompose this migration into safe steps
when_not_to_use:
  - the task is clearly scoped with no ambiguity
  - the user has explicitly given a step-by-step plan to execute
next_skills:
  - orchestrator
  - coding/explore
  - coding/edit_block
---

## When to Use This Skill

Activate `think` before any tool calls when:

1. **The request is ambiguous** — multiple valid interpretations exist
2. **The task is multi-step and order matters** — wrong step order could corrupt state
3. **Risk is non-trivial** — file deletions, migrations, refactors touching many files
4. **You've hit an unexpected state** — stuck, looping, or observing inconsistent outputs
5. **Design decisions need comparison** — two or more reasonable approaches

Do NOT use for well-scoped single-file edits, clear bug fixes with known location, or simple informational answers.

## Workflow

### Step 1: Classify and Frame

Write out (in scratch or inline):
- What is the user actually asking for? (restate in your own words)
- What is the success condition?
- What could go wrong?
- What information do I NOT have yet?

### Step 2: Decompose

Break the task into an ordered list of sub-tasks. For each:
- What tool(s) will satisfy it?
- Does it depend on any previous step's output?
- Is it reversible? If not, flag it.

### Step 3: Identify the Critical Path

- Which steps must be sequential? (data dependency, file state)
- Which steps can be parallelized? (independent reads, independent file edits)
- Where is the highest risk of failure?

### Step 4: State Assumptions Explicitly

Before acting, list:
- **Fact**: information confirmed from the codebase / tools
- **Inference**: reasonable conclusion from what you've seen
- **Assumption**: things you're taking for granted that could be wrong

### Step 5: Execute with Checkpoints

After completing each major step:
- Observe the actual output
- Compare against expected
- If divergence → re-enter think before continuing

## Anti-Patterns to Avoid

| Anti-pattern | Correct behavior |
|---|---|
| Acting immediately on an ambiguous request | Frame and clarify first |
| Assuming file state without reading | Read before writing |
| Treating "I know what this does" as a fact | Label it as Inference |
| Planning all steps upfront then ignoring observations | Checkpoint after each major action |
| Re-running the same failing tool call without rethinking | Use `think` to re-analyze before retry |
| Skipping decomposition because task "seems simple" | Decompose all execution tasks |

## Scratch Integration

Use `scratch_write` to dump your reasoning:

```
scratch_write("
Task: Rename helper across 3 files
Success: All call sites updated, tests pass, no broken imports

Decomposition:
1. [explore] Find all usages with rg_search → parallel OK
2. [multi_edit] Apply rename in all 3 files → sequential (shared state)
3. [quality] Run ruff + pytest → verify correctness

Assumptions:
- Inference: helper is not exported from __init__.py (need to verify)
- Fact: helper is only used internally (confirmed by step 1)

Risk: step 2 edits multiple files; if one fails, others may be inconsistent → verify after each file
")
```

Then read it back before tool calls to keep reasoning grounded.

## Output Format When Reasoning Aloud

When the user asks you to "think" or "reason" before acting, structure your visible output as:

```
**Restatement**: [what you understand]
**Goal**: [success condition]
**Plan**:
  1. [step + tool]
  2. [step + tool]
  3. ...
**Key risks**: [what can go wrong]
**Starting with**: [first concrete action]
```

This makes your reasoning auditable and lets the user catch misunderstandings before you act.
