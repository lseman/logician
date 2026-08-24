---
title: Durability & Recovery
description: Continuation tracking, the thread ledger, file checkpoints, and run recovery.
---

# Durability & Recovery

Logician does not keep a single monolithic execution ledger. Durability is
split across a few focused, purpose-built mechanisms inside `log-core`,
each owning one kind of state.

::: tip Renamed from "Run Kernel"
Earlier versions of Logician centered durability on a single append-only
event ledger called the Run Kernel. It was replaced by a continuation-based
harness: less machinery, with durability distributed to the component that
actually needs it (thread state, file state, run status).
:::

## Continuation tracking

The harness turn loop (`runtime/harness/agent-harness.ts`) drives each
turn through intake, model call, tool execution, and hooks. In-memory runtime
status — phase (`idle` / `turn` / `compaction` / `branch_summary`),
streaming state, pending tool calls, retry attempts, and the last run
outcome — is projected by `runtime/state/runtime-state.ts` from the harness's
own event stream. This projection drives UI-facing status; it is
intentionally ephemeral and rebuilds from the next turn if lost.

Run-scoped policy (budgets, stop conditions, acceptance checks) is owned by
`control/policy/run-controller.ts`, `run-budget.ts`, and
`execution-policy.ts`; the pure vocabulary those files consume
(`RunBudgetLimits`, `RunOutcomeStatus`) lives under `system/types/` so
other layers can reference it without depending on the enforcement classes.
Task status shown to the harness comes from an optional, capability-supplied
`TaskLedger` (`system/types/task-ledger.ts`) — a read-only snapshot, not a
durable store itself.

## Thread ledger

`capabilities/session/thread-ledger.ts` is the append-only record of
conversation changes. Rather than mutating history in place, rewinds,
branch merges, and compactions append a `projection` item with the
replacement messages and a typed reason (`restore`, `rewind`,
`branch-discard`, `branch-merge`, `compaction`, `clear`, `run-commit`). The
current message projection is derived from the entry list, so provenance
for how the transcript reached its current shape is preserved rather than
erased.

## File checkpoints

`capabilities/session/file-checkpoints.ts` snapshots files before the
agent's own tools touch them, so `/rewind` restores the workspace alongside
the conversation:

- One frame is opened per prompt and popped on rewind; within a frame, only
  the **first** write to a path records its pre-state — that's what the
  frame restores to.
- `write_file` / `edit_file` calls capture the path's content before the
  write.
- `bash` calls snapshot the working tree to a git tree object (via a
  temporary index; the real index and worktree are untouched) before and
  after the command. The diff identifies which paths the command mutated,
  and their pre-call contents come from the "before" tree. This requires a
  git repository; gitignored and >1MB files are not captured.
- Restores stay per-file — a rewind never reverts edits made outside the
  agent's own tools.

Git failures during capture are silent; the corresponding bash mutation is
then simply not captured for that frame.

## Compaction checkpoints

When a transcript has been compacted repeatedly and keeps re-growing,
`runtime/compaction/run-checkpoint.ts` resets it to a small, structured
handoff instead of compacting again: the original objective, recent tool
evidence, the last assistant message, and current task state, rewritten as
a single "continue from here" message. This bounds context growth on long
autonomous runs without relying on the model to re-derive its own history.

## Recovery model

Recovery responsibilities are split by what changed:

- **Conversation state** recovers by replaying the thread ledger's
  projection — a fresh `AgentRuntimeState` is rebuilt from the next event.
- **Workspace state** recovers via `/rewind`, using the file-checkpoint
  frame stack.
- **Context budget pressure** recovers via compaction, escalating to a run
  checkpoint reset if compaction alone isn't keeping pace.

There is deliberately no single global "replay this session from byte zero"
operation. Each mechanism is scoped to the state it owns and recovers
independently.
