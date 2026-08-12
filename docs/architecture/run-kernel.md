---
title: Run Kernel
description: Durable execution state, replay, budgets, fencing, and tool recovery.
---

# Run Kernel

The Run Kernel is the authoritative execution ledger for a Logician session.
Model context, runtime status, task state, trajectory reports, and operator views
are projections rebuilt from its append-only event stream.

Ledgers live under the workspace:

```text
.logician/run-kernel/<session-id>.jsonl
```

Conversation messages remain in the session transcript. Legacy
`.logician/runtime` and `.logician/trajectories` journals are imported when a
session has no kernel ledger, then moved to
`.logician/migrations/v1-archive/`. Integrated sessions never write the old
formats.

## Guarantees

- Every envelope has a schema version, monotonic sequence, session/task/run
  identities, fencing epoch, timestamp, and validated event payload.
- Replay is deterministic and invalid transitions leave the last valid
  projection unchanged.
- Provider and tool budgets span internal continuations and process restarts.
- Pending steering, follow-up, and next-turn guidance is one ordered queue
  projection and survives restart.
- Intervention escalation is an explicit typed projection, so it survives
  continuations and process restarts without being inferred from telemetry.
- Permission decisions are recorded before execution with their rule, mode,
  user, or fail-closed source and approval scope.
- Subagent lifecycle is projected explicitly. Children inherit the parent's
  permissions, hooks, path sandbox, and budgets.
- A lease takeover advances the fencing epoch; stale owners cannot commit.
- Executed tools follow `intent recorded → result recorded → committed`.
- Tools receive a stable idempotency key through their execution context.
- Unknown external side effects are quarantined rather than retried blindly.
- Recorded-but-uncommitted results are restored into conversation context and
  committed exactly once; intent-only effects receive recovery-specific error
  results and are never replayed implicitly.

Tool authors can declare `recoverySemantics` as `pure`, `idempotent`,
`receipt_recoverable`, or `at_most_once_unknown`. Tools backed by an external
provider may return `recoveryReceipt` in their result for crash reconciliation.

## Operator commands

```sh
logician run replay <session-id>
logician run replay <session-id> --json
logician run doctor <session-id>
logician run doctor <session-id> --json
logician run migrate <session-id>
```

`replay` materializes the current projection. `doctor` reports parse errors,
invariant violations, torn final records, incomplete operations, and the safe
recovery action for each operation. `migrate` explicitly imports and archives
pre-kernel execution journals; `replay` and `doctor` never mutate them.
