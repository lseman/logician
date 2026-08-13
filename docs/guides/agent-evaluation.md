---
title: Agent evaluation
description: Run independently graded Logician coding-task trials.
---

# Agent evaluation

The `@logician/agent-eval` workspace defines versioned task corpora and grades
the resulting repository state independently of the agent's completion claim.

Validate the bundled smoke corpus:

```sh
bun run eval:smoke
```

Run the three-task baseline three times each in automatically provisioned,
disposable workspaces:

```sh
bun run tui/packages/agent-eval/src/cli.ts run \
  tui/packages/agent-eval/corpus/baseline.json \
  --trials 3 \
  --work-root outputs/agent-eval-workspaces \
  --output outputs/agent-eval-baseline.json
```

The runner appends the task prompt to the configured agent command. Task
manifests should pin a repository and revision, impose wall-time/token/cost
limits, and use deterministic command, file-content, file-absence, or diff-scope
graders. Run tasks only in disposable checkouts because the evaluated agent is
expected to modify its workspace.

Fixture revisions may be Git commits or `sha256:` content digests. The runner
verifies the revision before invoking the agent, initializes each copied fixture
as a clean Git repository, enforces process-tree deadlines, and writes the
headless JSONL stream to a sibling `.artifacts` directory.

Reports deliberately separate `agentDeclaredComplete` from
`environmentGradedPass`. Quality gates must use the latter.

## First baseline

The 2026-08-13 local `default-model` baseline passed 9/9 independently graded
trials across bug-fix, feature, and refactor tasks. Median end-to-end duration
was 17.6 seconds; every trial declared completion, changed exactly one allowed
source file, and passed its executable tests. The durable report is
`outputs/agent-eval-baseline-2026-08-13.json`, with a compact Markdown view and
nine JSONL trajectory artifacts beside it.

This small corpus validates the evaluation plumbing; it is not evidence of
general coding-agent quality. Expand it with longer, private tasks and hidden
tests before using the score as a release claim.
