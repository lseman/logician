# Logician codebase audit — 2026-09-06

This pass inspected package contracts, cancellation, event delivery/replay,
evaluation execution/reporting, test discovery, CI, and TUI responsiveness. It
ran workspace-wide checks. It is a targeted architectural and reliability review,
not an exhaustive review of every implementation or a benchmark ranking.

## Assessment

The core/runtime/TUI dependency direction is explicit and enforced by tests.
Logician already has bounded event retention, cancellation scopes, durable session
storage, an evaluation package, and incremental TUI rendering. Preserve these
interfaces and strengthen their guarantees before introducing more orchestration
layers. There is currently insufficient measured agent-outcome evidence to call
the system state of the art.

## Changes made

| Area | Failure trigger | Resulting behavior |
| --- | --- | --- |
| Cancellation | Cancel after `run()` but before its queued callback executes | Rechecks cancellation before starting work; removes its abort listener when the run settles |
| Cleanup | Two callers await `close()` concurrently | Both await the same cleanup completion or failure; closed scopes reject new work |
| Core journal and runtime events | A subscriber emits another event during notification delivery | Every subscriber receives notifications in sequence order |
| Replay | A replay callback emits an event | Live delivery is registered before replay, with new events buffered until retained history is delivered |
| Journal diagnostics | An error reporter itself throws | Other subscribers still receive events |
| Eval subprocesses | Long or noisy output; multibyte characters split across writes | Retains only the last 100,000 characters per stream during execution and decodes UTF-8 across chunks |
| Typechecking | Todo test assumes every tool result is a string | Normalizes the existing string-or-structured-result contract |
| Test discovery | Root discovery traverses the entire checkout, including ecosystem trees | Runs tests under application/package source roots, avoiding fixture and external checkout discovery; uses sequential test scheduling because tests share environment/global state |

Also applied safe formatting/import cleanup to changed files, including the
preceding theme work. No dependency upgrades, model changes, external agent
trials, commits, or deployments were performed.

## Validation

- All seven workspace typechecks pass. Initially log-runtime failed on the todo
  test helper's return type.
- Focused cancellation, journal, runtime replay/gap, todo, and eval tests:
  **89 passed, 0 failed** before formatting. The full run below includes them.
- Full scoped test run: **1,139 passed, 23 failed**, 1,162 tests across 148 files,
  about 4.45 seconds. The baseline scoped run had **1,130 passed, the same 23
  failures**. Nine regression tests were added.
- The original root test command was interrupted after producing only its
  startup banner. The scoped command now completes and reports its failures.
- Existing failures include six PTY tests, four persistent-shell tests, one
  bubblewrap test, one ConversationIdentity test, two Legroom SDK tests, and nine
  eval-kernel tests. PTY and bubblewrap logs explicitly report permission errors.
  The identity test assumes writable user-home transcript storage and observes an
  empty path here. SDK/kernel failures need a provisioned integration environment
  and further diagnosis; they are not all proven to be sandbox-only failures.
- Repository lint remains red: **113 errors, 156 warnings, 21 infos**, down from
  127 errors initially. Changed TypeScript/JSON files have **zero lint errors**,
  with 23 warnings and one informational diagnostic remaining. Most remaining
  changed-file warnings concern ANSI regexes and existing unused code.
- `eval:smoke` passes **corpus validation only**. Its single smoke task uses a
  `true` agent command; this is not a measured coding-agent trial.
- Synthetic keystroke benchmark, run directly with Bun because the tsx launcher
  could not create its IPC socket: nine history sizes, 300 keystrokes each,
  120 columns × 40 rows. Worst p95 **1.610 ms**, worst p99 **1.828 ms**. These
  numbers measure this local synthetic render workload, not terminal round trips,
  provider latency, or end-to-end task performance.
- `git diff --check` passes.

## Prioritized next work

| Priority | Evidence | Concrete next step and acceptance criterion |
| --- | --- | --- |
| P1 | `packages/log-eval/src/types.ts` declares `maxTokens` and `maxCostUsd`; `runner.ts` forwards only wall time to process execution | Either enforce these limits with cumulative token/cost telemetry and termination, or reject unsupported limits. Test that a declared limit cannot silently be ignored. Context-window occupancy is not cumulative token usage. |
| P1 | `.github/workflows/ci.yml` invokes failing repository lint and environment-sensitive tests | Fix lint errors in owned batches; provision PTY, sandbox, and SDK/kernel integration jobs explicitly. Keep deterministic tests separate from environment integration tests, with both required where supported. Do not hide failures by skipping them globally. |
| P1 | CI validates `corpus/smoke.json`; the baseline corpus and runner exist, but no outcome trials were run in this review | Establish repeated, pinned-model baseline trials on isolated fixtures. Record environment-graded success, scope violations, false completion, tool errors, token usage, duration, and cost. Add cases for cancellation, recovery, compaction, and permissions. |
| P2 | `packages/log-eval/src/report.ts` reports aggregate pass rate and median duration | Report per-task/per-model repeated-trial results and latency tails, then add cost per successful task when telemetry exists. Use uncertainty estimates before calling a model/harness change an improvement. |
| P2 | Keystroke benchmarks exist but CI does not run them | Add a reproducible performance job with a calibrated regression threshold and retained JSON artifacts. Compare equivalent hardware/runtime/workloads. |
| P2 | Core journal and runtime bus independently implement subscriber delivery | Consider consolidating ordered delivery behind a shared module once protocol correlation and replay-gap contracts are covered. Avoid an interface-only refactor without caller simplification. |

## External evidence informing priorities

[Anthropic's agent evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
distinguishes task outcomes from an agent's completion claim, recommends repeated
trials, and separates capability evaluations from regression coverage. Applied
here, the priority is to exercise Logician's existing evaluation runner against
real outcomes and measure regressions before expanding features.

[OpenAI's harness engineering account](https://openai.com/index/harness-engineering/)
describes making architecture and repository constraints mechanically enforceable.
Logician already has architecture tests; the immediate opportunity is making the
existing quality gates consistently executable and actionable.

These sources inform engineering priorities, not a claim that matching their
patterns proves comparable agent performance.
