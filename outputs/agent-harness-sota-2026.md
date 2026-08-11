# Logician Agent Harness: State-of-the-Art Review

**Date:** 2026-08-11  
**Scope:** continuation reliability, context management, tools, subagents, security, observability, and evaluation.

## Executive conclusion

Logician already has a comparatively sophisticated harness: structured task state, bounded in-run continuation nudges, acceptance verification, reflection, loop detection, retries, proactive and emergency compaction, steering queues, checkpoints, subagents, and structured intervention events. Its main opportunity is not adding more prompt-driven behavior. It is making durable state, progress, continuation, compaction, and termination deterministic.

The highest-risk gap is cross-run continuation. A single agent run has iteration and nudge limits, but queued `nextTurn` messages can start a fresh run whose counters reset. An extension can therefore create an effectively unbounded chain. The current bridge also turns continuation into a fabricated user message (`"continue"`), weakening task identity and provenance.

The recommended direction is:

> Persistent truth lives in an append-only event log and typed task checkpoint. Model context is a disposable projection rebuilt from that truth.

This one principle improves continuation, compaction, crash recovery, auditability, subagent handoffs, and evaluation.

## What current agents teach us

### Pi

[Pi](https://github.com/earendil-works/pi) has a clean separation between its provider API, agent core, coding-agent application, and telemetry. Its most transferable features are an append-only session tree; resume, fork, and branch navigation; structured incremental compaction; recent raw-tail retention; preservation of tool-call/result boundaries; and cumulative read/modified-file tracking. Its extension lifecycle also covers turns, tools, compaction, session replacement, and shutdown.

Logician already incorporates several Pi-style compaction ideas. The remaining improvement is to pin typed task and verification state outside the generated summary and make compaction an atomic event commit.

Sources: [Pi sessions](https://pi.dev/docs/latest/sessions), [session format](https://pi.dev/docs/latest/session-format), [compaction](https://pi.dev/docs/latest/compaction), [extensions](https://pi.dev/docs/latest/extensions).

### Whale

[Whale](https://github.com/usewhale/Whale) is the likely project meant by “Whale.” Its notable ideas are cache-conscious prompt layout, DeepSeek-native long-context support, and programmable JavaScript workflows for fan-out research, pipelines, synthesis, and adversarial review. Its approximately 98% cache-hit figure is a project claim, not an independently verified benchmark.

The lesson for Logician is to make the stable prompt prefix measurable and enforce a typed workflow-result contract. Programmable orchestration is useful only when child context, concurrency, budget, depth, and partial failure are bounded.

### OpenHands

[OpenHands](https://docs.openhands.dev/sdk/arch/agent) uses a stateless, event-driven step model: events are durable, while each agent step is atomic and interruptible. Its condenser is a replaceable component rather than being fused to the main loop. Its default stuck detector recognizes repeated action/observation cycles, repeated errors, monologues, alternating patterns, and repeated context failures using semantic event comparisons.

Logician’s detector is already strong on exact repetition, degeneration, and stagnation. It should add alternating-pattern detection and, more importantly, measure semantic progress rather than treating any successful tool call as progress.

Source: [OpenHands stuck detector](https://docs.openhands.dev/sdk/guides/agent-stuck-detector).

### Codex

[Codex](https://github.com/openai/codex) provides durable threads, typed item events correlated by thread/turn/item identifiers, explicit turn interruption and steering, thread forking, sandbox profiles, and approval flows. Its execution policy separates sandboxing from approval and supports argument-sensitive rules. Its app-server protocol is a useful model for keeping UI deltas separate from durable events.

Recent Codex compaction-loop reports are also instructive: compaction needs a retry cap, a generation/version check, and atomic commit semantics. A summary that fails to reduce context must not repeatedly replace the active projection.

Sources: [Codex app server](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md), [Python SDK sandbox and turns](https://github.com/openai/codex/blob/main/sdk/python/docs/api-reference.md), [compaction-loop failure report](https://github.com/openai/codex/issues/13946).

### OpenCode

[OpenCode](https://github.com/opencode-ai/opencode) distinguishes primary agents, subagents, and hidden system agents for compaction and summaries. It offers per-agent models, step limits, and wildcard permission policies. Its session design keeps original events durable and commits only completed compaction results.

Logician should copy the atomic compaction rule: an interrupted condenser leaves the previous context projection valid. It should also snapshot the effective permission policy for each turn.

Sources: [OpenCode agents](https://opencode.ai/docs/agents), [session specification](https://github.com/anomalyco/opencode/blob/dev/specs/v2/session.md).

### SWE-agent and mini-SWE-agent

[SWE-agent](https://github.com/SWE-agent/SWE-agent) separates durable trajectories from model-facing history processors. Exact run configuration, thoughts, actions, observations, and results are retained for replay and evaluation. It classifies bounded re-queries for malformed or blocked actions.

[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) is an important counterweight to feature accumulation: keep the loop kernel small and make policies, telemetry, retries, and projection replaceable. Every additional harness mechanism should prove value through ablation.

### Aider, Cline, Goose, and Gemini CLI

[Aider](https://github.com/Aider-AI/aider) demonstrates the value of a token-budgeted repository map, automatic commits/undo, and deterministic lint/test feedback after edits.

[Cline](https://github.com/cline/cline) uses mutation checkpoints and can independently restore conversation or workspace state. This is safer than conflating transcript rewind with filesystem rewind.

[Goose](https://github.com/aaif-goose/goose) exposes explicit context-limit strategies, separate main/subagent turn ceilings, concurrency limits, and reusable recipes. These controls should be attached to a durable run budget rather than scattered configuration values.

[Gemini CLI](https://github.com/google-gemini/gemini-cli/blob/main/docs/core/subagents.md) gives subagents isolated context, restricted tools, explicit turn and time budgets, and recursion prevention. Its checkpointing snapshots project and conversation state before AI file modifications.

## Logician audit

### Existing strengths

- `agent-loop-runner.ts` bounds iterations, continuation nudges, reflection, and acceptance retries.
- `task-state-controller.ts` records objective, phase, evidence, changed files, verification, blockers, and tool failures.
- `loop-detector.ts` detects duplicates, repeated failures, exact repeats, degenerate tool sequences, and stagnation.
- `session.ts` maintains an append-only JSONL journal and checkpoints.
- `compaction.ts` respects turn boundaries, keeps a recent tail, tracks usage, and supports structured summaries.
- `intervention-controller.ts` emits structured recovery and escalation events.
- Steering, follow-up, and next-turn queues are first-class.
- The recurring `LoopManager` prevents overlapping executions and uses a generation token to reject stale scheduling.

### Ranked gaps

1. **Unbounded cross-run continuation.** `nextTurn` starts a new run after settlement, resetting iteration and nudge counters. Add a durable continuation lease spanning all internal runs.
2. **Continuation masquerades as user input.** Replace literal `"continue"` prompts with a native continuation/resume envelope carrying cause, cursor, and budget.
3. **Progress is too permissive.** Any successful tool work can reset recovery state. Require semantic progress: a new evidence hash, state/diff change, completed plan edge, or fresh passing verification.
4. **Completion relies partly on English prose heuristics.** Make typed outcomes authoritative. Use prose detection only as a low-confidence fallback.
5. **Some undeclared stop paths become completed.** Default uncertain termination to `unknown` or `blocked`; promote to completed only with evidence.
6. **Intervention state is fragmented.** Use one harness-owned, durable intervention controller across built-ins, runs, compactions, and resumes.
7. **Continuation policies compete implicitly.** Centralize queue, loop recovery, stop policies, acceptance, and reflection in one typed continuation arbiter.
8. **A single iteration counter prices unequal work equally.** Use hierarchical budgets for model calls, tools, verifier retries, tokens, cost, wall time, and reserved finalization.
9. **Compaction does not pin enough typed state.** Persist objective, constraints, plan, changed/read files, failures, verification ledger, queues, intervention history, and remaining budgets outside summary prose.
10. **Recurring loops retry forever at fixed cadence.** Add exponential backoff, jitter, a failure circuit breaker, deadlines, durable schedule state, and idempotency keys.
11. **Verification execution needs stronger control.** Bound concurrency, use typed executable/arguments where possible, retain output digests/artifacts, and propagate cancellation.
12. **No trajectory-level harness gate.** Unit tests are necessary but cannot measure premature stopping, redundant continuation, compaction fidelity, or real-task success.

## Recommended continuation algorithm

Use a durable run state machine:

`ORIENT → PLAN → ACT → VERIFY → REPAIR → DONE | NEEDS_INPUT | BLOCKED | FAILED`

Before every model call:

1. Materialize a context projection from the durable task checkpoint, recent raw events, retrieved repository context, effective permissions, and remaining budget.
2. Execute one tool batch transactionally and append `started`, `completed`, `failed`, or `cancelled` events.
3. Compute semantic progress against the prior checkpoint.
4. Classify failures: retry transient transport/rate-limit errors with backoff; replan deterministic tool/test failures; stop after repeated equivalent failures.
5. At candidate completion, run deterministic acceptance checks, then an independent semantic verifier when needed.
6. If verification fails, append typed feedback and continue within the reserved repair budget.
7. At the context watermark, create a new projection and commit it only if it is smaller, valid, and based on the current event generation.
8. Return `DONE` only when acceptance evidence is green and there are no pending tools, approvals, processes, queues, or unresolved failures.

## Implementation roadmap

### P0: continuation reliability

Build a `ContinuationController` owned by the harness. It should persist:

- run/session/continuation IDs and cause;
- state-machine phase;
- global model/tool/token/cost/time budgets;
- reserved verification and repair budgets;
- last semantic-progress fingerprint;
- consecutive no-progress and equivalent-failure counts;
- terminal status and recovery hint.

Replace bridge `sendMessage("continue")` with a first-class `continueRun(envelope)` path. Ensure rejected continuation promises become durable failure outcomes.

### P0: structured completion and progress

Make `task_status` or an equivalent typed result mandatory for autonomous work. Add an evidence-based completion gate inspecting acceptance criteria, current diff, verification freshness, pending processes, approvals, queues, and unresolved failures.

### P0: atomic compaction checkpoint

Version the event stream. A condenser reads generation `N`; its result commits only if generation remains `N`, otherwise it retries from the new head. Retain the old projection until the new summary and typed checkpoint are durably committed. Cap attempts and reject non-shrinking results.

### P1: trajectory evaluation

Add replayable NDJSON trajectories containing exact model/config/tool versions and correlated IDs. Create fault-injection fixtures for tool timeout, malformed calls, rate limits, context overflow, process death, compaction interruption, subagent failure, and restart/resume.

Track:

- task/acceptance pass rate;
- premature-stop and unnecessary-continuation rates;
- repeated-call and loop-escape rates;
- resume and compaction-fidelity success;
- regression count;
- tokens, cache hit, cost, latency, and tool failures;
- approval burden and sandbox violations.

### P1: policy and subagent hardening

Add argument/path/network-sensitive policy rules, mutation idempotency IDs, child-agent lineage, depth 1 by default, per-child turn/time/token budgets, typed result packets, and partial-failure synthesis.

### P2: repository intelligence and UX

Add a change-refreshed, token-budgeted symbol/import map. Show run phase, remaining budget, retry/repair state, compaction generation, and active subagent count in the status bar.

## Best first engineering slice

Implement the continuation controller before expanding orchestration or memory. It directly closes the largest correctness hole and creates the foundation for semantic progress, durable recovery, budgeting, observability, and evals.

Acceptance criteria for that slice:

- Internal continuation never creates a user-authored transcript entry.
- A continuation chain has durable turn, token, time, and no-progress limits across runs.
- Every continuation records its cause and parent run.
- Semantic progress, not generic tool success, resets the no-progress counter.
- Errors in fire-and-forget continuation are persisted and surfaced.
- Restart resumes from the same continuation lease and remaining budget.
- Tests cover a malicious extension that continuously enqueues `nextTurn`, repeated successful no-op reads, a real edit/test recovery, and crash/resume.

## Anti-patterns to avoid

- Prompt-only TODO or progress state.
- Counting a final sentence as completion.
- Resetting loop guards after any successful tool call.
- Retrying the same deterministic failure verbatim.
- Summaries that omit files, tests, failed approaches, or pending work.
- Compaction that replaces history before its result is validated and committed.
- Security enforcement only through optional hooks.
- Raw child-agent transcripts dumped into the parent context.
- Harness features shipped without trajectory ablation on fixed tasks and models.

