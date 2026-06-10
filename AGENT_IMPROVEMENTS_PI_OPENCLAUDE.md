# Agent Improvements — Lessons from `pi` and `openclaude`

Source repos studied: `repos/pi` (TypeScript, `packages/agent` + `packages/coding-agent`) and
`repos/openclaude` (TypeScript, `src/query`, `src/services`, `src/coordinator`, `src/memdir`).

This doc proposes improvements to **our** agent (`src/agent/*` Python core + `tui/src/agent-core`
TS loop). Each item lists: what they do, what we have today, the gap, and a concrete action with priority.

Priorities: **P0** cheap+high-impact · **P1** high-value · **P2** foundation · **P3** longer-term.

---

## Status (2026-06-10)

Most of the P0–P2 items below **have since landed in the TS core** (`tui/src/agent-core/`).
This doc is kept as a design reference; per-item status is flagged inline (✅ done / 🟡 partial /
⬜ not started). The TS implementation is now the source of truth — the Python notes describe the
older `src/agent/*` core.

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Tool-failure loop guard | ✅ | `guards.ts` (signature + path + category buckets) |
| 2 | Diminishing-returns budget stop | ✅ | `budget.ts` + `shouldStopAfterTurn` |
| 3 | Compact tool descriptions | ✅ | `skills.ts` catalog + on-demand `read-skill` |
| 4 | AgentHarness (phase / save points) | ✅ | `harness.ts` phase state machine |
| 5 | Steering / follow-up / nextTurn queues | ✅ | `harness.ts` queues + drain hooks |
| 6 | Two-tier compaction (micro + full) | ✅ | `messages.ts` `compactToFit` |
| 7 | Coordinator / worker mode | ⬜ | — |
| 8 | Typed result-producing hook bus | ✅ | `hook-bus.ts` reducers |
| 9 | Stop-hook veto / completion gate | 🟡 | `shouldStopAfterTurn` exists; no veto-and-resume chain |
| 10 | Durable / resumable harness | ⬜ | — |
| 11 | Memory-dir relevance scoring | ⬜ | — |
| 12 | Session tree / branching | 🟡 | `harness.ts` fork/branchSummary (single-level) |

## Already adopted / in flight

- **Richer loop contract** (`beforeToolCall` / `afterToolCall` / `prepareNextTurn` / `shouldStopAfterTurn`)
  — landed in `tui/src/agent-core/loop.ts`, composed through the typed `HookBus`.
- **String-event plugin hooks** (`UserPromptSubmit`/`PreToolUse`/`PostToolUse`/`Stop`) — already live.
- **Token-budget trim** (`AgentLoop._trim_to_budget`), **tool-call repair**, **guardrails**
  (duplicate/consecutive/hallucination/read-before-edit/verification) — already live in Python core.

The items below were the original gap analysis; remaining open work is #7, #9 (finish), #10, #11.

---

## P0 — Cheap, high-impact

### 1. Tool-failure loop guard (openclaude `src/query/toolFailureLoopGuard.ts`)

**They do:** track tool failures by *signature*, *error category*, and *path*. If the same failure
repeats ≥ N times (default 3, env `CLAUDE_CODE_TOOL_FAILURE_LOOP_THRESHOLD`), trip a guard that injects a
message telling the model to stop retrying the same broken call and change approach.

**We have:** `DuplicateToolGuard` / `ConsecutiveToolGuard` catch *identical* calls, but not *semantically
repeated failures* (e.g. editing the same file with three different-but-failing patches, or three
distinct rg patterns that all error). We just rediscovered this class of bug live — the `rg_search`
arg bug would have looped silently.

**Gap:** no failure-category aggregation, no path-bucketing, no "you keep failing on X, try something else."

**Action (P0):** add `ToolFailureLoopGuard` to `src/agent/guardrails.py`. Keep three maps
(signature/category/path → count), categorize errors (parse `error.error` payload we already produce in
`_tool_error_payload`), trip on threshold, emit a nudge. Mirrors the existing `Guard` protocol — small,
self-contained.

### 2. Token-budget continuation / diminishing-returns stop (openclaude `src/query/tokenBudget.ts`)

**They do:** per-turn budget tracker. While `turnTokens < 0.9 * budget`, inject a continuation nudge and
keep going. Detect *diminishing returns*: after ≥ 3 continuations, if the last two token deltas are both
< 500, stop — the agent is spinning without making progress.

**We have:** `_trim_to_budget` (drops oldest messages) but **no diminishing-returns stop**. Our loop runs
to `max_iterations` regardless of whether progress stalled.

**Gap:** no signal "the model is still emitting tokens but no longer doing useful work — stop now."

**Action (P0):** add a `BudgetTracker` to `AgentLoop.run`: record per-turn token delta, stop early when
deltas collapse below a floor across consecutive turns. ~40 lines, directly portable. Wire into the new
`shouldStopAfterTurn` contract on the TS side too.

### 3. Compact tool descriptions (pi `coding-agent`, already noted in `PI_ARCHITECTURE_COMPARISON.md` #4)

**They do:** one-line tool descriptions in the request; full schema only when the tool is selected.

**We have:** full descriptions every turn.

**Action (P0):** confirmed ~20% token savings in the comparison doc. Low effort, already scoped — finish it.

---

## P1 — High value

### 4. AgentHarness: a real orchestration layer above the loop (pi `packages/agent/docs/agent-harness.md`)

**They do:** `AgentHarness` sits above the low-level loop and owns: explicit **phase**
(`idle | turn | compaction | branch_summary | retry`), **turn snapshots** (immutable per-turn state so
config changes never mutate an in-flight request), **save points** (flush persisted writes + refresh
model/thinking/tools/resources between turns *within the same run*), **steering / follow-up / nextTurn**
queues drained at safe points, and **durable pending-write ordering**.

**We have:** a single `AgentLoop.run` that mutates `messages` in place; config is read live mid-turn.
No phase model, no turn snapshot, no mid-run config refresh, no steering queue.

**Gap (the big one):** we cannot safely change model/tools/system-prompt mid-run, cannot steer a running
turn, and have no deterministic persistence ordering. This blocks half of the features below.

**Action (P1, multi-step):**
1. Introduce an explicit `phase` enum + "structural ops rejected while busy" rule.
2. Add `create_turn_state()` → frozen snapshot consumed by the turn; setters mutate *next* snapshot only.
3. Add a save-point hook between turns that re-reads config. This is the foundation for #5, #7, #9.

### 5. Steering / follow-up / nextTurn queues (pi harness)

**They do:** `steer(text)` injects guidance into a *running* turn; `followUp(text)` queues a message for
after the current turn; `nextTurn(text)` queues a message inserted *before* the next user prompt. Abort
clears steering/follow-up but **preserves** `nextTurn`.

**We have:** nothing — user input only enters at the top of `run`. No way to nudge a long autonomous run.

**Gap:** no live interactivity during autonomous/Ralph-style execution.

**Action (P1):** once #4's save points exist, add the three queues drained at save points. Wire TUI input
to `steer`/`followUp`. High UX payoff for long runs.

### 6. Two-tier compaction: micro-compact + full compact (openclaude `src/services/compact/`)

**They do:** **micro-compact** (`microCompact.ts`, `apiMicrocompact.ts`) trims individual large
tool-results in place without summarizing the whole conversation — cheap, frequent. **Full compact**
(`compact.ts`, `autoCompact.ts`) summarizes the whole history with a cooldown
(`autoCompactCooldown.test.ts`) so it doesn't thrash. Plus `postCompactCleanup`, `compactWarningHook`.

**We have:** `_trim_to_budget` = drop-oldest only. No summarization, no per-tool-result micro-trim, no
cooldown. Dropping oldest loses information the summary would keep.

**Gap:** lossy trimming vs. lossy-but-informed summarization; no targeted big-result trimming.

**Action (P1):**
- Micro-compact first: when a single tool result is huge, replace its body with a short summary +
  "(elided, N chars)" — cheap, keeps structure. Wire into the new `afterToolCall` contract.
- Then full auto-compact with a cooldown to avoid thrash. Reuse our existing summarizer/reasoner stack.

### 7. Coordinator / worker-agent mode (openclaude `src/coordinator/coordinatorMode.ts`)

**They do:** a coordinator agent restricted to a tool subset (spawn/delegate/send-message/synthetic-output)
that orchestrates worker sub-agents instead of doing the work itself. Tool allow-lists per role.

**We have:** `TaskTool` exists for sub-agents, but no first-class coordinator *mode* with a restricted
tool surface and orchestration prompt.

**Gap:** no clean separation between "planner that delegates" and "worker that executes."

**Action (P1/P2):** add a coordinator config that swaps in a restricted tool set + orchestration system
prompt. Pairs naturally with our existing `TaskTool` and skill routing.

---

## P2 — Foundation

### 8. Typed, result-producing hook bus (pi `packages/agent/docs/hooks.md`)

**They do:** one hook system where the **event type carries its own result type** (phantom symbol). One
`on(type, handler)` API; whether a handler may return a result is determined by the event. Result-producing
events use typed reducers: `context` (transform messages), `tool_call` (early-exit block), `tool_result`
(sequential patch accumulation), `before_provider_payload` (ordered transform), `before_agent_start`
(inject messages + chain system prompt). `observe()` sees everything read-only.

**We have:** two disjoint systems — string-event plugin hooks (fire-and-forget) and the new in-process
contract callbacks (single-handler). No multi-handler chaining, no typed results, no provenance/source
metadata, no cleanup registry.

**Gap:** can't compose multiple extensions on the same event with deterministic reducer semantics.

**Action (P2):** unify our two hook layers into one typed bus with reducer semantics per event. The pi
design (reducers: transform / patch-accumulate / first-cancel / early-block) is the blueprint. Add
source-metadata scopes (`createScope({ sourceInfo })`) and `errorMode: continue | throw`.

### 9. Stop-hooks with task-completion / idle semantics (openclaude `src/query/stopHooks.ts`)

**They do:** on stop, run hooks that can *continue* the agent (task not actually done), trigger memory
extraction, task-completed and teammate-idle hooks, and inject summary messages. Stop is a decision point,
not just an exit.

**We have:** `Stop` is a fire-and-forget notification; it cannot resurrect the turn.

**Gap:** no "are we actually done?" gate — pairs with our existing `VerificationGuard`.

**Action (P2):** make `shouldStopAfterTurn` consult a stop-hook chain that can veto the stop and inject a
continuation message. Hook our `VerificationGuard` into it.

---

## P3 — Longer-term

### 10. Durable / resumable harness (pi `packages/agent/docs/durable-harness.md`)

**They do:** design (spike-stage) for persisting queues, pending writes, operations, turns, provider
requests, and tool calls as durable session entries so an interrupted run can resume from a boundary.
Notes: provider streams are *not* resumable; unfinished tool calls are unsafe to retry unless declared
idempotent.

**We have:** SQLite history persistence but no operation/turn-level durability or resume-from-boundary.

**Action (P3):** revisit after #4 lands. Requires tools to declare idempotency. High effort, real payoff
for long autonomous runs that crash.

### 11. Memory-directory relevance scoring (openclaude `src/memdir/`)

**They do:** structured memory dir (`findRelevantMemories.ts`, `memoryAge.ts`, `memoryScan.ts`, typed
memory kinds in `memoryTypes.ts`, team memory paths) with age-decay relevance scoring and auto-extraction
of memories at stop time.

**We have:** `src/memory/` + the file-based `MEMORY.md` index, but no age-decay relevance scoring or
auto-extract-at-stop.

**Action (P3):** add age-weighted relevance scoring to memory recall; auto-extract memories in the stop
hook (#9). Aligns with our existing memory frontmatter scheme.

### 12. Session tree / branching (pi `navigateTree` + branch summary)

**They do:** session is a tree; `navigateTree` + branch-summary generation lets users fork a conversation
and explore alternatives. Leaf cursor persisted durably (`setLeafId` appends a `leaf` entry).

**We have:** linear history only.

**Action (P3):** experimentation workflow. Already flagged P3 in `PI_ARCHITECTURE_COMPARISON.md` #6.

---

## Suggested order

1. **P0 quick wins** (#1 loop guard, #2 diminishing-returns stop, #3 compact descriptions) — days, no
   architectural change.
2. **#4 AgentHarness phase + turn snapshot + save points** — unlocks #5, #6, #7, #9.
3. **#6 two-tier compaction** + **#5 steering** on top of #4.
4. **#8 typed hook bus** to unify the hook story before more extensions pile on.
5. **P3** (durable harness, memory scoring, tree) as research-grade follow-ups.

## File-pointer index

| Topic | pi | openclaude |
|---|---|---|
| Orchestration layer | `packages/agent/docs/agent-harness.md`, `src/agent.ts`, `src/agent-loop.ts` | — |
| Typed hooks | `packages/agent/docs/hooks.md` | `src/hooks/`, `src/utils/hooks.ts` |
| Observability | `packages/agent/docs/observability.md` | `src/services/analytics/` |
| Durable/resumable | `packages/agent/docs/durable-harness.md` | — |
| Tool-failure loop guard | — | `src/query/toolFailureLoopGuard.ts` |
| Token budget / continuation | — | `src/query/tokenBudget.ts` |
| Stop hooks | — | `src/query/stopHooks.ts` |
| Compaction (micro + full) | (harness `compact`) | `src/services/compact/` |
| Coordinator mode | (harness sub-agents) | `src/coordinator/coordinatorMode.ts` |
| Memory dir | — | `src/memdir/` |
