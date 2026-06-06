# logician-tui Agent Improvements — Lessons from `pi` and `openclaude`

Scope: improvements to **logician-tui**'s TypeScript agent (`src/agent-core/*`), benchmarked against
`repos/pi` (`packages/agent`, `packages/coding-agent`) and `repos/openclaude` (`src/query`,
`src/services`, `src/coordinator`, `src/memdir`).

Each item: *what they do · what logician-tui has · the gap · concrete action + priority.*
Priorities: **P0** cheap+high-impact · **P1** high-value · **P2** foundation · **P3** longer-term.

---

## What logician-tui already has

Read of `src/agent-core/` before writing this:

- **`AgentLoop`** (`loop.ts`) — ReAct while-loop, `maxIterations` cap, AbortSignal.
- **Contract hooks** (`loop.ts` / `types.ts`) — `beforeToolCall` / `afterToolCall` / `prepareNextTurn` /
  `shouldStopAfterTurn` (just landed).
- **Plugin hooks** (`plugins.ts`) — string events `UserPromptSubmit` / `PreToolUse` / `PostToolUse` /
  `Stop` / `SessionStart` / `SessionEnd`, fire-and-forget, multi-handler via `runHookEvent`.
- **Reactive compaction** (`messages.ts`) — on context-full error only: `compactMessagesForContext`
  summarizes older messages + `compactLargeMessageContent` truncates oversized message bodies.
- **MCP** (`mcp.ts`), **tool registry** (`tools/registry.ts`), **transcript persistence**, **token
  estimation** (`estimateChatPayloadTokens`).

The items below are what logician-tui does **not** yet have.

---

## P0 — Cheap, high-impact

### 1. Guardrails layer — there is none today (openclaude `src/query/*`, pi `tool_call` hook)

**They do:** openclaude runs structural checks every turn; pi exposes a `tool_call` hook with early-exit
`block`. Both stop the common failure modes: identical re-calls, semantically-repeated failures,
unbounded consecutive tool calls, fabricated tool output.

**logician-tui has:** **nothing**. The loop executes whatever the model emits. No duplicate detection, no
failure-loop detection, no "you're spinning" stop.

**Gap:** the loop can silently burn `maxIterations` retrying a broken call. We hit exactly this live with
the `rg_search` arg bug — it would have looped until the iteration cap.

**Action (P0):** add a small `guards/` module + a `Guard` interface returning
`{ pass } | { nudge } | { hardStop }`. Run it in the loop right before tool dispatch. Wire it through the
**already-present** `beforeToolCall` contract hook so it needs no loop surgery. Start with two guards:
- **DuplicateToolGuard** — same `tool + args` signature seen ≥ 3× → hard-stop.
- **ToolFailureLoopGuard** (below, #2).

### 2. Tool-failure loop guard (openclaude `src/query/toolFailureLoopGuard.ts`)

**They do:** track failures by *signature*, *error category*, and *path*. Same failure ≥ N times
(default 3, env-tunable) → trip; inject "stop retrying, change approach."

**logician-tui has:** tool errors are detected only as `result.startsWith("Error:")` for the
`tool_call_end` event — never *counted*. Three different-but-all-failing edits to the same file, or three
distinct `rg_search` patterns that all error, are invisible.

**Gap:** no failure aggregation by category/path; identical-arg matching alone misses the real loop.

**Action (P0):** implement as a guard from #1. Maintain three `Map<string, number>` (signature / error
category / path) over failed results this run; trip when a pending tool call matches any key ≥ threshold.
Categorize via the error text prefix (we don't have structured error payloads in TS — use first ~120
chars). ~80 lines, self-contained, plugs into `beforeToolCall`.

### 3. Token-budget continuation + diminishing-returns stop (openclaude `src/query/tokenBudget.ts`)

**They do:** per-run token tracker. While under 90% of budget, nudge and continue. After ≥ 3
continuations, if the last two token deltas are both < 500, **stop** — the model is emitting tokens but
no longer making progress.

**logician-tui has:** only the fixed `maxIterations` cap. No notion of "still talking, no longer working."

**Gap:** long autonomous runs either stop too early (low cap) or spin to the cap doing nothing.

**Action (P0):** add a `BudgetTracker` and consult it in the **already-present** `shouldStopAfterTurn`
contract hook — zero loop surgery. Track per-turn delta from `estimateChatPayloadTokens`; stop when deltas
collapse across consecutive turns. ~40 lines, directly portable from `tokenBudget.ts`.

### 4. Proactive compaction threshold + cooldown (openclaude `src/services/compact/autoCompact*`)

**They do:** compact *before* hitting the wall, gated by a cooldown (`autoCompactCooldown.test.ts`) so it
doesn't thrash. Two tiers: **micro-compact** (trim individual huge tool results) vs **full** (summarize
history).

**logician-tui has:** `compactMessagesForContext` is solid but **only fires reactively** inside the
context-full `catch` in `loop.ts`. `compactLargeMessageContent` already does micro-trim — but only as part
of the full pass, never standalone.

**Gap:** no proactive trigger (e.g. at 80% of `contextWindowTokens`), no cooldown, no standalone
micro-compact pass.

**Action (P0):** in `prepareNextTurn`, if `estimateChatPayloadTokens` > `0.8 * contextWindowTokens`, run a
**micro-compact-only** pass first (cheap: just `compactLargeMessageContent` over the tail); fall back to
full `compactMessagesForContext` only if still over. Add a per-N-turn cooldown. Reuses existing helpers.

---

## P1 — High value

### 5. AgentHarness: orchestration layer above the loop (pi `packages/agent/docs/agent-harness.md`)

**They do:** `AgentHarness` owns an explicit **phase** (`idle | turn | compaction | branch_summary |
retry`), **turn snapshots** (frozen per-turn state so config changes never mutate an in-flight request),
and **save points** (between turns: flush persisted writes, then re-read model/tools/system-prompt/stream
options for the *next* turn within the same run). Structural ops rejected while busy.

**logician-tui has:** `AgentLoop.run` mutates `this._messages` in place; `config` read live mid-turn.
No phase, no snapshot, no mid-run config refresh.

**Gap:** can't safely change model/tools/system-prompt mid-run; no deterministic persistence ordering.
This is the foundation for #6 and #8.

**Action (P1):** wrap `AgentLoop` in a thin `AgentHarness` that adds a `phase` field + `createTurnState()`
snapshot consumed per iteration + a save-point between turns. The contract hooks already give the
injection points; this formalizes the state model around them.

### 6. Steering / follow-up / nextTurn queues (pi harness)

**They do:** `steer(text)` injects guidance into a *running* turn; `followUp(text)` queues for after the
current turn; `nextTurn(text)` queues before the next user prompt. Abort clears steer/follow-up but
**preserves** `nextTurn`.

**logician-tui has:** input enters only at the top of `run(userMessage)`. No way to nudge a running turn —
and this is a TUI, where live steering is the whole point.

**Gap:** no live interactivity during long autonomous runs.

**Action (P1):** once #5's save points exist, add the three queues drained at save points; wire the TUI
input bar to `steer` / `followUp`. High UX payoff.

### 7. Typed, result-producing hook bus (pi `packages/agent/docs/hooks.md`)

**They do:** one hook system where the **event type carries its own result type**. `on(type, handler)`;
result-producing events use typed reducers — `context` (transform messages), `tool_call` (early-exit
block), `tool_result` (sequential patch accumulation), `before_provider_payload` (ordered transform),
`before_agent_start` (inject messages + chain system prompt). `observe()` = read-only firehose.

**logician-tui has:** two disjoint systems — string-event plugin hooks (multi-handler but fire-and-forget,
no typed results) and the new contract hooks (typed results but **single** handler per event). Neither
composes multiple result-producing handlers.

**Gap:** can't stack multiple extensions on the same intercepting event with deterministic reducers.

**Action (P1/P2):** unify both layers into one typed bus with per-event reducer semantics (transform /
patch-accumulate / first-cancel / early-block) and source-metadata scopes. The contract hooks become the
base reducers; plugin hooks become observers. pi's `hooks.md` is the blueprint.

---

## P2 — Foundation

### 8. Stop-hooks that can resurrect the turn (openclaude `src/query/stopHooks.ts`)

**They do:** on stop, hooks can *continue* the agent (task not actually done), trigger memory extraction,
and inject a summary/continuation message. Stop is a decision point, not just an exit.

**logician-tui has:** the `Stop` plugin event is fire-and-forget; `shouldStopAfterTurn` can force-stop but
nothing can *veto* a stop and continue.

**Gap:** no "are we actually done?" gate.

**Action (P2):** let `shouldStopAfterTurn` consult a stop-hook chain whose handlers may return a
continuation message instead of stopping. Combine with a verification check (lint/test ran after edits).

### 9. Coordinator / worker-agent mode (openclaude `src/coordinator/coordinatorMode.ts`)

**They do:** a coordinator agent restricted to a tool subset (spawn / delegate / send-message) that
orchestrates worker sub-agents instead of doing work itself. Per-role tool allow-lists.

**logician-tui has:** no sub-agent spawning, no role-restricted tool surface.

**Gap:** no planner/worker separation for larger tasks.

**Action (P2):** add a coordinator config that swaps in a restricted tool set + orchestration system
prompt, plus a worker-spawn tool. Pairs with the tool registry's existing per-run tool selection.

---

## P3 — Longer-term

### 10. Durable / resumable runs (pi `packages/agent/docs/durable-harness.md`)

Persist queues, pending writes, turns, provider requests, and tool calls as durable entries so an
interrupted run resumes from a boundary. Provider streams aren't resumable; unfinished tool calls only
retried if declared idempotent. logician-tui persists a transcript but cannot resume a run. Revisit after
#5. High effort.

### 11. Memory-dir relevance scoring (openclaude `src/memdir/`)

Age-decay relevance scoring (`memoryAge.ts`, `findRelevantMemories.ts`) + auto-extract memories at stop.
logician-tui has no in-agent memory recall layer. Pairs with stop-hooks (#8).

### 12. Session tree / branching (pi `navigateTree` + branch summary)

Session-as-tree with fork + branch-summary generation; durable leaf cursor. logician-tui is linear.
Experimentation workflow, P3.

---

## Suggested order

1. **P0** (#1 guards scaffold, #2 failure-loop guard, #3 budget stop, #4 proactive compaction) — all four
   ride the **existing** contract hooks (`beforeToolCall` / `shouldStopAfterTurn` / `prepareNextTurn`), so
   **no loop surgery** and days of effort.
2. **#5 AgentHarness** (phase + snapshot + save points) — unlocks #6, #8.
3. **#6 steering** + **#7 hook-bus unification**.
4. **P3** (durable, memory, tree) as research follow-ups.

## File-pointer index

| Topic | pi | openclaude | logician-tui today |
|---|---|---|---|
| Orchestration layer | `packages/agent/docs/agent-harness.md` | — | `src/agent-core/loop.ts` (no harness) |
| Typed hooks | `packages/agent/docs/hooks.md` | `src/hooks/` | `plugins.ts` + contract hooks (split) |
| Guardrails | `tool_call` hook | `src/query/toolFailureLoopGuard.ts` | **none** |
| Token budget / stop | — | `src/query/tokenBudget.ts` | `maxIterations` only |
| Stop hooks | — | `src/query/stopHooks.ts` | `Stop` event (fire-and-forget) |
| Compaction | (harness `compact`) | `src/services/compact/` | `messages.ts` (reactive only) |
| Steering | harness `steer`/`followUp` | — | **none** |
| Coordinator | harness sub-agents | `src/coordinator/coordinatorMode.ts` | **none** |
| Memory recall | — | `src/memdir/` | **none** |
| Durable resume | `durable-harness.md` | — | transcript only |
