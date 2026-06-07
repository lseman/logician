# Agent Core Audit — Dedup / Unify / SOTA

Scope: `tui/src/agent-core/` (loop, harness, hook-bus, builtin-hooks, backend,
messages, types, plugins). Findings ranked by payoff/risk. Each has a concrete
fix. Nothing here is applied yet — this is the plan.

---

> **Status (2026-06-07):** ALL findings applied — A1-A3, B1-B5, C1-C4, D3-D4,
> and the steering-storage unification (section E). D1/D2 intentionally left as
> noted (single-producer alias table; stringly error convention is a wide
> change). Typecheck: the 3 `OpenAIBackend` errors are closed; the remaining 14
> errors are pre-existing UI-layer issues in untouched files (plugins.ts,
> slash-commands.ts, tui-core.ts, tui.ts, utils.ts). eslint clean on all
> touched files (the `generate` max-params warning is gone with C1).
>
> **Summary of what changed:**
> - B1/B2: 6 `run<Hook>` wrappers stripped of try/catch; `composeHooks` takes an
>   `onError` and the HookBus is the sole error owner (wired to the loop's error
>   event).
> - B3/C3: one HookBus composition. Harness supplies queue drains via
>   `config.internalHooks` (composed built-ins → harness-queues → user); the
>   second wrapping bus and the `addSteering`/`addFollowUp` monkey-patch (plus
>   the dead loop-level nextTurn queue) are gone.
> - B4/B5: `compactMessages` + `splitForCompaction` shared skeleton in
>   messages.ts; harness.compact passes the LLM summarizer, builtins/loop pass
>   none. `COMPACTION_TARGET_FRACTION` (0.65) replaces scattered literals.
> - C1: `backend.generate(messages, options)` with a `callbacks` bag.
> - C2: `convertToLlm` moved to messages.ts. C4: `estimateMessageTokens`
>   delegates to `estimateChatPayloadTokens` (one estimate basis).
> - D3: dropped `_sleep` unused param. D4: inlined `batchTerminate`.

## A. Broken / dead code (fix first — these are bugs)

### A1. `cycleModel` is broken — `OpenAIBackend` not imported + private fields ✅ DONE
**File:** `loop.ts:1064-1088`
`cycleModel` does `new OpenAIBackend({...})` and reads `this.backend.baseUrl`,
but:
- `OpenAIBackend` is **not imported** in `loop.ts` → `TS2304` (3×, current
  typecheck failures).
- `OpenAIBackend.baseUrl` / `.model` are `private` (`backend.ts:32-33`) →
  cross-instance access wouldn't compile even with the import.

The loop hard-codes the OpenAI backend, defeating the `LLMBackend` interface.
**Fix:** add a `withModel(model: string): LLMBackend` method to the `LLMBackend`
interface (each backend clones itself). `cycleModel` calls
`this.backend = this.backend.withModel(newModel)`. Removes the import and the
private-field reach-in. Closes the 3 live TS errors.

**Applied:** added `LLMBackend.withModel()` + `readonly model`; `OpenAIBackend`
clones itself; `cycleModel` calls `this.backend.withModel(newModel)`.

### A2. `drainQueuedMessages` is dead ✅ DONE
**File:** `loop.ts:992-1006`
Steering/follow-up draining moved to `harness.withDrainHook`; this private
helper has no callers. **Fix:** delete.

**Applied:** deleted the helper; also removed now-orphaned
`steeringQueueMode`/`followUpQueueMode` fields from the loop (drain lives in the
harness).

### A3. Stale "eventual hook bus" comments ✅ DONE
**Files:** `builtin-hooks.ts:5-6`, references to `AGENT_IMPROVEMENTS.md #7`
HookBus already exists and is used. Comments say it's future work.
**Fix:** drop the "see #7 for the eventual typed hook bus" notes.

---

## B. Duplication — high payoff

### B1. Six near-identical `run<Hook>` wrappers
**File:** `loop.ts:547-687` (`runBeforeToolCall`, `runAfterToolCall`,
`runPrepareNextTurn`, `runShouldStopAfterTurn`, `runGetSteeringMessages`,
`runGetFollowUpMessages`)
Every one is the same shape: `if (!this.hooks.X) return default; try { return
await this.hooks.X(ctx) } catch (e) { emit error; return default }`. ~140 lines
of boilerplate.
**Fix:** one private generic:
```ts
private async runHook<E extends keyof AgentLoopHooks>(
  event: E,
  ctx: Parameters<NonNullable<AgentLoopHooks[E]>>[0],
): Promise<Awaited<ReturnType<NonNullable<AgentLoopHooks[E]>>> | undefined>
```
that wraps the try/catch + error event once. Callers shrink to one line.
Note the HookBus already guards handler errors (`guard()`), so this layer is
double-guarding — see B2.

### B2. Error-guarding happens twice (HookBus + loop wrappers)
**Files:** `hook-bus.ts:298-311` (`guard`) and `loop.ts:547-687`
`composeHooks`/`withDrainHook` return a HookBus that already swallows handler
exceptions per-source. The loop's `run<Hook>` wrappers then wrap the *composed*
handler in another try/catch. The loop's catch can now only fire on a bug in
the bus reducer itself — effectively dead for handler errors.
**Fix:** decide one owner. Recommend: HookBus owns error isolation (it has
`source` for diagnostics + `onError`). Loop wrappers become thin
no-try calls (folds into B1). Wire `HookBus.onError` to emit the loop's
`error` event so diagnostics aren't lost.

### B3. Two-layer HookBus composition
**Files:** `harness.withDrainHook` (builds bus #1) → `config.hooks` →
`loop.run` → `composeHooks(builtin, config.hooks)` (builds bus #2)
Every prompt builds **two** HookBus instances and flattens one into the other.
Works, but the harness bus is immediately re-wrapped as a single handler inside
the loop's bus, so source attribution from the harness bus is lost.
**Fix:** single composition point. Either (a) harness passes its queue handlers
as plain `AgentLoopHooks` and lets the loop's `composeHooks` register
`builtin` + `harness-queues` + `user` into one bus, or (b) loop accepts a
prebuilt bus. Option (a) is smaller and keeps one bus with full source tags
(`builtin` / `harness-queues` / `user`).

### B4. Compaction logic triplicated
**Files:** `messages.ts` (`compactMessagesForContext`, `microCompactMessages`),
`harness.ts:202-270` (`compact`)
`harness.compact` reimplements the same pipeline messages.ts already exports:
system/non-system split, `keepRecentMessages`, tool-pair tail adjustment,
`<context-compaction>` wrapper, micro-then-full fallback. Only genuinely new
part is the **LLM summary** pass (`generateSummary`).
**Fix:** extract the shared skeleton into messages.ts (e.g.
`compactMessages(messages, { summarize })` where `summarize` is an optional
async hook). `harness.compact` passes its `generateSummary`; the proactive
builtin passes none. Removes ~50 lines from harness and the duplicated
tool-pair / keep-recent constants.

### B5. `0.65` / `0.8` / keep-recent magic numbers scattered
**Files:** `loop.ts:333` (`*0.65`), `harness.ts:209` (`*0.65`),
`builtin-hooks.ts:19,90` (`0.8`, `*0.65`), `messages.ts:106` (`8`)
Same compaction target fraction (`0.65`) hard-coded in 3 files.
**Fix:** single `COMPACTION_TARGET_FRACTION` const (co-locate with the
extracted compaction helper from B4).

---

## C. API shape — SOTA / maintainability

### C1. `LLMBackend.generate` has 12 positional params
**Files:** `backend.ts:14-28` (interface), `backend.ts:57-69` (impl),
called at `loop.ts:254` and `harness.ts:290`
Callers pass `undefined, undefined, undefined` placeholders to reach later
args. Adding a param means touching the interface, impl, and every call site by
position. Not SOTA.
**Fix:** collapse to `generate(messages, options)` where `options` is:
```ts
interface GenerateOptions {
  tools?; temperature?; maxTokens?; signal?; thinkingLevel?;
  callbacks?: {
    onDelta?; onThinking?; onTextStart?; onTextEnd?;
    onToolCallStart?; onToolCallDelta?;
  };
}
```
Two call sites only — low-risk mechanical change, big readability win. The loop
call site (`loop.ts:254-306`, ~50 lines of positional callbacks) becomes a
named object.

### C2. `convertToLlm` logic lives in `types.ts`
**File:** `types.ts:39-49`
A runtime function (filter by role) in the types module. `messages.ts` already
owns message transforms (`convertToChatFormat`).
**Fix:** move `convertToLlm` to `messages.ts`; keep only the type exports in
`types.ts`. Update the import in `loop.ts:27`.

### C3. `addSteering` / `addFollowUp` monkey-patch `this.hooks`
**File:** `loop.ts:1013-1031`
They wrap-and-replace `this.hooks.getSteeringMessages` at runtime, capturing the
previous closure. After `composeHooks` returns a bus-backed handler, this builds
an ad-hoc handler chain *outside* the bus — bypassing source tagging and the
single-composition model from B3.
**Fix:** once B3 lands (loop owns the bus), these push to a harness/loop queue
that a registered `harness-queues` handler drains — same mechanism as steering
already uses. Removes the closure-rewrapping.

### C4. `estimateMessageTokens` vs `estimateChatPayloadTokens`
**File:** `messages.ts:69-83`
Two estimators: one over raw `Message[]`, one over chat-format + tools. They
disagree (raw JSON vs chat JSON), and compaction mixes them
(`compactMessagesForContext` uses raw; loop/builtins use payload). Token
budgets compared across the two are apples-to-oranges.
**Fix:** standardize on `estimateChatPayloadTokens` everywhere a budget
decision is made; keep the raw one only if a caller truly needs pre-conversion
size (audit shows none outside messages.ts).

---

## D. Lower priority / notes

- **D1. `hookMatcherValue` alias table** (`loop.ts:1101-1114`) duplicates the
  tool↔claude-code-name mapping that plugins also need (`plugins.ts:615`
  consumes `matcher_value`). Consider a shared `TOOL_NAME_ALIASES` const if
  plugins ever need the reverse map. Not urgent — single producer today.
- **D2. `isError: result.startsWith("Error:")`** (`loop.ts:842`) is a stringly
  error convention. Tools return strings; an error sentinel object would be
  cleaner but is a wide change. Leave unless tool error handling is reworked.
- **D3. `_sleep(ms, _turnId)`** takes an unused `_turnId`. Drop the param.
- **D4. `batchTerminate` getter** (`loop.ts:95-97`) wraps a field with no extra
  logic. Inline the field read.

---

## E. Steering queue — three storages, UI desync ✅ DONE

**Files:** `harness.ts`, `agent-bridge.ts`, `transcript.ts`, `tui.ts`

### Problem (why the queue wasn't showing)
Steering messages lived in **three** places:
1. `harness.steeringQueue` — the real queue that gets drained.
2. `bridge._steeringMessages` — a mirror for UI, kept in sync by
   `_removeConsumedSteeringMessages` (matched steering text against assistant
   message bodies — fragile, false-positive prone).
3. `transcript._steerQueue` — a third copy populated from `queue_update`.

`tui.ts` set the SteerQueue widget from **two** paths (`queue_update` handler
*and* `transcript.onChange`). The actual display bug: when a user typed while a
turn was running, `sendMessage` called `harness.steer()` **directly**, skipping
`bridge.steer()` — so neither the mirror nor `queue_update` fired and the
widget never showed the item.

### Fix (applied)
Single source of truth = **harness queues**.
- Harness gains `getQueues()`, `setOnQueueChange(cb)`, `clearQueues()`, and an
  `emitQueueChange()` fired on every enqueue, **drain** (inside the
  `withDrainHook` closures), abort, prompt-clear, and nextTurn drain.
  `nextTurnQueue` is now `private`.
- Bridge dropped `_steeringMessages` / `_followUpMessages` and the
  text-matching `_removeConsumedSteeringMessages`. `steer/followUp/nextTurn`
  delegate to the harness; `_emitQueueUpdate` reads `harness.getQueues()`;
  `setOnQueueChange` wires harness changes → `queue_update`. The in-flight path
  in `sendMessage` now calls `this.steer()` (the bug fix).
- Transcript dropped `_steerQueue` / `_followUpQueue` /
  `handleQueueUpdate` — it holds conversation turns, not queue state.
- `tui.ts` drives the SteerQueue widget **only** from the `queue_update` event;
  removed the redundant `transcript.onChange` sync.

Result: queue updates (including drains as the loop consumes them) reach the UI
live, from one authoritative store.

---

## Suggested order (low-risk → higher)

1. **A1-A3** dead/broken code — unblocks typecheck, zero behavior change.
2. **C2, D3, D4** trivial moves/cleanups.
3. **B1+B2** collapse hook wrappers + single error owner.
4. **B3+C3** single HookBus composition; retire monkey-patching.
5. **B4+B5** unify compaction + constants.
6. **C1** backend `generate` → options object.
7. **C4** token-estimator standardization (verify budgets unchanged).

Each step is independently shippable and typecheck-gated. No step changes the
external loop contract (`run()` → `Message[]`).
