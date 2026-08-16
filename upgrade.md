# Logician → Pi-style Consolidation Plan

> Goal: Make Logician simpler, slimmer, more like Pi — fewer layers, single exit paths, one hook system,
> fewer config fields, and a clearer mental model. Keep Logician's unique strengths (event ledger,
> permissions, branching, memory/RAG) behind feature flags or optional layers.

---

## 1. Current State (as of 2026-08-16)

### Package sizes

| Package | Lines | Key files |
|---------|-------|-----------|
| `agent-core` | ~13K | agent-loop-runner.ts (1,383), harness.ts (2,216), run-kernel.ts (711), session.ts (708) |
| `coding-agent` | ~24K | agent-bridge.ts (2,617), slash-commands.ts (831), transcript.ts (1,241) |
| `memory` | ~11K | store/index.ts (4,770), hooks/memory-hooks.ts (586) |
| `tui` | ~5K | components/editor.ts (2,351), markdown.ts (861) |
| Other packages | ~3K | agent-capabilities, autoresearch, agent-eval, rag |

### Architecture comparison: Logician vs Pi

| Dimension | Logician | Pi |
|-----------|----------|-----|
| Loop entrypoint | `runAgentLoop()` in agent-loop-runner.ts | `agentLoop()` in agent-loop.ts |
| State container | RunKernel (event ledger) + Harness + RuntimeState + agent-settings | Agent class (mutable state + listeners) |
| Session storage | JSONL-based `.logician/sessions/<id>/` | Pluggable: in-memory, SQLite |
| Tool execution | `executeToolBatch()` (permissions, receipts, recovery) | `executeToolCalls()` (seq/parallel, hooks) |
| Model/provider | backend.ts + loop/provider-request/response.ts abstraction | StreamFn callback, direct delegation |
| Hooks | AgentHooks (15+ points) + ExtensionEventBus (typed event bus) | Agent listeners + hook callbacks per HookName |
| Budget tracking | Per-resource (provider_call, tool_call, token) | Token usage in AgentState |
| Error recovery | Receipt-based idempotent retry | Custom message injection |
| Message format | Hybrid: Message[] + streaming tool calls | AgentMessage[] → convertToLlm() → Message[] |
| Concurrency | Parallel with receipt tracking | Sequential/parallel tool execution |
| **Total agent-core** | **~13K lines** | **~4K lines** |

### Known duplication (from consolidation.md)

1. **Three exit paths** in agent-loop-runner.ts: iteration limit, provider budget check, tool budget check, token budget check — 6+ conditional branches for the same "stop" decision
2. **Three policy layers**: ExecutionPolicy.embeddedPoliciesEnabled + AcceptanceContract + CompletionGate — ~150 LOC of conditional logic
3. **Two hook systems**: AgentHooks (15+ points) + ExtensionEventBus — parallel registration patterns
4. **4-layer state hierarchy**: RunKernel → Harness → agent-settings → Session
5. **Config cascade**: AgentConfig (100+ fields) + TUI inference-settings + bridge config storage
6. **Thin abstraction layer**: loop/provider-request.ts + loop/provider-response.ts — 150 LOC of thin wrapper over backend

### Consolidation status (from consolidation.md)

| Phase | Status |
|-------|--------|
| Phase 1: Provider/Response | **Complete** — deleted provider-request.ts, split into provider-options/response/streaming |
| Phase 2: Exit Path Unification | **Complete** — exit-path.ts is the single decision point (resolveExit/resolveOutcome/evaluatePostTurn); agent-loop-runner.ts and completion-gate.ts flattened into it |
| Phase 3: Config & Inference | **Partial** — deleted inference-modes.ts, runtime-state.ts; created agent-settings.ts |
| Phase 4: Hook System | **Partial** — guard system consolidated (900→286 lines); ExtensionEventBus expanded (+192 lines) but AgentHooks still exists in parallel |
| Phase 5: Kernel Simplification | **Partial** — deleted runtime-state.ts; 4-layer hierarchy partially remains; harness/utilities.ts extracted (hashing/trajectory helpers) |

**Net lines removed**: 3,668 lines (since consolidation.md was created)

**2026-08-16 session notes**:
- Phase 1 (exit-path.ts) landed and is fully wired into agent-loop-runner.ts — verified via full agent-core test suite (422/422 passing).
- Phase 0 and Phase 5's planned deletions (`tasks/adaptive-mode.ts`, `tasks/run-task-state.ts`, `tasks/task-status-state.ts`, `tasks/todo-state.ts`, `guards/output-guard.ts`, `guards/guard-engine.ts`) have **not** happened — all are still actively imported by agent-loop-runner.ts, harness.ts, builtin-hooks.ts, and/or external packages (agent-capabilities, coding-agent). Deleting them is more invasive than originally scoped; treat those line items as needing a fresh usage audit before touching, not a straight deletion.
- Found and fixed two unrelated regressions while verifying: a silent temperature-override drop in `loop/provider-options.ts` (useProviderDefaults gate discarded explicit config.temperature), and an `executionProfile` default mismatch between agent-core's "minimal" (from the agent-settings consolidation) and three TUI placeholder fallbacks still hardcoded to "autonomous".
- Two known-flaky/pre-existing failures confirmed unrelated to this consolidation (present on clean HEAD before any of today's changes): `agent-capabilities/src/__tests__/delegation-runtime.test.ts` (2 tests) and `apps/tui/src/__tests__/pty-regression.test.ts` ("Ctrl+I changes and persists the execution profile").

---

## 2. Target State: Pi-style Architecture

### Design principles

1. **One loop, one exit path** — single while-condition, single stop check
2. **One hook system** — choose the simplest model that covers our needs
3. **3-layer state** — Agent (state + listeners) → Loop (execution) → Session (persistence)
4. **Single config entry point** — one settings object, no cascading
5. **Keep unique strengths** — event ledger, permissions, branching, memory/RAG as optional layers
6. **Delete over refactor** — if it's not core to the loop contract, cut it

### Target package structure

```
packages/
├── agent-core/          # Core agent loop (~4-5K lines, down from 13K)
│   ├── agent.ts         # Agent class: state, listeners, queue management
│   ├── agent-loop.ts    # Loop logic: stream, execute tools, emit events
│   ├── backend.ts       # LLM backend (keep as-is)
│   ├── run-kernel.ts    # Event ledger (keep, but simplify interface)
│   ├── session.ts       # JSONL persistence (keep, remove runtime semantics)
│   ├── run-budget.ts    # Simple resource counter
│   └── types/           # All types in 2-3 files
│
├── coding-agent/        # Orchestration (~15-18K lines, down from 24K)
│   ├── agent-bridge.ts  # Simplified bridge (merge tool-router, slash-commands)
│   ├── tools/           # Built-in tools (keep as-is)
│   ├── mcp/             # MCP integration (keep as-is)
│   ├── skills/          # Skill loading (keep as-is)
│   ├── sessions/        # Session management (keep as-is)
│   └── commands/        # Slash commands (keep as-is)
│
├── memory/              # Memory system (keep as-is, ~11K)
├── rag/                 # RAG system (keep as-is)
├── tui/                 # Terminal UI (keep as-is, ~5K)
└── autoresearch/        # Experiment loops (keep as-is)
```

---

## 3. Phase-by-Phase Plan

### Phase 0: Foundation (Low Risk) — **Partially superseded, see audit below**

**Goal**: Clean up what's already marked for deletion in consolidation.md.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| ~~0.1~~ | ~~Delete `tasks/todo-state.ts`~~ — **retained, public API** | tasks/todo-state.ts | — |
| ~~0.2~~ | ~~Delete `tasks/task-status-state.ts`~~ — **retained, public API** | tasks/task-status-state.ts | — |
| ~~0.3~~ | ~~Delete `tasks/run-task-state.ts`~~ — **retained, async-context isolation for concurrent loops** | tasks/run-task-state.ts | — |
| ~~0.4~~ | ~~Delete `tasks/adaptive-mode.ts`~~ — **retained, live sampling-mode selector for `inferenceMode: "auto"`** | tasks/adaptive-mode.ts | — |
| 0.5 | Flatten `tasks/completion-gate.ts` into `tasks/outcome-resolution.ts` | tasks/completion-gate.ts → outcome-resolution.ts | **Done** (2026-08-16, as part of Phase 2 exit-path work) |
| 0.6 | Update all imports | agent-loop-runner.ts, harness.ts, tests | **Done** |

**Test gate**: `bun run test` passes. No behavior changes.

---

### Phase 1: Exit Path Unification (Medium Risk)

**Goal**: Single loop exit decision point in agent-loop-runner.ts.

**Current state**: 3+ exit mechanisms (iteration limit, provider budget, tool budget, token budget, completion gate, acceptance contract) scattered across 1,383 lines.

**Target**: Single `checkStopConditions()` called at the top of each iteration.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| 1.1 | Extract `checkStopConditions()` — combine all budget checks + iteration guard | agent-loop-runner.ts | 2 hours |
| 1.2 | Merge `execution-policy.ts` + `guards/acceptance-contract.ts` + `tasks/outcome-resolution.ts` | execution-policy.ts, guards/acceptance-contract.ts, tasks/outcome-resolution.ts | 2 hours |
| 1.3 | Replace 3 budget check locations with single `checkStopConditions()` call | agent-loop-runner.ts | 1 hour |
| 1.4 | Move acceptance verification to post-loop (optional finalization step) | agent-loop-runner.ts, harness.ts | 2 hours |
| 1.5 | Delete redundant exit-path code | agent-loop-runner.ts | 1 hour |

**Expected reduction**: agent-loop-runner.ts 1,383 → ~900 lines.

**Test gate**:
- `agent-loop-runner.test.ts` passes
- `completion-gate.test.ts` passes
- `acceptance.test.ts` passes
- Verify same # of iterations before exit on existing test cases

---

### Phase 2: Hook System Consolidation (Medium-High Risk)

**Goal**: Single hook registration pattern.

**Current state**: Two parallel systems — `AgentHooks` (15+ hook points, callback-per-hook) and `ExtensionEventBus` (typed event dispatch with result merging).

**Decision**: Adopt Pi's callback-per-hook model. It's simpler and covers our needs. ExtensionEventBus can be removed for internal hooks; keep it only if plugins/skills require typed events.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| 2.1 | Audit all hook registrations — which system each caller uses | agent-loop-runner.ts, harness.ts, hooks/, tests | 2 hours |
| 2.2 | Choose which hook points to keep (eliminate duplicates between AgentHooks + ExtensionEventBus) | types/types-hooks.ts, hooks/extensions/event-bus.ts | 1 hour |
| 2.3 | Migrate harness hook dispatch to single callback model | harness.ts | 3 hours |
| 2.4 | Update agent-loop-runner.ts hook calls | agent-loop-runner.ts | 1 hour |
| 2.5 | Delete ExtensionEventBus (or keep as optional plugin layer) | hooks/extensions/event-bus.ts | 1 hour |
| 2.6 | Update all tests | hooks/*test*.ts | 2 hours |

**Expected reduction**: types/types-hooks.ts 222 → ~120 lines. harness.ts hook dispatch ~200 lines simpler.

**Test gate**:
- All hook.test.ts files pass
- No hook-dispatch behavior changes
- Plugin integration tests pass (if any)

---

### Phase 3: Config & Inference Consolidation (Low-Medium Risk)

**Goal**: Single source of truth for inference modes, thinking levels, execution profile.

**Current state**: AgentConfig (100+ fields) + TUI inference-settings.ts + bridge config storage + agent-settings.ts.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| 3.1 | Consolidate AgentConfig fields — remove duplicates with StreamOptions, inline simple types | types/types-config.ts | 2 hours |
| 3.2 | Remove `inferenceMode` from core config — keep only `thinkingLevel` + per-model sampling params | types/types-config.ts, agent-settings.ts | 1 hour |
| 3.3 | Update TUI inference-settings.ts to match new config shape | apps/tui/src/app/inference-settings.ts | 1 hour |
| 3.4 | Update coding-agent bridge config storage | packages/coding-agent/src/application/agent-bridge.ts | 1 hour |
| 3.5 | Delete inference-modes.ts remnants if any | configuration/inference-modes.ts | 15 min |

**Test gate**:
- Config validation tests pass
- TUI cycle/persist tests pass
- Bridge setInferenceMode / setThinkingLevel tests pass

**2026-08-16 progress**: Started 3.1 by grepping every `AgentConfig` field for real (non-declaration) readers across the whole monorepo, the same audit method used for Phase 0/5. Found 16 fully dead fields and removed them end-to-end (config type → coding-agent config parsing/bridge plumbing → TUI settings UI → tests):
- 15 fields with **zero** readers anywhere outside their own type declaration: `progressSignalEnabled`, `progressSignalMinScore`, `progressSignalMinLowScoreTurns`, `goalDecompositionEnabled`, `goalDecomposerMaxSubgoals`, `recoveryMemoryMaxEntries`, `hypothesisTrackingEnabled`, `hypothesisTrackerMaxHypotheses`, `guardFusionEnabled`, `guardFusionWeights`, `guardGraduatedIntervention`, `thinkingLoopMinThinkingLength`, `thinkingLoopThinkingOnlyThreshold`, `thinkingLoopEscalationRatio`, `thinkingLoopMetaReasoningThreshold` — these were config knobs for guard modules (`goal-decomposer.ts`, `hypothesis-tracker.ts`, `progress-signal.ts`, `thinking-loop-detector.ts`) already deleted in an earlier phase; the config fields just never got cleaned up alongside the modules.
- 1 field (`thinkingLoopDetectionEnabled`) had live plumbing all the way through config parsing → bridge → a TUI settings toggle ("Thinking-loop guard: Detect reasoning loops without action"), but no consumer in agent-core actually read it — the toggle was a silent no-op. Removed with explicit user sign-off rather than left as a fake feature.
- `recoveryMemoryEnabled` (sibling of the deleted `recoveryMemoryMaxEntries`) was kept — it genuinely gates real logic in `builtin-hooks.ts` (`recoveryMemoryOn`) even though the standalone `recovery-memory.ts` module is gone; the concept was reimplemented inline.
- Verified via full `bun run test`: agent-core 422/422, coding-agent 253/253, tui 257/258 (1 known pre-existing failure), agent-capabilities 29/31 (2 known pre-existing failures) — no new failures.
- Remaining Phase 3 work (3.1 continued, 3.2–3.5) not started: `AgentConfig` still has ~90 fields; `inferenceMode` dual-tracking (core config vs `thinkingLevel`) untouched; TUI `inference-settings.ts` not yet reconciled with `agent-settings.ts`'s shape.

**2026-08-16 progress, continued**:
- Audited `harness/phase.ts` and `harness/contracts.ts` (the last two unaudited Phase 0/5 deletion targets, see corrected table in §5) — both kept, small and actively used, merging into harness.ts would be a net regression.
- Removed `InferenceModeDef.thinking: boolean` — populated on all 10 mode definitions, read by nothing (sampling behavior comes entirely from `params`/`useProviderDefaults`; the loop's real thinking-effort control is the separate `settings.thinkingLevel`). Dead struct field, not user-facing.
- Investigated the "config cascade" duplication upgrade.md flags directly: `agent-settings.ts`'s `resolveAgentSettings()` was never exported from `agent-core`'s barrel (`index.ts`), so `coding-agent/agent-bridge.ts`'s `getSettingsData()` independently re-declared the same four defaults (`executionProfile`, `inferenceMode`, `maxIterations`, `thinkingLevel`) reading straight off `AgentConfig` — exactly the pattern that caused the `executionProfile` "minimal" vs "autonomous" mismatch bug fixed earlier this session. Exported `resolveAgentSettings`/`AgentSettings` and wired `getSettingsData()` to delegate for the 4 overlapping fields, closing the drift risk at its actual source.
- Deliberately left two things alone: (1) the `AgentCoreBridge` constructor's own config-building defaults (`opts.thinkingLevel ?? "off"` etc., ~agent-bridge.ts:502) — that's baking initial values into a fresh `AgentConfig` at construction time, a different concern from resolving live settings for display; (2) the TUI's hardcoded placeholder literals (`footer/layout.ts`, `widget-factory.ts`, `tui.ts` class-field initializers) — these are pre-load/no-config-in-scope defaults for display state, not derived from an `AgentConfig` object, so importing `resolveAgentSettings` there would be over-engineering rather than deduplication.
- `streamOptions` vs top-level `turnTimeoutMs`/`maxRetries`/`retryBaseDelayMs` duplication (flagged in 3.1's original scope) was investigated and found to be intentional, not accidental: the top-level fields are the public flat config surface used by `coding-agent` and `agent-capabilities` (subagent delegation), and `AgentHarnessStreamOptions`/`streamOptions` is the internal structured mirror the harness resolves them into (with a real, working `?? ` merge, not silent duplication). Left as-is — collapsing it would mean breaking one of two public APIs across 3 packages for a cosmetic win.
- Verified via full `bun run test` + `bun run typecheck` after every change: agent-core 422/422, coding-agent 253/253, tui 257/258 (1 known pre-existing failure), agent-capabilities 29/31 (2 known pre-existing failures), typecheck clean across all workspaces.

---

### Phase 4: Kernel Simplification (Medium Risk)

**Goal**: Reduce state hierarchy from 4 layers to 3.

**Current state**: RunKernel (event ledger) → Harness (phase + queue + steering) → agent-settings → Session (persistence).

**Target**: Agent (state + listeners + queue management) → Loop (execution) → Session (persistence). Event ledger becomes an optional feature of the agent, not a separate layer.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| 4.1 | Merge RuntimeState into Agent class (already mostly done) | run-kernel.ts, types.ts | 2 hours |
| 4.2 | Simplify phase management — reduce intermediate states | harness/phase.ts, harness.ts | 2 hours |
| 4.3 | Keep Session as persistence-only (remove runtime semantics) | session.ts | 1 hour |
| 4.4 | Move branching/summaries behind feature flag | harness/branching.ts, harness/compaction.ts | 1 hour |
| 4.5 | Simplify queue management | harness/queue-ops.ts | 1 hour |

**Expected reduction**: harness.ts 2,216 → ~1,500 lines.

**Test gate**:
- run-kernel.test.ts passes
- Harness state management tests pass
- Session load/save tests pass
- Branching tests pass (if kept)

---

### Phase 5: Guard System Cleanup (Low Risk) — **SUPERSEDED, see audit below**

**Goal**: Remove overengineered guards, keep essential ones.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| ~~5.1~~ | ~~Delete `guards/output-guard.ts`~~ — **retained, see audit** | guards/output-guard.ts | — |
| ~~5.2~~ | ~~Delete `guards/guard-engine.ts`~~ — **retained, see audit** | guards/guard-engine.ts | — |
| 5.3 | Keep `guards/loop-detector.ts` behind feature flag only | guards/loop-detector.ts | 30 min |
| 5.4 | Keep `guards/guard-callbacks.ts` — pi-style callbacks are clean | (keep) | — |
| 5.5 | Keep `guards/response-patterns.ts` — useful for non-committal detection | (keep) | — |

**Expected reduction**: ~~~750 lines deleted~~ 0 — deletion targets are load-bearing (see 2026-08-16 audit below).

**Test gate**:
- `guards-simplified.test.ts` passes
- Loop detection tests pass (if feature flag enabled)

---

### Phase 6: Coding-Agent Simplification (Medium Risk)

**Goal**: Reduce orchestration layer complexity.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| 6.1 | Merge tool-router into agent-bridge | application/tool-router.ts → agent-bridge.ts | 2 hours |
| 6.2 | Simplify event mapping (runtime/event-mapping.ts) | runtime/event-mapping.ts | 1 hour |
| 6.3 | Remove redundant session management layers | sessions/session-store.ts, sessions/transcript.ts | 1 hour |
| 6.4 | Clean up slash-commands dependencies on deleted modules | commands/slash-commands.ts | 1 hour |

**Expected reduction**: agent-bridge.ts 2,617 → ~1,800 lines.

**Test gate**:
- All coding-agent integration tests pass
- Slash command tests pass
- Session management tests pass

---

### Phase 7: TUI Cleanup (Low Risk)

**Goal**: Align TUI with simplified config and hook models.

| Step | Action | Files | Effort |
|------|--------|-------|--------|
| 7.1 | Update inference-settings.ts to new config shape | apps/tui/src/app/inference-settings.ts | 1 hour |
| 7.2 | Remove references to deleted guard systems | TUI components that reference guards | 1 hour |
| 7.3 | Simplify tool card rendering (remove guard-related UI) | TUI tool card components | 1 hour |

---

## 4. What to Keep (Logician's Unique Strengths)

These are NOT part of Pi but are valuable for Logician. Keep them, but make them optional:

| Feature | Current location | How to keep |
|---------|-----------------|-------------|
| Event ledger / replay | run-kernel.ts | Keep as optional feature; not required for basic loop |
| Permission engine | tool-batch-controller.ts + permissions.ts | Keep; it's a security feature, not loop logic |
| Receipt-based recovery | tool-batch-controller.ts | Keep; useful for idempotent tool execution |
| Branching & summaries | harness/branching.ts, summaries/ | Keep behind feature flag |
| Memory system | packages/memory/ | Keep as separate package |
| RAG system | packages/rag/ | Keep as separate package |
| Autoresearch | packages/autoresearch/ | Keep as separate package |
| Skills & plugins | packages/coding-agent/src/skills/, plugins/ | Keep |
| MCP integration | packages/coding-agent/src/mcp/ | Keep |

---

## 5. What to Delete

**2026-08-16 audit**: Grepped every module below for real (non-test) importers across `packages/` and `apps/` before touching anything. Result: every module upgrade.md proposed deleting for "not core loop contract" / "simplify" reasons is still actively imported — several are re-exported as public API from `agent-core/index.ts` and consumed by `coding-agent` and `agent-capabilities`, not just internal to agent-loop-runner.ts. **None of these are safe to delete as originally scoped.** The table below is corrected to reflect actual usage; treat "Reason" as historical intent, not current fact.

| Module | Lines | Original reason | Audit result |
|--------|-------|--------|--------|
| `tasks/todo-state.ts` | ~60 | Not core loop contract | **Keep** — public API (`agent-core/index.ts` exports `getTasks`/`onTodosChanged`/`Task`), consumed by `coding-agent/agent-bridge.ts` and `agent-capabilities/tasks/todo.ts` |
| `tasks/task-status-state.ts` | ~30 | Merge into outcome-resolution | **Keep** — public API (`agent-core/index.ts` exports `TaskStatusRecord` + helpers), consumed by `agent-capabilities/tasks/task-status.ts` and `exit-path.ts` |
| `tasks/run-task-state.ts` | ~30 | Merge into outcome-resolution | **Keep** — AsyncLocalStorage isolation so concurrent parent/child loops don't share todo/task_status state; used by agent-loop-runner.ts |
| `tasks/adaptive-mode.ts` | ~100 | Simplify objective extraction | **Keep** — live sampling-mode selector for `inferenceMode: "auto"`, has dedicated test coverage in agent-loop-runner.test.ts |
| `guards/output-guard.ts` | ~465 | Replace with simple check | **Keep** — wired as an optional loop-level guard (config.outputGuard) in agent-loop-runner.ts |
| `guards/guard-engine.ts` | ~286 | Merge into main loop | **Keep** — builtin-hooks.ts explicitly documents it as "the canonical source" for tool guard rails (duplicate + failure-loop detection) |
| `tasks/completion-gate.ts` | ~35 | Flatten into outcome-resolution | **Done** — flattened into outcome-resolution.ts as part of the 2026-08-16 exit-path work |
| `harness/phase.ts` | ~35 | Simplify phase hierarchy | **Keep** — small state-machine validator (`assertPhaseTransition`/`assertIdlePhase`), `HarnessBusyError` it defines is used by harness.ts, harness/queue-ops.ts, and surfaces through coding-agent's bridge. Merging into harness.ts would un-split an already well-factored file for no reduction in real complexity — works against the goal of shrinking harness.ts. |
| `harness/contracts.ts` | ~35 | Merge into harness.ts | **Keep** — pure type definitions (`AgentHarnessOptions`, `HarnessTurnSnapshot`, `HarnessQueues`, `AbortResult`); `AgentHarnessOptions` is consumed directly by coding-agent/agent-bridge.ts. Same reasoning as phase.ts. |
| `loop/provider-request.ts` | ~336 | Already deleted (Phase 1) | Confirmed gone |
| `configuration/inference-modes.ts` | ~254 | Already deleted (Phase 3) | Confirmed gone |
| `agent/runtime-state.ts` | ~140 | Already deleted (Phase 5) | Confirmed gone |
| `agent/tasks/task-state-controller.ts` | ~371 | Already deleted | Confirmed gone |
| `agent/guards/goal-decomposer.ts` | — | Already deleted | Confirmed gone |
| `agent/guards/hypothesis-tracker.ts` | — | Already deleted | Confirmed gone |
| `agent/guards/progress-signal.ts` | — | Already deleted | Confirmed gone |
| `agent/guards/recovery-memory.ts` | — | Already deleted | Confirmed gone |
| `agent/guards/thinking-loop-detector.ts` | — | Already deleted | Confirmed gone |

---

## 6. Migration Sequence with Dependencies

```
Phase 0 (Foundation)
  ↓
Phase 1 (Exit Path Unification)
  ↓
Phase 2 (Hook System) ← can overlap with Phase 1
  ↓
Phase 3 (Config Consolidation) ← can overlap with Phase 2
  ↓
Phase 4 (Kernel Simplification)
  ↓
Phase 5 (Guard Cleanup) ← can overlap with Phase 4
  ↓
Phase 6 (Coding-Agent)
  ↓
Phase 7 (TUI Cleanup)
```

**Total estimated effort**: ~40-50 hours of focused work.

**Expected outcome**:
- agent-core: 13K → ~5-6K lines (55-60% reduction)
- coding-agent: 24K → ~18K lines (25% reduction)
- Total: ~37K → ~23-24K lines (35-40% reduction)
- Simpler mental model: 3-layer state instead of 4-layer
- Single exit path instead of 3+
- One hook system instead of two

---

## 7. Compatibility Risks

### HIGH RISK (Breaks External Contracts)

1. **Hook name/signature changes**
   - Risk: Plugins + skills depend on hook names
   - Mitigation: Keep hook names stable; migrate internal hook logic only; deprecate ExtensionEventBus before deleting

2. **AgentConfig field removal** (executionProfile if removed)
   - Risk: Existing callers pass executionProfile
   - Mitigation: Keep field, map to new "outcome decision mode" internally; add deprecation warning

3. **Stop policy evaluation timing**
   - Risk: Policies currently evaluated mid-loop; moving to post-idle changes behavior
   - Mitigation: Add "pre-loop" evaluation phase if policies rely on intermediate state

### MEDIUM RISK (Behavior Changes)

1. **Acceptance verification timing**
   - Current: Runs inside loop as finalization turns
   - Proposed: Runs after loop exits as optional step
   - Risk: Reduces iterations available for acceptance
   - Mitigation: Keep finalization turns; move verification outside loop body

2. **Budget tracking granularity**
   - Current: Per-resource (provider_call, tool_call, token) with separate checks
   - Proposed: Single consolidated check
   - Risk: Loses detailed budget decision logging
   - Mitigation: Log full decision object at single checkpoint

### LOW RISK (Internal Refactoring)

1. Loop detection (output-guard, loop-detector) — behind feature flag only
2. Task status state (todo, adaptive-mode) — not on critical path
3. Branch summaries (harness/branching) — optional feature
4. Phase management (harness/phase.ts) — internal state only

---

## 8. Success Criteria

1. **Line count**: agent-core < 7K lines (down from 13K)
2. **Single loop exit**: One `checkStopConditions()` call in the loop
3. **Single hook system**: No parallel hook registration patterns
4. **3-layer state**: Agent → Loop → Session (no intermediate RuntimeState)
5. **Config**: AgentConfig < 60 fields (down from 100+)
6. **All tests pass**: No behavior regressions
7. **Mental model**: A new contributor can understand the agent loop in < 30 minutes

---

## 9. Notes on What NOT to Change

The following are intentionally NOT part of this consolidation:

- **Memory system** (`packages/memory/`) — keep as-is, it's a separate concern
- **RAG system** (`packages/rag/`) — keep as-is
- **TUI rendering** (`packages/tui/`) — keep as-is, it's a UI concern
- **Autoresearch** (`packages/autoresearch/`) — keep as-is
- **MCP integration** — keep as-is
- **Skills system** — keep as-is
- **Plugins system** — keep as-is (but update hook interfaces if Phase 2 changes them)
- **Provider/backend** (`backend.ts`) — keep as-is, it's a clean abstraction
