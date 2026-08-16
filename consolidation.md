# Pi vs Logician Agent Control Loop Analysis

## Pi Core File Paths

### Entrypoint & Agent Loop
- [**agent.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/agent.ts#L173): `Agent` class, lifecycle state management, stream setup
- [**agent-loop.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/agent-loop.ts#L31): `agentLoop()` & `agentLoopContinue()` functions, main loop dispatcher
- [**stream-fn.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/stream-fn.ts): Stream function registration

### Tool Execution
- [**agent-loop.ts (lines 200-500+)**](file:///home/seman/logician/repos/pi/packages/agent/src/agent-loop.ts#L200):
  - `executeToolCalls()` (dispatcher)
  - `executeToolCallsSequential()`, `executeToolCallsParallel()` (impl)
  - `prepareToolCall()`, `executePreparedToolCall()` (preparation/execution)
  - `failToolCallsFromTruncatedMessage()` (truncation handling)

### Session & State Persistence
- [**harness/session/types.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/harness/session/types.ts): Entry types (`MessageEntry`, `CompactionEntry`, `ModelChangeEntry`)
- [**harness/session/session.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/harness/session/session.ts): Session interface (in-memory & SQLite variants)
- [**types.ts (AgentState)**](file:///home/seman/logician/repos/pi/packages/agent/src/types.ts#L327): Transcript, tools, model, thinking level

### Model & Streaming
- [**ai/models.ts**](file:///home/seman/logician/repos/pi/packages/ai/src/models.ts): `Models` interface, auth, model listing, streaming delegation
- [**proxy.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/proxy.ts): Server proxy for centralized auth/streaming
- [**stream-fn.ts**](file:///home/seman/logician/repos/pi/packages/agent/src/stream-fn.ts): StreamFn callback contract

### Hooks & Extension System
- [**types.ts (BeforeToolCallContext, AfterToolCallContext)**](file:///home/seman/logician/repos/pi/packages/agent/src/types.ts#L92): Hook context types
- [**agent.ts (beforeToolCall, afterToolCall, listeners)**](file:///home/seman/logician/repos/pi/packages/agent/src/agent.ts#L106): Hook dispatch on `Agent` class
- [**harness/agent-harness.ts (HookName, Hooks interface)**](file:///home/seman/logician/repos/pi/packages/agent/src/harness/agent-harness.ts#L197): Full harness hook system (`before_run`, `before_request`, `after_response`, `after_tool`, etc.)

### Tests
- [**test/agent.test.ts**](file:///home/seman/logician/repos/pi/packages/agent/test/agent.test.ts): Agent lifecycle, subscriber tests
- [**test/agent-loop.test.ts**](file:///home/seman/logician/repos/pi/packages/agent/test/agent-loop.test.ts): Event emission, message conversion
- [**test/e2e.test.ts**](file:///home/seman/logician/repos/pi/packages/agent/test/e2e.test.ts): Full end-to-end with harness

---

## Logician Core File Paths

### Entrypoint & Agent Loop
- [**run-kernel.ts**](file:///home/seman/logician/packages/agent-core/src/agent/run-kernel.ts#L105): `RunKernel` class, versioned event ledger, session state
- [**agent-loop-runner.ts**](file:///home/seman/logician/packages/agent-core/src/agent/agent-loop-runner.ts): `runAgentLoop()` function, event sequence, budget tracking
- [**run-kernel-events.ts**](file:///home/seman/logician/packages/agent-core/src/agent/run-kernel-events.ts#L18): Event types, reduction semantics

### Tool Execution
- [**runtime/tool-batch-controller.ts**](file:///home/seman/logician/packages/agent-core/src/runtime/tool-batch-controller.ts): `executeToolBatch()`, permission gates, tool result building
- [**tools/shared/text-to-tool-calls.ts**](file:///home/seman/logician/packages/agent-core/src/tools/shared/text-to-tool-calls.ts): Text XML parsing, tool call extraction

### Session & State Persistence
- [**session.ts**](file:///home/seman/logician/packages/agent-core/src/agent/session.ts): JSONL-based session (`.logician/sessions/<id>/`), MessageSessionEntry
- [**runtime-state.ts**](file:///home/seman/logician/packages/agent-core/src/agent/runtime-state.ts): RuntimeState (messages, model, tools, thinking level)
- [**trajectory.ts**](file:///home/seman/logician/packages/agent-core/src/agent/trajectory.ts): Turn-by-turn trajectory tracking

### Model & Streaming
- [**backend.ts**](file:///home/seman/logician/packages/agent-core/src/agent/backend.ts): Provider call setup, streaming, error handling
- [**run-budget.ts**](file:///home/seman/logician/packages/agent-core/src/agent/run-budget.ts): Token/call budgeting

### Hooks & Extension System
- [**agent/types/types-hooks.ts**](file:///home/seman/logician/packages/agent-core/src/agent/types/types-hooks.ts#L149): `AgentHooks` interface (`beforeAgentStart`, `beforeToolCall`, `afterToolCall`, `prepareNextTurn`, `transformContext`, `beforeProviderRequest`, etc.)
- [**hooks/extensions/events.ts**](file:///home/seman/logician/packages/agent-core/src/hooks/extensions/events.ts): Typed `ExtensionEventBus`, events like `BeforeAgentStartEvent`, `TurnEndEvent`
- [**runtime/runner.ts**](file:///home/seman/logician/packages/agent-core/src/runtime/runner.ts): Extension event dispatching

### Tests
- [**__tests__/agent-loop-runner.test.ts**](file:///home/seman/logician/packages/agent-core/src/__tests__/agent-loop-runner.test.ts): Loop tests, steering, tool batch execution
- [**__tests__/run-kernel.test.ts**](file:///home/seman/logician/packages/agent-core/src/__tests__/run-kernel.test.ts): Kernel state reduction, event replay
- Multiple test files for guards, extensions, branching

---

## Comparison Matrix

| Dimension | Pi | Logician |
|-----------|-----|----------|
| **Loop Entrypoint** | `agentLoop(prompts, context, config)` in agent-loop.ts | `runAgentLoop(prompts, config, emit)` in agent-loop-runner.ts |
| **State Container** | `Agent` class (mutable state, listeners) | `RunKernel` (event ledger, reduction semantics) |
| **Session Storage** | Pluggable: in-memory, SQLite-based Session repo | JSONL-based `.logician/sessions/<id>/` files |
| **Tool Execution** | `executeToolCalls()` (seq/parallel, hooks) | `executeToolBatch()` (permissions, receipts, recovery) |
| **Tool Hooks** | `beforeToolCall`, `afterToolCall` on Agent | `beforeToolCall`, `afterToolCall`, + permission gates |
| **Model/Provider** | `Models` interface, provider-agnostic delegation | Direct backend.ts calls, provider-specific logic |
| **Streaming** | StreamFn callback, proxy support | Backend streaming with event framing |
| **Extension Hooks** | Agent-level listeners + Harness hooks | Typed ExtensionEventBus + AgentHooks |
| **Extension Pattern** | Per-hook callbacks (ad-hoc) | Event bus + handlers (scalable, decoupled) |
| **Budget Tracking** | Token usage in AgentState | Per-resource budgets (provider_call, tool_call, token) |
| **Error Recovery** | Custom message injection | Recovery modes (pure, idempotent, receipt_recoverable) |
| **Message Format** | AgentMessage[] → convertToLlm() → Message[] | Hybrid: Message[] + streaming tool calls |
| **Concurrency** | Sequential/parallel tool execution | Parallel with receipt tracking |

---

## Transferable Practices (Pi → Logician)

1. **StreamFn abstraction**: Pi's delegated streaming (agents.ts line 102) decouples provider logic—Logician could wrap backend.ts streaming identically.

2. **Hook context types**: Pi's structured `BeforeToolCallContext`, `AfterToolCallContext` (types.ts lines 92–110) provide clean contracts—Logician mirrors these but could adopt Pi's explicit `args: unknown` validation.

3. **Message conversion layer**: Pi's `convertToLlm()` callback allows custom message filtering—Logician could expose similar transform hooks.

4. **SessionStopReason enum**: Pi's `SessionStopReason` (exclude "pending", add "deferred") is cleaner than Logician's string-based stop reasons.

5. **Tool call truncation handling**: Pi's `failToolCallsFromTruncatedMessage()` explicitly handles output-token-limit truncation (agent-loop.ts line 381)—Logician's truncation logic could adopt this pattern.

6. **Listener pattern for events**: Pi's Agent.listeners Set (agent.ts line 175) is simpler than ExtensionEventBus for simple pub/sub.

---

## Non-Transferable Capabilities (Logician-specific)

1. **Event ledger with replay**: RunKernel's event-sourcing model enables deterministic replay and root-cause analysis—Pi's imperative loop cannot replay.

2. **Receipt-based tool recovery**: Idempotency keys + receipts allow safe retry/replay of partial tool results—Pi has no recovery model.

3. **Permission engine**: Tool-call gating via PermissionManager (tool-batch-controller.ts line 50+) is granular and audit-friendly—Pi has only hook-level control.

4. **Budget-per-resource**: Logician tracks tool_call, provider_call, token separately—Pi only tracks token usage.

5. **Branching & summarization**: Logician's tree-based session with branch summaries enables exploration—Pi's linear transcript doesn't support this.

6. **Typed ExtensionEventBus**: Logician's event-type dispatch with result merging (events.ts) is more extensible than Pi's callback-per-hook model.

---

## Simplification Opportunities

**If Logician adopted Pi patterns:**
- Replace EventLedger with simpler callback-based hooks (trade: lose replay/auditability).
- Unify session storage around SQLite (Pi's SessionRepo pattern).
- Simplify tool result handling by removing receipts (trade: lose idempotent retry).

**If Pi adopted Logician patterns:**
- Add event-sourcing for crash recovery (add complexity).
- Introduce permission gates (security/compliance feature).
- Support branching via tree-based session (exploration feature).

---

## DETAILED CONSOLIDATION ASSESSMENT (2026-08-16)

### Current Duplicate Abstractions in Logician

1. **Loop Shape & Exit Paths** (agent-loop-runner.ts:506-1680)
   - THREE exit mechanisms: iteration limit, budget check, completion gate + acceptance
   - Provider budget check (line 508) duplicates while-condition (line 506)
   - Tool budget check (line 961) duplicates provider check logic
   - Token budget check (line 835) creates third budget decision path
   - **Cost**: 6+ conditional branches for same "stop" decision
   - **Pi comparison**: Pi has single loop condition + optional stop policies

2. **Execution Profile + Policies** (execution-policy.ts + agent-loop-runner.ts)
   - ExecutionProfile: "autonomous" | "minimal" controls embeddedPoliciesEnabled
   - Three independent policy layers:
     - ExecutionPolicy.embeddedPoliciesEnabled (autonomous mode)
     - AcceptanceContract (post-run verification)
     - CompletionGate (outcome resolution)
   - Used in 17 distinct locations in agent-loop-runner.ts
   - **Cost**: 3 layers of decision logic, ~150 LOC of conditional logic
   - **Pi comparison**: Pi has ThinkingLevel + inference params, no embedded policies

3. **Inference Modes & Thinking Levels**
   - InferenceMode: 10 predefined sampling parameter sets (auto, none, thinking-general, thinking-coding, etc.)
   - ThinkingLevel: "off" | "minimal" | "low" | "medium" | "high" | "xhigh"
   - Defined in: configuration/inference-modes.ts + types-config.ts
   - Used by:
     - TUI inference-settings.ts (cycle, persist, notify)
     - Coding-agent bridge (setInferenceMode, setThinkingLevel)
     - Agent-loop-runner (adaptive mode selection)
   - **Cost**: Replicated in both Logician and Pi; TUI layer adds configuration persistence
   - **Pi comparison**: Pi has ThinkingLevel only; inference params are per-model

4. **Provider Request/Response Abstraction**
   - Logician: loop/provider-request.ts + loop/provider-response.ts (abstraction layer)
   - Direct wrappers around backend.ts calls with hook injection
   - Normalizes provider responses (parse text tool calls, stop reason mapping)
   - **Cost**: 150 LOC of thin wrapper over backend
   - **Pi comparison**: Pi streams directly via StreamFn; no request/response abstraction

5. **Hooks System**
   - Logician: AgentHooks (15+ hook points) + ExtensionEventBus (typed event dispatch)
   - Pi: Agent listeners + hook callbacks per HookName
   - Logician hooks: beforeAgentStart, beforeProviderRequest, beforeToolCall, afterToolCall, afterProviderResponse, transformContext, etc.
   - **Cost**: Two parallel hook systems (AgentHooks + ExtensionEventBus)
   - **Pi comparison**: Single hook system per hook point; simpler callback model

6. **Kernel vs Agent State Management**
   - RunKernel: Event ledger with reduction semantics (durable, replay-enabled)
   - AgentHarness: Wraps RunKernel, adds phase/queue management
   - RuntimeState: Transient state (messages, model, tools, thinking)
   - Session: JSONL-based persistence
   - **Cost**: 4-layer hierarchy; hard to reason about state flow
   - **Pi comparison**: Agent class + Harness + Session (3 layers, simpler)

7. **Configuration Cascade**
   - AgentConfig: 100+ fields (inference modes, execution profile, guard configs, thinking levels, etc.)
   - AgentHarness: Accepts AgentConfig + backend + backend options
   - TUI inference-settings.ts: Overlays inference mode / thinking level with disk persistence
   - Coding-agent bridge: Stores thinkingLevel, inferenceMode, executionProfile locally
   - **Cost**: Config scattered across TUI, bridge, harness; multiple sources of truth
   - **Pi comparison**: Config centralized at Agent; TUI can read/write via accessor methods

### Modules to Keep/Merge/Delete (Phase 1: Consolidation)

#### KEEP (Core Agent Loop Contract)
1. **agent-loop-runner.ts**: Main loop, but refactored to single exit path
2. **run-budget.ts**: RunBudgetController (simple resource counter)
3. **run-kernel.ts**: Event ledger (durable, task-spanning state)
4. **session.ts**: JSONL session storage
5. **messages.ts**: Message utilities (conversion, truncation)
6. **backend.ts**: LLM backend abstraction
7. **tool-batch-controller.ts**: Tool execution & permission checking (Logician-specific)

#### MERGE (Reduce Duplication)
1. **execution-policy.ts** + **guards/acceptance-contract.ts** + **tasks/completion-gate.ts**
   → Single "OutcomeDecision" module
   - Outcome = { status, summary, source }
   - Policies evaluated at ONE point: after loop naturally idles
   - Accept/verify as optional post-loop steps, not in-band

2. **loop/provider-request.ts** + **loop/provider-response.ts**
   → Inline into agent-loop-runner.ts or thin backend wrapper
   - Remove thin abstraction; move hook injection to loop
   - Normalize responses in backend.ts

3. **types/types-hooks.ts** + **hooks/extensions/events.ts**
   → Single hook system (choose one pattern)
   - Option A: Adopt Pi's simpler callback-per-hook model
   - Option B: Standardize on ExtensionEventBus, flatten hook points

4. **inference-modes.ts** + **configuration/config.ts**
   → Consolidate to single definitions; move sampling params to backend config

#### DELETE (Overengineered)
1. **guards/output-guard.ts**: Replace with simple context-full check in loop
2. **guards/guard-engine.ts**: Merge loop detection into main loop or remove for "minimal" profile
3. **guards/loop-detector.ts**: Same as above
4. **tasks/adaptive-mode.ts**: Simplify objective extraction
5. **agents/tasks/todo-state.ts**: Not core loop contract
6. **agents/tasks/task-status-state.ts**: Merge into completion-gate outcome

#### HYBRID (Conditional on "autonomous" mode)
1. **harness/branching.ts**: Keep, but remove from main loop
2. **summaries/**: Keep, behind feature flag
3. **guards/guard-callbacks.ts**: Move loop detection to optional post-loop analysis

### Migration Sequence with Test Gates

#### Phase 1: Provider/Response Consolidation
**Goal**: Remove provider-request/provider-response abstraction layer

Steps:
1. Read provider-request.ts hook logic into agent-loop-runner.ts
2. Delete loop/provider-request.ts, loop/provider-response.ts
3. Inline hook calls at provider.stream() site
4. **Test gate**: All tests in agent-loop-runner.test.ts pass; identical behavior

#### Phase 2: Exit Path Unification
**Goal**: Single loop exit decision point

Steps:
1. Combine three budget checks (provider, tool, token) into `checkBudget()`
2. Move while-condition iteration guard + checkBudget() into single pre-loop check or loop-condition guard
3. Delete executionPolicy.embeddedPoliciesEnabled conditional branches
4. Flatten acceptance-contract + completion-gate into single "resolveOutcome()"
5. **Test gate**:
   - Acceptance.test.ts still passes (feature works)
   - Loop tests verify same # of iterations before exit

#### Phase 3: Config & Inference Consolidation
**Goal**: Single source of truth for inference modes, thinking levels, execution profile

Steps:
1. Move inference mode definitions to types-config.ts (not separate file)
2. Consolidate inference-settings.ts + coding-agent bridge config storage
3. Reduce AgentConfig fields (remove duplicates with StreamOptions)
4. **Test gate**:
   - Config validation tests pass
   - TUI cycle/persist tests pass
   - Bridge setInferenceMode / setThinkingLevel tests pass

#### Phase 4: Hook System Consolidation
**Goal**: Single hook registration pattern

Steps:
1. Choose hook model: callback-per-hook (Pi) vs event-bus (Logician current)
2. Migrate all hook registrations to chosen model
3. Update harness hook dispatch
4. **Test gate**:
   - All hook.test.ts files pass
   - No hook-dispatch behavior changes

#### Phase 5: Kernel Simplification
**Goal**: Reduce state hierarchy from 4 layers to 3

Steps:
1. Merge RuntimeState + RunKernel state into single contract
2. Keep Session (JSONL) as persistence layer only, not runtime
3. Simplify phase management (fewer intermediate states)
4. **Test gate**:
   - run-kernel.test.ts passes
   - Harness state management tests pass
   - Session load/save tests pass

### Compatibility Risks

#### HIGH RISK (Breaks External Contracts)
1. **Hook names/signature changes**
   - Risk: Plugins + skills depend on hook names
   - Mitigation: Keep hook names stable; move internal hook logic only

2. **AgentConfig field removal** (executionProfile if removed)
   - Risk: Existing harness callers pass executionProfile
   - Mitigation: Keep field, map to new "outcome decision mode" internally

3. **Stop policy evaluation timing**
   - Risk: Policies currently evaluated mid-loop; moving to post-idle changes behavior
   - Mitigation: Add "pre-loop" evaluation phase if policies rely on intermediate state

#### MEDIUM RISK (Behavior Changes)
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

#### LOW RISK (Internal Refactoring)
1. Loop detection (output-guard, loop-detector) — behind feature flag only
2. Task status state (todo, adaptive-mode) — not on critical path
3. Branch summaries (harness/branching) — optional feature

### Exact Files to Modify (by phase)

**Phase 1: Provider Consolidation**
- [agent-loop-runner.ts](agent-loop-runner.ts#L520-L650): Inline provider-request hooks
- DELETE [loop/provider-request.ts](loop/provider-request.ts)
- DELETE [loop/provider-response.ts](loop/provider-response.ts)
- [backend.ts](backend.ts): Keep as-is
- **Tests to update**: agent-loop-runner.test.ts, request-provider-turn.test.ts (delete or merge)

**Phase 2: Exit Path Unification**
- [agent-loop-runner.ts](agent-loop-runner.ts#L506-L1680): Refactor loop condition + exit logic
- [execution-policy.ts](execution-policy.ts): Simplify to single "outcome mode"
- [tasks/completion-gate.ts](tasks/completion-gate.ts): Flatten with acceptance contract
- DELETE [guards/acceptance-contract.ts](guards/acceptance-contract.ts) or merge into completion-gate
- **Tests to update**: agent-loop-runner.test.ts, completion-gate.test.ts, acceptance.test.ts

**Phase 3: Config Consolidation**
- [configuration/inference-modes.ts](configuration/inference-modes.ts): Move definitions to types-config
- [types/types-config.ts](types-config.ts#L69-L130): Consolidate AgentConfig fields
- [apps/tui/src/app/inference-settings.ts](inference-settings.ts): Simplify to bridge accessors
- [packages/coding-agent/src/application/agent-bridge.ts](agent-bridge.ts#L499-L510): Config storage via bridge only
- **Tests to update**: config.test.ts, bridge.test.ts, inference-settings.test.ts

**Phase 4: Hook System**
- [types/types-hooks.ts](types-hooks.ts): Flatten or adopt Pi model
- [hooks/extensions/event-bus.ts](extensions/event-bus.ts): Keep or delete based on chosen model
- [harness.ts](harness.ts#L600-L700): Update hook dispatch
- **Tests to update**: All hook*.test.ts files

**Phase 5: Kernel Simplification**
- [run-kernel.ts](run-kernel.ts#L100-L200): Merge with RuntimeState
- [runtime-state.ts](runtime-state.ts): Absorb into run-kernel
- [session.ts](session.ts): Keep persistence only
- **Tests to update**: run-kernel.test.ts, runtime-state.test.ts
