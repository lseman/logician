# Agent Improvements — Gap Analysis

Source: `repos/pi` (agent-core, coding-agent, TUI), `repos/openclaude` (query loop, tools, services), compared against `logician-tui/src/agent-core/`.

---

## 1. Recovery & Retry

### 1.1 Auto-retry on provider errors (pi/coding-agent)
**What:** On API errors (429, 500, 502, 503, 504, rate limit, connection lost, websocket closed, timeout), pi auto-retries with exponential backoff. Emits `auto_retry_start` / `auto_retry_end` events. Cancels with `abortRetry()`.

**Current:** logician-tui retries once on context-full errors only (inner loop `for (let attempt = 0; attempt < 2)`). No backoff. No retry on provider errors.

**Gap:** No auto-retry for transient provider errors. No exponential backoff. No retry events for the TUI to display.

**Action:** Add `_retryAttempt` counter, `_prepareRetry()` with exponential backoff, `auto_retry_start/end` events, `abortRetry()` method. Wire into loop after `backend.generate()` failure.

### 1.2 Context overflow recovery with compaction + retry (pi/coding-agent)
**What:** When context overflows, pi compacts then auto-retries. If recovery fails, emits a message suggesting context reduction or model switch.

**Current:** logician-tui compacts on context-full error and retries once (same inner loop). No compaction on non-error provider failures.

**Gap:** Compaction only on context-full. No compaction-then-retry on other errors. No recovery message on failed recovery.

**Action:** Move compaction trigger outside the error-only branch. On any LLM failure, check if compaction can help, compact, retry.

### 1.3 Tool failure loop guard (openclaude + logician-tui)
**What (openclaude):** `toolFailureLoopGuard.ts` — tracks tool call signatures, error categories, and file paths. Trips after N identical failures (default 3). Resets on any success from same tool. Distinguishes user-interrupt from real errors.

**What (logician-tui):** `guards.ts` — `GuardEngine` does the same thing (duplicate detection, failure signature, failure category, path-based). Already ported.

**Gap:** logician-tui guard is already ported and functional. Minor gap: openclaude resets path counters on successful mutations to the same path; logician-tui resets only on successful tool execution (same behavior). **No action needed.**

---

## 2. Agent Loop Architecture

### 2.1 Steering messages & follow-up messages (pi/agent-loop)
**What:** `getSteeringMessages()` injects messages before each assistant response. `getFollowUpMessages()` injects messages when the agent would stop (outer loop). Enables multi-agent handoff, auto-continue, and user-injected steering without blocking the main thread.

**Current:** logician-tui uses `continueAfterTurn` hook — injects ONE user message and loops. No steering/follow-up distinction. No outer loop for queued follow-ups.

**Gap:** Single continuation message vs. queued message pipeline. No steering messages (pre-response injection). No follow-up queue.

**Action:** Replace `continueAfterTurn` with `getSteeringMessages()` + `getFollowUpMessages()` contract hooks. Add outer loop that checks for follow-up messages when agent would stop.

### 2.2 prepareNextTurn / shouldStopAfterTurn (pi + logician-tui)
**What:** `prepareNextTurn` can mutate messages and swap model/thinking level per-turn. `shouldStopAfterTurn` decides early termination.

**Current:** logician-tui has both hooks wired. **No gap.**

### 2.3 Model switching per-turn (pi/agent-session)
**What:** `prepareNextTurn` can return a new model. `cycleModel()` cycles through configured models. `setModel()` sets model with session override.

**Current:** logician-tui has no model switching. Single model per run.

**Gap:** No model cycling. No per-turn model swap. No `model_select` events.

**Action:** Add `cycleModel()` (forward/backward through configured models), `setModel()`, `model_select` event. Wire into `prepareNextTurn` hook.

### 2.4 Sequential vs parallel tool execution (pi/agent-loop)
**What:** `executeToolCalls` checks `executionMode === "sequential"` per tool or globally. Sequential tools execute one at a time; parallel tools execute concurrently with `Promise.all`.

**Current:** logician-tui always executes tool calls sequentially (for loop). No parallel mode.

**Gap:** No parallel tool execution. Slower when tools are independent.

**Action:** Add `executeToolCallsParallel()` — spawn all tool executions, await with `Promise.all`, preserve order.

---

## 3. Compaction

### 3.1 Proactive compaction (logician-tui + openclaude)
**What (logician-tui):** `builtin-hooks.ts` — proactive compaction with cooldown (5 turns). Micro-compaction first (truncate tool results), then full summarizing compaction.

**What (openclaude):** `autoCompact.ts` — token-threshold-based compaction with cooldown circuit breaker. `consecutiveFailures` tracking prevents retry storms. `microCompact.ts` — targeted tool result compaction for specific tools. `sessionMemoryCompact.ts` — session-level compaction.

**Gap:** logician-tui compaction works but lacks:
- Token threshold config (uses fixed fraction)
- Cooldown circuit breaker with consecutive failure tracking
- Micro-compaction targeting specific tool types (openclaude has `COMPACTABLE_TOOLS` set)
- Branch summarization (pi has `generateBranchSummary`, `collectEntriesForBranchSummary`)

**Action:** Add token threshold config. Add `consecutiveFailures` to compaction state. Implement `COMPACTABLE_TOOLS`-style targeting for micro-compaction.

### 3.2 Branch summarization (pi/coding-agent)
**What:** Session branching — fork a session, run work, summarize the branch into a compact entry. `prepareBranchEntries`, `collectEntriesForBranchSummary`, `generateBranchSummary`.

**Current:** logician-tui has `UndoStack` (file-level undo). No session branching. No branch summarization.

**Gap:** No session fork/branch. No branch summarization for context management.

**Action:** Add session branching: `forkSession()`, branch summarization, compact branch entries into parent session.

---

## 4. Task Management

### 4.1 TodoWrite tool (openclaude + logician-tui)
**What (openclaude):** `TodoWriteTool` — stores todos in AppState keyed by agent/session. Returns `oldTodos`/`newTodos`. Has `verificationNudge` — when 3+ tasks closed without a verification step, nudges the agent to spawn a verification sub-agent.

**What (logician-tui):** `todo-write.ts` — `todoWrite` tool with `TodoItem`/`TodoStatus`. `onTodosChanged` event. `getTodos()` accessor.

**Gap:** logician-tui todo tool lacks:
- `oldTodos`/`newTodos` diff in result (openclaude returns both)
- Verification nudge on task completion
- Structural nudge for multi-step workflows

**Action:** Add `oldTodos`/`newTodos` to tool result. Add verification nudge when all tasks complete and count >= 3.

### 4.2 Multi-agent task system (openclaude)
**What:** `Task.ts` — 7 task types: `local_bash`, `local_agent`, `remote_agent`, `in_process_teammate`, `local_workflow`, `monitor_mcp`, `dream`. Task lifecycle: pending → running → completed/failed/killed. File-based output with offset tracking.

**What (logician-tui):** No multi-agent task system. Single agent loop.

**Gap:** No sub-agent spawning. No task lifecycle management. No task output files.

**Action:** Add `AgentTask` type with lifecycle. Add `spawnAgent()` for sub-agent tasks. File-based output with offset tracking.

### 4.3 Verification agent (openclaude)
**What:** `verificationAgent.ts` — dedicated sub-agent type that tries to break the implementation. Strictly prohibited from modifying project files. Runs tests, checks edge cases, validates against success criteria. Receives: original task, files changed, approach taken, plan file.

**Current:** logician-tui has no verification agent.

**Gap:** No post-implementation verification step. No dedicated verifier sub-agent.

**Action:** Add `VerificationAgent` tool — spawns a read-only sub-agent that validates changes against the original task.

---

## 5. Context Management

### 5.1 Context collapse (openclaude)
**What:** `contextCollapse/index.ts` — selectively removes older tool results while preserving granular context. Used as fallback when proactive compaction fails.

**Current:** logician-tui has no context collapse. Only full compaction.

**Gap:** No selective context pruning. No collapse drain before full compaction.

**Action:** Add context collapse — selectively truncate older tool results by tool type, preserving recent interactions.

### 5.2 Token budget tracking (logician-tui + openclaude)
**What (logician-tui):** `budget.ts` — `BudgetTracker` detects diminishing returns (two consecutive turns below token delta floor after min continuations). Stops early.

**What (openclaude):** `tokenBudget.ts` — similar diminishing-returns detection. Also tracks per-turn token deltas with exponential decay.

**Gap:** logician-tui budget tracker is already ported. **No gap.**

---

## 6. TUI Features

### 6.1 Kill ring (pi + logician-tui)
**What:** Emacs-style kill/yank ring. Consecutive kills accumulate. Yank-pop cycling.

**Current:** logician-tui has `kill-ring.ts` — full implementation. **No gap.**

### 6.2 Undo stack (pi + logician-tui)
**What (pi):** `UndoStack` — deep clone on push, pop returns detached snapshot, clear(), length.

**What (logician-tui):** `undo-stack.ts` — past/future stacks with max depth, push/pop/peek/redo/hasPast/hasFuture. More features than pi version.

**Gap:** logician-tui has MORE features than pi. **No gap.**

### 6.3 Plan mode (openclaude)
**What:** `/plan` command — enters plan mode, writes a plan file, restricts tools to read-only. `/ultraplan` — remote multi-agent planning with Opus model. `ExitPlanModeTool` — exits plan mode, re-enables tools.

**Current:** logician-tui has no plan mode. No read-only restriction mode.

**Gap:** No plan mode. No tool restriction during planning.

**Action:** Add plan mode — restrict tools to read-only, write plan file, exit plan mode to resume full tools.

### 6.4 Session management (pi/coding-agent)
**What:** Session branching, session switching, session export to HTML, session header management, compaction entries, branch entries.

**Current:** logician-tui has `/new`, `/sessions`, `/load`, `/export` (bridge commands). No local session manager.

**Gap:** No local session persistence. No session branching. No HTML export.

**Action:** Add local session manager — persist sessions to disk, support branching, export to HTML.

---

## 7. Error Handling & Diagnostics

### 7.1 Provider fallback (openclaude)
**What:** On rate limit / 429, switches to a fallback provider/model. One-shot guard prevents repeated fallbacks in same session.

**Current:** logician-tui has no provider fallback. Single provider.

**Gap:** No automatic provider/model fallback on rate limits.

**Action:** Add fallback provider config. On 429/rate limit, switch to fallback model/provider.

### 7.2 Diagnostics (pi/coding-agent)
**What:** `diagnostics.ts` — session diagnostics, model info, context usage, tool availability, compaction status, retry status.

**Current:** logician-tui has `/status` command (state dispatch). No structured diagnostics.

**Gap:** No session diagnostics endpoint. No structured health check.

**Action:** Add diagnostics endpoint — session state, model, context tokens, tool count, compaction status, retry status.

---

## 8. Hooks & Plugins

### 8.1 Hook bus (logician-tui)
**What:** `hook-bus.ts` — structured hook bus with typed events. `buildBuiltinHooks()` composes guards, budget, compaction. `composeHooks()` merges user hooks with builtins.

**Gap:** Already well-structured. Minor gap: no `PostToolUse` hook for post-execution processing (openclaude has it).

**Action:** Add `PostToolUse` hook for post-execution processing (e.g., auto-compact, memory extraction).

---

## Priority Summary

| Priority | Feature | Source | Effort |
|----------|---------|--------|--------|
| P0 | Auto-retry with backoff on provider errors | pi/coding-agent | Low |
| P0 | Model cycling (`cycleModel`) | pi/coding-agent | Low |
| P0 | Provider fallback on rate limit | openclaude | Medium |
| P1 | Steering/follow-up message pipeline | pi/agent-loop | Medium |
| P1 | Parallel tool execution | pi/agent-loop | Low |
| P1 | Verification nudge on todo completion | openclaude | Low |
| P1 | Context collapse (selective pruning) | openclaude | Medium |
| P2 | Plan mode (read-only restriction) | openclaude | Medium |
| P2 | Branch summarization | pi/coding-agent | High |
| P2 | Token threshold config for compaction | openclaude | Low |
| P2 | Post-implementation verification agent | openclaude | High |
| P3 | Multi-agent task system | openclaude | High |
| P3 | Local session manager | pi/coding-agent | High |
| P3 | Provider fallback + retry events for TUI | pi + openclaude | Medium |
