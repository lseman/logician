# logician-tui — Improvement Roadmap

An audit of the current agent core (`tui/src/agent-core/`) against state-of-the-art
agent harnesses (Claude Code, the Claude Agent SDK, pi, OpenAI Agents SDK), with
concrete proposals. Organized by subsystem; each item notes where it would land
in the codebase.

## Where we stand

The core is already solid and unusually well-layered for a TUI agent:

- **Inner loop** (`agent-core/loop.ts`): ReAct turn loop with typed backend
  errors (`BackendError{category,retryable}`), exponential-backoff retry,
  context-full compaction ladder (`compactToFit`), per-turn timeout that
  actually cancels the in-flight request, pure-tool result cache, parallel
  tool execution, empty-response recovery, pi-style continuations with a cap
  that only counts unproductive turns, and batch-terminate semantics.
- **Outer loop** (`agent-core/harness.ts`): guarded phase state machine
  (`idle | turn | compaction | branch_summary`), three queues (steering /
  follow-up / nextTurn) drained at save points, conversation fork/merge with
  LLM branch summaries, model cycling, optional reasoner pre-phase.
- **Hooks**: typed in-process `HookBus` with per-event reducer semantics and
  error isolation, layered builtin → harness → user; plus Claude-Code-style
  external settings hooks (`plugins.ts`: SessionStart, UserPromptSubmit,
  Pre/PostToolUse, Stop, …).
- **Skills** (`agent-core/skills.ts`): recursive SKILL.md discovery with
  ignore-file support, compact catalog injection + on-demand `read_skill`
  tool, `disable-model-invocation` honored.
- **Events** (`agent-core/events.ts`, `types.ts`): discriminated-union
  `AgentEvent` with streaming deltas, rich `turn_end` payloads
  (stopReason/message/toolResults), context updates, compaction events.

The gaps below are what separates this from a SOTA harness.

## 1. Subagents (highest impact)

There is no `Task`/subagent tool. Every exploration, review, or long search
burns the main context window. This is the single biggest lever for long
sessions.

- Add a `spawn_agent` tool that constructs a child `AgentHarness` with its own
  message history, a scoped tool registry (e.g. read-only for an "explorer"),
  its own `maxIterations`, and the same backend. Return only the child's final
  message to the parent as the tool result.
- Reuse existing pieces: the harness already supports independent instances;
  the loop already takes `initialMessages` and a `signal`. Wire the child's
  events into the parent emitter under a `subagent` envelope
  (`{ type: "subagent_event", agentId, event }`) so the TUI can render a
  collapsed child transcript.
- Agent definitions as markdown with frontmatter (mirror skills): name,
  description, allowed tools, model override. Discovery can share
  `loadSkills()` traversal logic.
- Add a `SubagentStop` external hook event in `plugins.ts` for parity with
  Claude Code.

## 2. Permission system

`PreToolUse` external hooks are fire-and-forget (`runHookSafely` in
`loop.ts` ignores results), and the in-process `beforeToolCall` can rewrite or
short-circuit but there is no user-facing approval flow.

- Honor blocking output from `PreToolUse` hooks: a hook returning a deny
  decision (or exit code 2, Claude Code convention) should become a
  `{ content: <reason>, isError: true }` tool result instead of executing.
- Add permission modes to `AgentConfig`: `ask | acceptEdits | acceptAll`,
  with an async `onPermissionRequest(toolCall) → allow | deny | always`
  callback the TUI implements as a modal. The natural insertion point is
  `prepareLoopToolCall()` between `beforeToolCall` and `PreToolUse`.
- Per-tool allowlist/denylist rules (glob on tool name + argument patterns,
  e.g. `bash(git *)`), persisted to the config file so "always allow" sticks.

## 3. Skills

Catalog + `read_skill` is the right architecture. Missing features:

- **Frontmatter extensions**: `allowed-tools` (constrain the loop while a
  skill is active), `argument-hint`, `model` override, `when-to-use` /
  trigger keywords. Parse them in `loadSkillFromFile()`; today only `name`,
  `description`, `disable-model-invocation` survive.
- **User-invocable slash skills**: let `/name args` in the input bar resolve
  to a skill and inject `formatSkillInvocation(skill, args)` as the prompt.
  The slash-popup component already exists for commands.
- **Scripts**: support a `scripts/` dir next to SKILL.md whose files are
  referenced by relative path in the body (resolve against `filePath` dir so
  the model can run them with the bash tool).
- **Hot reload**: watch skill dirs (`fs.watch`) and refresh the catalog
  between turns — the system prompt is rebuilt per run already, so this is
  just cache invalidation.
- **Catalog ranking**: the full catalog is injected every run. For large
  skill sets, rank or filter it per prompt (even plain keyword scoring on the
  user message, or BM25 over name/description) instead of always injecting
  everything.

## 4. Hooks

The in-process `HookBus` is good. Gaps:

- **Per-handler timeout**: one slow hook handler stalls the turn; wrap
  `guard()` with `withTimeout` (already in `async-utils.ts`) using a
  per-registration `timeoutMs`.
- **Priority/ordering**: registration order is implicit. Add an optional
  `priority` to `HookRegistration` so plugins can order deterministically
  without depending on load order.
- **External hook coverage**: `PreCompact`/`PostCompact` are declared in
  `plugins.ts` but never fired by the loop or harness compaction paths; emit
  them from `compactToFit` call sites and `harness.compact()`. Same for a
  `Notification` event when the agent goes idle waiting for input.
- **Hook decisions as data**: external hooks currently contribute only
  `additional_contexts`. Parse structured JSON output (`{decision, reason,
  additionalContext}`) like Claude Code so external hooks can block tools,
  rewrite prompts, or force continuation — that makes the external hook
  system as expressive as the in-process one.

## 5. Events

- **Envelope metadata**: add monotonic `seq` and `timestamp` to every event
  at the emit point (`EventEmitter.emit`). Required for replay, debugging,
  and ordering when events are forwarded over the bridge.
- **Persistence/replay**: the emitter keeps an in-memory ring of 1000.
  Turn-level persistence now exists (`session-store.ts`, SQLite), but
  event-level replay does not — streaming a JSONL event log alongside would
  let a crashed TUI re-render mid-turn state (partial tool output, thinking)
  that turn snapshots lose.
- **Missing events**: `hook_start`/`hook_end` (which hook ran, how long),
  `tool_permission_request`, `subagent_*`, `retry_give_up`. `thinking_delta`
  lacks `turnId` while every other delta carries it — unify.
- **Typed channel for UI state**: phase changes flow via `onPhaseChange`
  callback while everything else flows via events; emitting a
  `harness_phase` event too would give consumers one stream.

## 6. Inner loop

- **Token/cost budget**: `BudgetTracker` is opt-in and token-estimate based.
  Track real `usage` per response (already captured as `_lastUsageTokens`),
  accumulate into `TurnMetrics` (input/output/cached split), and support
  `maxCostUsd`/`maxTotalTokens` run budgets with a clean `budget_exhausted`
  stop reason instead of a boolean stop.
- **Retry-After**: on `rate_limit`, prefer the provider's `Retry-After`
  header (plumb through `BackendError`) over blind exponential backoff, and
  add jitter.
- **Structured stop instead of regex**: `declaresStop()` in
  `builtin-hooks.ts` sniffs prose. Offer the model an explicit `task_status`
  tool (`{status: done|blocked, summary}`); when registered, skip the regex
  entirely. Cheaper and far more reliable for continuation decisions.
- **Tool result blocks**: results are strings; support content-block arrays
  (text + image) end-to-end so screenshot-producing tools work. The
  `string | ToolResult` union already exists — extend `ToolResult` with
  typed blocks and teach `convertToChatFormat` to pass them through.
- **Mid-stream steering**: steering drains only at save points between LLM
  calls. Add an opt-in "interrupt" mode: on `steer()` during streaming, abort
  the in-flight call (per-call controller already exists in
  `callLLMGuarded`), keep the partial assistant text as a truncated message,
  inject the steering message, and continue. That's what makes steering feel
  instant in Claude Code.

## 7. Outer loop / harness

- **Plan mode**: a harness-level mode where the tool registry is swapped to
  read-only (the registry patching logic in `setTools()` already supports
  this) and exit requires explicit user approval. Pairs naturally with the
  permission system (§2).
- **Checkpoints/rewind**: snapshot `history` + a file-system checkpoint
  (`file-mutation-queue.ts` already tracks mutations) before each prompt, so
  the user can rewind a bad turn. Branching covers conversation state but
  not files.
- **Nested branches**: `fork()` is single-level by design; the stack
  structure is already there (`branches: Branch[]`), so allowing nesting is
  mostly lifting the doc constraint and testing summary-merge ordering.
- **Reasoner integration**: the pre-reasoner runs once, blocking, with a
  fixed 60s timeout, and its trace enters as a synthetic assistant message.
  Better: expose reasoners as tools (`think_tot`, `reflect`) the model can
  invoke mid-task, and emit reasoner progress events for the thinking panel.
- **Session resume** *(partially done)*: `src/session-store.ts` now persists
  sessions to per-project SQLite (better-sqlite3, WAL mode) — turns saved on
  `turn_end`, most-recent session auto-resumed on startup, full session
  manager UI (list / filter / rename / delete / new). Remaining gap: resume
  restores the **UI transcript only** — `tui.ts` rebuilds the transcript from
  stored user messages and never repopulates the harness `history`, so the
  model starts cold after a restart and "continue" loses all context. Feed
  the loaded turns back into `AgentHarness` as `Message[]` history on resume
  and on session switch. Also still unpersisted: nextTurn queue, model
  index, thinking level; and the transcript rebuild drops assistant/tool
  content (only `userMessage.content` is re-added).

## 8. Compaction & memory

- **Cache-aware compaction**: compaction rewrites the head of the message
  list, which invalidates any provider prompt cache. Prefer tail-preserving
  strategies (summarize oldest N, keep system + recent K intact) — partially
  done in `microCompactMessages`; make the cache boundary explicit.
- **Tool-result-first compaction**: before summarizing conversation, truncate
  stale tool results (oldest, largest) — they dominate token use and lose
  the least information.
- **Auto-memory**: a `MEMORY.md`-style file injected per session and a
  `remember` tool to append durable facts. There is currently no persistent
  cross-session memory at all.

## 9. Observability

- Emit OpenTelemetry-compatible spans (turn, llm_call, tool_call, hook) or at
  minimum a structured `--debug-events` JSONL dump. `TurnMetrics` is a good
  start; expose it through an event at `agent_end` so the status bar can show
  per-run tokens/time/retries without polling.

## Implementation status (2026-06-10)

| Item | Status | Where |
|---|---|---|
| Harness-history resume (§7) | ✅ done | `harness.setHistory`, `bridge.restoreHistory`, `tui.restoreSession` — resume/switch now restores model context, not just the UI |
| Subagents (§1) | ✅ done | `agent-core/subagent.ts` — `spawn_agent` tool, `.logician/agents/*.md` defs + built-in `general`/`explorer`, `subagent_*` events, progress streamed via tool onUpdate |
| Permission system (§2) | ✅ done | `agent-core/permissions.ts` — modes acceptAll/acceptEdits/ask/plan, `bash(git *)` rules, interactive y/a/n approval in the input bar, PreToolUse hooks block via exit code 2 / `permissionDecision` JSON, `/permissions` command |
| Event envelope + log (§5) | ✅ done | seq/ts stamped at the emit boundary; JSONL event log next to the transcript (`*.events.jsonl`); `thinking_delta` carries turnId |
| Structured stop + budgets (§6) | ✅ done | `task_status` tool (terminates run, beats `declaresStop` regex), `maxTotalTokens` run budget with `budget_exhausted` event, usage in `TurnMetrics` |
| Skill frontmatter + slash (§3) | ✅ done | `allowed-tools` / `argument-hint` / `model` parsed; `/<skill-name> args` invokes skills; skills listed in slash popup |
| Mid-stream steering (§6) | ✅ done | `steeringInterrupt` config — `steer()` aborts the in-flight stream, keeps partial text, injects guidance at the next save point |
| Plan mode + checkpoints (§7) | ✅ done | `/plan` toggle (read-only tool gate via permission mode), per-prompt conversation checkpoints + `/rewind` |
| Tool `readOnly` flags | ✅ done | All read/search/fetch tools marked; drives plan/acceptEdits/explorer-agent tool sets |

### Round 2 (same day)

| Item | Status | Where |
|---|---|---|
| Test suite | ✅ done | `src/agent-core/__tests__/` (node:test via `npm test`, 31 tests): permissions, hook bus + timeouts, file checkpoints, backend classification, harness (history/rewind/queues/branches), loop (interrupt, task_status terminate, permission deny, budget stop, event seq) |
| Hook per-handler timeout | ✅ done | `HookBus` `defaultTimeoutMs` (60s via composeHooks) + per-registration override; timed-out handler skipped + reported |
| Retry-After + jitter | ✅ done | `BackendError.retryAfterMs` parsed from the header (clamped 5 min); retry delay prefers it over backoff, ±20% jitter |
| File checkpoints | ✅ done | `agent-core/file-checkpoints.ts` — pre-write snapshots (write_file/edit_file) per prompt frame; `/rewind` restores conversation AND files; bash mutations not captured |
| Subagent UI rendering | ✅ done | Child tool calls / failures render as `↳ agent_N` notice lines in the parent transcript |

Still open: catalog ranking (§3), PreCompact/PostCompact firing (§4),
event-log replay on startup (§5), tool result content blocks (§6), nested
branches (§7), bash-mutation capture for checkpoints (§7), auto-memory (§8),
OTel spans (§9).
