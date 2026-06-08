# Agent Loop / Flow Improvements: Logician TUI vs Pi

Comparison of the TypeScript agent loop in `tui` against `repos/pi/packages/agent`.
Actionable improvements ranked by impact.

---

## 1. Session Persistence (High Impact)

**Pi has it.** Full `Session` abstraction with JSONL storage, branch summaries, compaction entries, and model/thinking-level change history. Sessions survive restarts.

**tui has nothing.** Messages live only in memory. A crash or reload loses everything.

**What to build:**
- `Session` class wrapping a JSONL file per conversation
- `SessionRepo` interface (memory + file implementations)
- `appendMessage`, `buildContext`, `getBranch`, `getLeafId`, `setLeafId`
- Persist model changes, thinking-level changes, active-tools changes
- Auto-load last session on startup (or prompt user)

**Pi reference:** `harness/session/*.ts` — `session.ts`, `jsonl-storage.ts`, `jsonl-repo.ts`, `repo-utils.ts`

---

## 2. Skills System (Medium-High Impact)

**Pi has it.** `loadSkills()` walks directories for `SKILL.md` files with YAML frontmatter (`name`, `description`, `disable-model-invocation`). Skills get injected into the system prompt as `<skill>` blocks. Supports sourced skills with provenance.

**tui has nothing.** No skill loading, no skill injection.

**What to build:**
- `loadSkills(env, dirs)` — recursive SKILL.md discovery with ignore-file support
- Frontmatter parsing (yaml)
- Skill validation (name matches dir, description required, length limits)
- `formatSkillInvocation(skill)` → `<skill name="..." location="...">...</skill>`
- Inject loaded skills into system prompt before model call
- Support `disable-model-invocation` flag

**Pi reference:** `harness/skills.ts`

---

## 3. Prompt Templates (Medium Impact)

**Pi has it.** `promptTemplates` resource array. `promptFromTemplate(name, args)` invokes a template with string interpolation. Templates are loaded alongside skills.

**tui has nothing.**

**What to build:**
- `PromptTemplate` interface: `{ name, template, description }`
- Load from `.md` files alongside skills
- `promptFromTemplate(name, args: string[])` — positional arg injection
- Use in harness for predefined actions (e.g., "review this PR", "write tests")

**Pi reference:** `harness/prompt-templates.ts`

---

## 4. Context Transform Hook (Medium Impact)

**Pi has it.** `transformContext: (messages, signal) => Promise<AgentMessage[]>` runs before `convertToLlm`. Enables pre-processing at the AgentMessage level: pruning old messages, injecting external context, filtering UI-only messages.

**tui has no equivalent.** Context compaction happens inline in the loop or via `prepareNextTurn`, which is post-turn.

**What to build:**
- Add `transformContext` to `AgentLoopConfig`
- Runs before LLM call, after steering injection, before `convertToLlm`
- Enables proactive context pruning without waiting for `prepareNextTurn`
- Can be used by hooks for cross-cutting context manipulation

**Pi reference:** `types.ts` — `AgentLoopConfig.transformContext`

---

## 5. Dynamic API Key Resolution (Medium Impact)

**Pi has it.** `getApiKey(provider) => Promise<string | undefined>` resolves API keys per-request. Critical for short-lived OAuth tokens (GitHub Copilot) that expire mid-run.

**tui has nothing.** API key is static in config/backend.

**What to build:**
- Add `getApiKey` to `AgentLoopConfig`
- Call before each LLM request in the backend
- Merge with existing headers
- Graceful fallback to config key

**Pi reference:** `types.ts` — `AgentLoopConfig.getApiKey`; `agent-harness.ts` — `getApiKeyAndHeaders`

---

## 6. Granular Streaming Events (Medium Impact)

**Pi has it.** Distinct events: `text_start`, `text_delta`, `text_end`, `thinking_start`, `thinking_delta`, `thinking_end`, `toolcall_start`, `toolcall_delta`, `toolcall_end`, plus `message_update` with the full partial message.

**tui has basic deltas.** `message_delta` and `thinking_delta` — no text/tool boundaries, no partial message snapshots.

**What to build:**
- Emit `thinking_start` / `thinking_end` markers (already partially done)
- Emit `text_start` / `text_end` for text content boundaries
- Emit `toolcall_start` / `toolcall_end` per tool call in the response
- `message_update` event with full partial assistant message for UI updates
- Enables richer TUI rendering (thinking indicator, per-tool-call status)

**Pi reference:** `agent-loop.ts` — `streamAssistantResponse` event switch

---

## 7. AgentMessage Abstraction (Medium Impact)

**Pi has it.** `AgentMessage` is a union of LLM messages + custom app messages (notifications, status updates, UI-only artifacts). `convertToLlm` filters/transforms to LLM-compatible format. Apps extend via declaration merging.

**tui has plain `Message[]`.** No room for non-LLM messages. All messages go to the model.

**What to build:**
- `AgentMessage` union type extending `Message` with custom roles
- `convertToLlm` config option to filter non-LLM messages
- Declaration merging for app-specific message types
- Enables status messages, notifications, artifacts without polluting context

**Pi reference:** `types.ts` — `AgentMessage`, `CustomAgentMessages`, `convertToLlm`

---

## 8. Compaction Improvements (Medium Impact)

**Pi has it.** Multi-pass compaction:
- `microCompactMessages` — trim oversized bodies locally (no LLM call)
- `compact` — LLM-generated summary of older messages
- `branchSummarization` — summarize a branch when navigating to a different point in history
- Configurable `DEFAULT_COMPACTION_SETTINGS`

**tui has basic compaction.** `compactMessagesForContext` (LLM-based) + `microCompactMessages` exists but is not well integrated. No branch summarization.

**What to build:**
- Integrate micro-compaction as first pass before LLM compaction
- Add branch summarization for session navigation
- Expose compaction config (`targetTokens`, `keepRecentMessages`)
- LLM summary fallback to local truncation on failure (already partially done)

**Pi reference:** `harness/compaction/` — `compaction.ts`, `branch-summarization.ts`, `utils.ts`

---

## 9. Event System Overhaul (Low-Medium Impact)

**Pi has it.** Structured event types with rich payloads:
- `tool_execution_start/update/end` with full result objects
- `message_update` with partial message snapshots
- `turn_end` with `message` and `toolResults` arrays
- `agent_end` carries final `messages` array

**tui has flat events.** `tool_call_start/end/update` with string results. `message_delta` instead of partial message. `turn_end` only has `turnId`.

**What to build:**
- Enrich `tool_call_end` with structured result (content, details, isError)
- Add `message_update` event with partial assistant message
- `turn_end` should carry message + tool results
- `agent_end` should carry final messages array
- Align event shapes with Pi for easier cross-repo tooling

**Pi reference:** `types.ts` — `AgentEvent` union

---

## 10. Queue Mode Defaults (Low Impact)

**Pi defaults to "one-at-a-time".** Steering and follow-up messages drain one at a time, giving the user fine-grained control over message injection cadence.

**tui defaults to "all".** Drains everything at once. Can overwhelm the model with queued messages.

**What to build:**
- Change defaults to "one-at-a-time" for both steering and follow-up
- Keep "all" as an option for batch operations
- Document the UX difference: "one-at-a-time" = user can interrupt between injected messages

**Pi reference:** `agent-harness.ts` — constructor defaults

---

## 11. Error Encoding vs Throwing (Low-Medium Impact)

**Pi encodes errors as messages.** When the LLM call fails, it produces an `AssistantMessage` with `stopReason: "error"` and `errorMessage`. The loop continues normally, the error is visible in the transcript, and the user can respond.

**tui throws.** Errors bubble up, caught by auto-retry or emitted as `error` events. No error message in the transcript.

**What to build:**
- On LLM failure, append an assistant message with the error text
- Set `stopReason: "error"` equivalent in the message
- Emit `error` event but don't break the turn flow
- User sees the error in the transcript and can respond ("try again", "use a different model")

**Pi reference:** `agent-loop.ts` — `streamAssistantResponse` error case

---

## 12. Tool Result Details (Low Impact)

**Pi has it.** `AgentToolResult` has `content: (TextContent | ImageContent)[]` and `details: T`. Tools can return structured metadata alongside text.

**tui has plain strings.** Tool `execute` returns `Promise<string>`. No structured details.

**What to build:**
- Change `Tool.execute` to return `{ content: string, details?: Record<string, unknown> }`
- Pass details through `tool_call_end` events
- Enables tools to return file diffs, line counts, structured data for UI

**Pi reference:** `types.ts` — `AgentToolResult`, `AgentTool.execute`

---

## 13. `before_provider_request` / `before_provider_payload` Hooks (Low Impact)

**Pi has it.** `before_provider_request` fires before each LLM call with the model, session ID, and stream options. Returns a patch for headers, timeout, retries, cache retention. `before_provider_payload` fires with the raw request payload for inspection/modification.

**tui has nothing.** No hook into the provider call boundary.

**What to build:**
- `beforeProviderRequest` hook: receive model, sessionId, streamOptions; return patch
- `beforeProviderPayload` hook: receive raw JSON payload; return modified payload
- Enables per-request header injection, timeout tuning, payload inspection
- Useful for analytics, A/B testing, request logging

**Pi reference:** `agent-harness.ts` — `emitBeforeProviderRequest`, `emitBeforeProviderPayload`

---

## 14. Harness Phase Tracking (Low Impact)

**Pi has it.** Explicit phases: `"idle"`, `"turn"`, `"compaction"`, `"branch_summary"`. Operations are rejected when the harness is in the wrong phase (e.g., `steer()` throws while idle).

**tui has basic phases.** `"idle"` and `"turn"` in the harness. No compaction or branch_summary phases.

**What to build:**
- Add `"compaction"` and `"branch_summary"` phases
- Phase-gate operations (compact requires idle, steer requires turn)
- `HarnessBusyError` with operation name (already exists in tui)

**Pi reference:** `agent-harness.ts` — `AgentHarnessPhase`, phase checks

---

## Summary Matrix

| Feature | Pi | tui | Priority |
|---------|----|--------------|----------|
| Session persistence | ✅ JSONL + branch | ❌ memory only | **High** |
| Skills system | ✅ SKILL.md loading | ✅ done (load + validate + catalog injection + read_skill on-demand tool + disable-model-invocation) | **High** |
| Prompt templates | ✅ template invocation | ❌ none | **Medium** |
| Context transform hook | ✅ pre-convert | ✅ done (transformContext) | **Medium** |
| Dynamic API keys | ✅ per-request | ⚠️ via beforeProviderRequest headers | **Medium** |
| Granular streaming | ✅ text/tool/thinking events | ✅ done (text/tool boundaries + message_update) | **Medium** |
| AgentMessage abstraction | ✅ custom messages | ✅ done (AgentMessage union + convertToLlm) | **Medium** |
| Compaction improvements | ✅ micro + branch | ⚠️ micro+LLM done; no branch summary | **Medium** |
| Event system overhaul | ✅ rich payloads | ✅ done (stopReason, turn_end msg/toolResults, agent_end messages) | **Low-Med** |
| Queue mode defaults | ✅ one-at-a-time | ✅ done (one-at-a-time) | **Low** |
| Error encoding | ✅ in transcript | ✅ done (assistant error msg + stopReason='error') | **Low-Med** |
| Tool result details | ✅ structured | ✅ done (string \| ToolResult union → tool_call_end.details) | **Low** |
| Provider hooks | ✅ before_request/payload | ✅ done (beforeProviderRequest + beforeProviderPayload) | **Low** |
| Phase tracking | ✅ 4 phases | ✅ done (4 phases + gating: compact↔idle, steer↔turn) | **Low** |
