# Hook Architecture

Logician's hook system is an interception layer between the agent loop and the LLM provider.
Four intentionally separate systems handle different concerns, composed into a single
`AgentHooks` interface the runner consumes.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         LOGICIAN HOOK ARCHITECTURE                                  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                    AGENT HOOK LAYERS (ordered composition)                     │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  LAYER 1: BUILTIN                                                       │  │  │
│  │  │  File: hooks/builtin/builtin-hooks.ts                                   │  │  │
│  │  │  Purpose: Internal runtime safeguard policies                           │  │  │
│  │  │  Events:                                                                │  │  │
│  │  │    • beforeToolCall   → pre-bash/file snapshots, guard-engine checks    │  │  │
│  │  │    • afterToolCall    → bash mutation recording, task_status terminate  │  │  │
│  │  │    • prepareNextTurn  → proactive compaction (80% window trigger)       │  │  │
│  │  │    • shouldStopAfterTurn → budget stop, thinking-loop detection         │  │  │
│  │  │    • afterProviderResponse → thinking-loop turn recording               │  │  │
│  │  │    • getFollowUpMessages → continuation nudges (todo circling, cutoffs) │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  LAYER 2: EXTENSIONS (TypeScript)                                       │  │  │
│  │  │  Files: hooks/extensions/events.ts, event-bus.ts, context.ts            │  │  │
│  │  │  Purpose: Lifecycle events for TypeScript extensions                    │  │  │
│  │  │  Events:                                                                │  │  │
│  │  │    • before_agent_start, turn_start, turn_end                           │  │  │
│  │  │    • message_start, message_update, message_end                         │  │  │
│  │  │    • tool_execution_start, tool_execution_update, tool_execution_end    │  │  │
│  │  │    • context_update, session_before_compact, session_compact            │  │  │
│  │  │    • before_provider_request, after_provider_response                   │  │  │
│  │  │    • agent_end, session_shutdown                                        │  │  │
│  │  │  Mechanism: ExtensionEventBus (typed, per-handler timeout, error isolate)│  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  LAYER 3: CLAUDE CODE COMPATIBILITY                                     │  │  │
│  │  │  Files: compatibility/claude-code/hook-layer.ts, plugin-executor.ts     │  │  │
│  │  │  Purpose: Bridge Claude Code plugin manifests → native AgentHooks       │  │  │
│  │  │  Events:                                                                │  │  │
│  │  │    • PreToolUse     → Claude stdin JSON (permission grant/deny)         │  │  │
│  │  │    • PostToolUse    → Claude stdin JSON (context injection)             │  │  │
│  │  │    • PostToolUseFailure → Claude stdin JSON (error context)             │  │  │
│  │  │    • UserPromptSubmit → Claude stdin JSON (prompt context)              │  │  │
│  │  │    • Stop           → Claude stdin JSON (stop continuation)             │  │  │
│  │  │  Mechanism: RuntimeClaudeCodeHookLayer — runs shell/HTTP/prompt cmds    │  │  │
│  │  │              via runHookEvent(), maps Claude event names to tool names   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  LAYER 4: USER                                                          │  │  │
│  │  │  Source: AgentConfig.hooks (directly in config)                         │  │  │
│  │  │  Purpose: Custom user-defined hook handlers                             │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                              │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    COMPOSITION via composeHooks()                       │  │  │
│  │  │  layers: [builtin → extensions → claude-code-compat → user]            │  │  │
│  │  │  Each layer's AgentHooks registered on one HookBus                      │  │  │
│  │  │  HookBus.toHooks() → single AgentHooks object                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                        HOOK BUS (native/hook-bus.ts)                          │  │
│  │                                                                               │  │
│  │  • Typed multi-handler bus — multiple handlers per event                      │  │
│  │  • Deterministic ordering: higher priority first, equal priority = reg order  │  │
│  │  • Per-event reducer semantics:                                               │  │  │
│  │      beforeToolCall   → early-block: first {content} short-circuits           │  │  │
│  │                         {args} rewrites thread to later handlers              │  │  │
│  │      afterToolCall    → patch-accumulate: each handler sees prior patch       │  │  │
│  │                         later non-undefined fields win                        │  │  │
│  │      prepareNextTurn  → transform: messages thread through all handlers       │  │  │
│  │      shouldStopAfterTurn → first-true wins                                    │  │  │
│  │      getSteeringMessages → append: results concatenated                       │  │  │
│  │      getFollowUpMessages → append: results concatenated                       │  │  │
│  │      beforeProviderRequest → merge: headers/metadata collected from all       │  │  │
│  │      beforeProviderPayload → transform: payload threaded through handlers     │  │  │
│  │      beforeCompact → cancel-early: first {cancel: true} stops compaction      │  │  │
│  │      afterProviderResponse → fire-and-forget: no result aggregation           │  │  │
│  │  • Per-handler timeouts (default 60s), error isolation, metrics               │  │  │
│  │  • AbortSignal per handler, parent cancellation linkage                       │  │  │
│  │  • Read-only observer firehose via observe()                                  │  │  │
│  │  • Duplicate handler ID rejection, LIFO cleanup                               │  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                    AGENT LOOP RUNNER integration                              │  │
│  │  File: core/agent-loop-runner.ts                                              │  │
│  │                                                                               │  │
│  │  The runner calls one composed AgentHooks handler per event:                  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  TURN FLOW                                                              │  │  │
│  │  │                                                                         │  │  │
│  │  │  1. beforeAgentStart(prompt, systemPrompt, messages)                    │  │  │
│  │  │     → Modify prompt, system prompt, or prepend messages                 │  │  │
│  │  │     → Called once per user turn                                         │  │  │
│  │  │                                                                         │  │  │
│  │  │  2. [Agent loop: while stop_reason == "tool_use"]                       │  │  │
│  │  │                                                                         │  │  │
│  │  │    2a. beforeProviderRequest(model, sessionId, streamOptions)           │  │  │
│  │  │       → Patch headers, timeout, retries, cache retention, metadata      │  │  │
│  │  │       → Runs BEFORE every LLM API request                               │  │  │
│  │  │                                                                         │  │  │
│  │  │    2b. beforeProviderPayload(model, payload)                            │  │  │
│  │  │       → Transform the full API request payload                          │  │  │
│  │  │       → Runs BEFORE serialization to provider                           │  │  │
│  │  │                                                                         │  │  │
│  │  │    2c. [LLM API call — streaming or non-streaming]                      │  │  │
│  │  │                                                                         │  │  │
│  │  │    2d. afterProviderResponse(model, content, toolCallCount,             │  │  │
│  │  │              stopReason, usageTokens, iteration)                        │  │  │
│  │  │       → Side effects: thinking-loop detection, extension events         │  │  │
│  │  │       → No result aggregation (fire-and-forget)                         │  │  │
│  │  │                                                                         │  │  │
│  │  │    2e. For each tool call in the response:                              │  │  │
│  │  │                                                                         │  │  │
│  │  │       i.   beforeToolCall(toolCall, args)                               │  │  │
│  │  │          → Guard: block duplicate/failure-loop calls                    │  │  │
│  │  │          → Claude Code: PreToolUse stdin → permission grant/deny        │  ��  │
│  │  │          → Rewrite args, or return {content} to short-circuit           │  ��  │
│  │  │                                                                         │  │  │
│  │  │       ii.  [Tool execution — bash/read/write/edit/grep/find/etc.]       │  │  │
│  │  │                                                                         │  │  │
│  │  │       iii. afterToolCall(toolCall, args, result, isError)               │  │  │
│  │  │          → Claude Code: PostToolUse stdin → context injection           │  │  │
│  │  │          → Record bash mutations, task_status terminate                 │  │  │
│  │  │          → Append <post-tool-use-hook> to result                        │  │  │
│  │  │          → Return {terminate: true} to end turn                         │  │  │
│  │  │                                                                         │  │  │
│  │  │  2f. prepareNextTurn(messages, iteration, hadToolCalls)                 │  │  │
│  │  │       → Proactive compaction at 80% context window                      │  │  │
│  │  │       → Transform messages before next iteration                        │  │  │
│  │  │                                                                         │  │  │
│  │  │  2g. shouldStopAfterTurn(messages, iteration, hadToolCalls)             │  │  │
│  │  │       → Budget stop, thinking-loop stop                                 │  │  │
│  │  │       → First handler returning true ends the loop                      │  │  │
│  │  │                                                                         │  │  │
│  │  │  2h. getSteeringMessages(messages, iteration)                           │  │  │
│  │  │       → Inject steering/context messages                                │  │  │
│  │  │       → Appended to messages for next iteration                         │  │  │
│  │  │                                                                         │  │  │
│  │  │  2i. getFollowUpMessages(assistantText, stopReason, messages)           │  │  │
│  │  │       → Continuation nudges (cutoff, circling, todo)                    │  │  │
│  │  │       → Appended as user messages for next iteration                    │  │  │
│  │  │                                                                         │  │  │
│  │  │  3. [Loop exits: stop_reason != "tool_use" OR shouldStop=true]          │  │  │
│  │  │                                                                         │  │  │
│  │  │  4. beforeCompact(messages, tokensBefore, reason)                       │  │  │
│  │  │       → Cancel compaction, or provide pre-built summary                 │  │  │
│  │  │       → Called before compaction runs                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │              CLAUDE CODE COMPAT LAYER — DETAILED FLOW                         │  │
│  │  File: compatibility/claude-code/hook-layer.ts                                │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  PLUGIN LOADING (startup)                                               │  │  │
│  │  │                                                                         │  │  │
│  │  │  .claude/plugins/<plugin>/                                              │  │  │
│  │  │  ├── .claude-plugin/plugin.json           → manifest (hooks path)      │  │  │
│  │  │  ├── hooks/hooks.json                     → event → HookDefinition[]   │  │  │
│  │  │  └── (shell/HTTP/prompt commands)        → executed on event           │  │  │
│  │  │                                                                         │  │  │
│  │  │  loadPluginHooks() → parse manifest → mergeManifestHooks()              │  │  │
│  │  │  → LoadedHook[] with eventType + HookDefinition[]                       │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  EVENT MAPPING (Logician → Claude)                                      │  │  │
│  │  │                                                                         │  │  │
│  │  │  Logician Hook Event       → Claude Event Type                          │  │  │
│  │  │  ──────────────────────────  ─────────────────────────────────────────── │  │  │
│  │  │  beforeToolCall            → PreToolUse (stdin JSON)                     │  │  │
│  │  │  afterToolCall             → PostToolUse (stdin JSON)                    │  │  │
│  │  │  afterToolCall (error)     → PostToolUseFailure (stdin JSON)             │  │  │
│  │  │  beforeAgentStart          → UserPromptSubmit (stdin JSON)               │  │  │
│  │  │  shouldStopAfterTurn       → Stop (stdin JSON)                           │  │  │
│  │  │  beforeCompact             → PreCompact (stdin JSON)                     │  │  │
│  │  │  session lifecycle         → SessionStart / SessionEnd                  │  │  │
│  │  │                                                                         │  │  │
│  │  │  Tool name → Claude matcher:                                            │  │  │
│  │  │    bash → "Bash", read_file → "Read", write_file → "Write",             │  │  │
│  │  │    edit_file → "Edit", grep → "Grep", find/list_files → "Glob",         │  │  │
│  │  │    todo → "TodoWrite|TaskCreate|TaskUpdate", spawn_agent → "Agent",      │  │  │
│  │  │    ask_user → "AskUserQuestion", mcp__* → passthrough                   │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  COMMAND TYPES (hook definition)                                        │  │  │
│  │  │                                                                         │  │  │
│  │  │  command  → spawn shell with stdin JSON payload, parse stdout JSON      │  │  │
│  │  │  prompt   → inject raw prompt text as additional_context                 │  │  │
│  │  │  http     → POST to URL with optional headers, parse response JSON       │  │  │
│  │  │  agent    → (reserved, not executed by executor)                        │  │  │
│  │  │                                                                         │  │  │
│  │  │  Exit code 2 = blocking error; stderr → reason fed back to model        │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  PRE-TOOL-USE FLOW                                                      │  │  │
│  │  │                                                                         │  │  │
│  │  │  beforeToolCall({ toolCall, args })                                     │  │  │
│  │  │    │                                                                    │  │  │
│  │  │  ├─► runHookEvent("PreToolUse", { tool_name, tool_input,                │  │  │
│  │  │  │                           matcher_value })                           │  │  │
│  │  │  │    │                                                                 │  │  │
│  │  │  │    ├─► shell: echo <JSON> | <command>                                │  │  │
│  │  │  │    ├─► http: POST <url> with JSON body                               │  │  │
│  │  │  │    └─► prompt: raw text as context                                   │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  │    ← parseHookResponse(stdout/stderr)                                │  │  │
│  │  │  │       → { permission_decision: "allow"|"deny"|"ask",                  │  │  │
│  │  │  │         permission_reason, additional_contexts }                      │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  ├─► Store context in preToolContext map (toolCall.id → context)        │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  ├─► If permission_decision === "deny"                                  │  │  │
│  │  │  │   └─► return { content: "Permission denied...", isError: true }      │  │  │
│  │  │  │      (tool NOT executed)                                             │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  └─► Return undefined → tool executes normally                          │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  POST-TOOL-USE FLOW                                                     │  │  │
│  │  │                                                                         │  │  │
│  │  │  afterToolCall({ toolCall, args, result, isError })                     │  │  │
│  │  │    │                                                                    │  │  │
│  │  │  ├─► Retrieve preToolContext[toolCall.id] (stored in PreToolUse)        │  │  │
│  │  │  │    Delete from map                                                   │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  ├─► If isError: runHookEvent("PostToolUseFailure", ...)                │  │  │
│  │  │  └─► Else: runHookEvent("PostToolUse", ...)                             │  │  │
│  │  │         Payload: { tool_name, tool_input, tool_response/tool_error,     │  │  │
│  │  │                        matcher_value }                                   │  │  │
│  │  │                                                                       │  │  │
│  │  │  ← Parse response → { additional_contexts }                             │  │  │
│  │  │                                                                       │  │  │
│  │  │  → Return { content: `${result}\n\n<post-tool-use-hook>\n${context}\n</post-tool-use-hook>` }
│  │  │     (context injected into tool result visible to model)                │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  USER PROMPT SUBMIT FLOW                                                │  │  │
│  │  │                                                                         │  │  │
│  │  │  userPromptMessages(prompt)                                             │  │  │
│  │  │    │                                                                    │  │  │
│  │  │  ├─► runHookEvent("UserPromptSubmit", { prompt, timeout_seconds: 30 })  │  │  │
│  │  │  │    ← { additional_contexts }                                         │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  ├─► Wrap in user message:                                               │  │  │
│  │  │  │   <user-prompt-submit-hook>                                          │  │  │
│  │  │  │   ${context}                                                         │  │  │
│  │  │  │   </user-prompt-submit-hook>                                         │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  └─► Prepend to initial messages before agent start                     │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │  STOP / CONTINUATION FLOW                                               │  │  │
│  │  │                                                                         │  │  │
│  │  │  finalStop() — called when agent decides it's done                      │  │  │
│  │  │    │                                                                    │  │  │
│  │  │  ├─► runHookEvent("Stop", { stop_hook_active: false })                  │  │  │
│  │  │  │    ← { additional_contexts, reason }                                 │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  ├─► If blocked (has context/reason):                                   │  │  │
│  │  │  │   set stopHookContinuationActive = true                              │  │  │
│  │  │  │   Inject <stop-hook> into follow-up messages                         │  │  │
│  │  │  │   Model continues with hook guidance                                 │  │  │
│  │  │  │                                                                       │  │  │
│  │  │  └─► If not blocked: clear continuation flag                            │  │  │
│  │  │                                                                         │  │  │
│  │  │  getFollowUpMessages() — called when agent stops naturally              │  │  │
│  │  │    → Sets stopObserved = true                                            │  │  │
│  │  │    → Runs Stop hook with stop_hook_active flag                          │  │  │
│  │  │    → Returns follow-up messages if blocked, empty if not                │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                   SESSION LIFECYCLE HOOKS (non-agent-loop)                    │  │
│  │  File: core/harness/session-lifecycle.ts                                      │  │
│  │                                                                               │  │
│  │  SessionStart  → runSessionStartHooks() (CLI: /plugins session-start)         │  │
│  │    • Loaded from plugin manifests at startup                                  │  │
│  │    • Source: "startup" | "resume"                                             │  │
│  │    • Payload: session_id, transcript_path, cwd                                │  │
│  │    • Can set initial_user_message, watch_paths                                │  │
│  │                                                                               │  │
│  │  SessionEnd    → runHookEvent("SessionEnd", { reason })                       │  │
│  │    • Called on harness shutdown                                               │  │
│  │    • Must not block cleanup (errors swallowed)                                │  │
│  │                                                                               │  │
│  │  PreCompact    → internalHook → userHook → runHookEvent("PreCompact")         │  │
│  │    • Called before compaction runs                                            │  │
│  │    • Internal hook can cancel or supply summary                               │  │
│  │    • Plugin hook runs after internal hooks                                    │  │
│  │                                                                               │  │
│  │  PostCompact   → runHookEvent("PostCompact")                                  │  │
│  │    • Called after compaction completes                                        │  │
│  │    • Fire-and-forget (errors swallowed)                                       │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                         HOOK TYPES SUMMARY                                    │  │
│  │                                                                               │  │
│  │  Event                    │ Pre / Post │ Reducer        │ Result Effect       │  │
│  │  ─────────────────────────┼────────────┼────────────────┼───────────────────── │  │
│  │  beforeToolCall           │ Pre        │ early-block    │ Short-circuit or    │  │
│  │                            │            │                │ rewrite args         │  │
│  │  afterToolCall            │ Post       │ patch-accumulate│ Modify result,      │  │
│  │                            │            │                │ inject context,       │  │
│  │                            │            │                │ terminate turn        │  │
│  │  prepareNextTurn          │ Post       │ transform      │ Rewrite messages     │  │
│  │  beforeProviderRequest    │ Pre        │ merge          │ Collect headers,     │  │
│  │                            │            │                │ timeout, retries      │  │
│  │  beforeProviderPayload    │ Pre        │ transform      │ Rewrite API payload  │  │
│  │  afterProviderResponse    │ Post       │ fire-and-forget│ Side effects only    │  │
│  │  shouldStopAfterTurn      │ Post       │ first-true     │ End agent loop       │  │
│  │  getSteeringMessages      │ Post       │ append         │ Add steering msgs    │  │
│  │  getFollowUpMessages      │ Post       │ append         │ Add follow-up msgs   │  │
│  │  beforeCompact            │ Pre        │ cancel-early   │ Skip or supply       │  │
│  │                            │            │                │ summary               │  │
│  │  beforeAgentStart         │ Pre        │ transform      │ Modify prompt,       │  │
│  │                            │            │                │ system prompt, msgs  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                     ERROR & LIFECYCLE MODEL                                   │  │
│  │                                                                               │  │
│  │  • errorMode: "continue" (default) → thrown handler skipped, reported         │  │
│  │  • errorMode: "throw" → thrown handler aborts entire chain                    │  │
│  │  • Per-handler timeout → AbortController fires, handler skipped               │  │
│  │  • Parent AbortSignal → linked to handler-scoped signal                       │  │
│  │  • Duplicate ID → rejected at registration                                    │  │
│  │  • dispose() → runs LIFO cleanups, rejects new registrations                  │  │
│  │  • Session lifecycle hooks → errors swallowed (must not break turns)          │  │
│  │  • Extension events → errors isolated, reported to onError                    │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## File Map

| Component | File | Role |
|-----------|------|------|
| **Hook Bus** | `agent-core/src/hooks/native/hook-bus.ts` | Multi-handler composition engine |
| **Hook Types** | `agent-core/src/core/types/types-hooks.ts` | `AgentHooks` interface + context/result types |
| **Builtin Hooks** | `agent-core/src/hooks/builtin/builtin-hooks.ts` | Compaction, budget, guard, thinking-loop, continuation |
| **Extension Events** | `agent-core/src/hooks/extensions/events.ts` | Typed event definitions (19 event types) |
| **Extension EventBus** | `agent-core/src/hooks/extensions/event-bus.ts` | Typed event emitter for extensions |
| **Extension Context** | `agent-core/src/hooks/extensions/context.ts` | Per-session mutable state for extensions |
| **Claude Code Layer** | `agent-core/src/compatibility/claude-code/hook-layer.ts` | Plugin manifest → `AgentHooks` adapter |
| **Plugin Executor** | `agent-core/src/compatibility/claude-code/plugin-executor.ts` | Shell/HTTP/prompt command execution |
| **Plugin Manager** | `agent-core/src/compatibility/claude-code/plugin-manager.ts` | Plugin install/enable/disable/lifecycle |
| **Session Lifecycle** | `agent-core/src/core/harness/session-lifecycle.ts` | SessionStart/End/Compact hooks |
| **Harness Queue Hooks** | `agent-core/src/runtime/harness-queue-hooks.ts` | Steering/follow-up from message queue |
| **Plugins Barrel** | `agent-core/src/tools/shared/plugins.ts` | CLI entry points + re-exports |
| **Agent Harness** | `agent-core/src/core/harness.ts` | Composes all layers, creates turn snapshots |
| **Agent Loop Runner** | `agent-core/src/core/agent-loop-runner.ts` | Calls composed `AgentHooks` per event |

## Key Design Decisions

1. **Deterministic composition** — handlers sorted by priority (desc), then registration order. Higher priority runs first.
2. **Error isolation** — thrown handlers are skipped (not propagated) in default mode. Each handler has its own `AbortSignal`.
3. **Reducer semantics per event** — each event type has a specific composition strategy (early-block, patch-accumulate, transform, merge, first-true, append).
4. **Layer ordering** — builtin → extensions → claude-code-compat → user. Earlier layers see raw data; later layers see transformed data.
5. **Claude Code boundary** — plugin protocol details (stdin JSON, event names, matcher values) never leak outside `compatibility/claude-code/`.
6. **Session hooks** — run outside the agent loop at session boundaries; errors are swallowed to avoid breaking turns.
7. **Extension events** — separate typed event bus from the hook bus. Extensions subscribe to lifecycle events, not hook reducers.
