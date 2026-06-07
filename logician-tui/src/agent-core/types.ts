// ── Core types ───────────────────────────────────────────────────────────────────
// Mirrors Python AgentEvent/AgentLoop shapes for clean TUI integration.

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface Message {
    role: MessageRole;
    content: string | null;
    tool_call_id?: string;
    tool_calls?: ToolCall[];
    name?: string;
    timestamp?: number;
}

export type AgentEvent =
    | { type: "agent_start" }
    | { type: "agent_end" }
    | { type: "turn_start"; turnId: string }
    | { type: "turn_end"; turnId: string }
    | { type: "message_start"; turnId: string; role: MessageRole }
    | { type: "message_delta"; turnId: string; delta: string }
    | { type: "message_end"; turnId: string }
    | { type: "thinking_start" }
    | { type: "thinking_delta"; delta: string }
    | { type: "thinking_end" }
    | {
          type: "context_update";
          tokens: number;
          maxTokens?: number;
          compacted?: boolean;
      }
    | {
          type: "compaction";
          reason: "context_full" | "manual";
          tokensBefore: number;
          tokensAfter: number;
      }
    | {
          type: "tool_call_start";
          toolName: string;
          toolCallId: string;
          args: string;
      }
    | {
          type: "tool_call_end";
          toolName: string;
          toolCallId: string;
          result: string;
          isError?: boolean;
      }
    | {
          type: "tool_call_update";
          toolName: string;
          toolCallId: string;
          partialResult: string;
      }
    | {
          type: "repair_nudge";
          turnId?: string;
          repairStage: string;
          toolName?: string;
          message: string;
      }
    | { type: "phase"; phase: "streaming" | "thinking" | "tool" | "idle" }
    | {
          type: "auto_retry_start";
          attempt: number;
          maxRetries: number;
          delayMs: number;
          error: string;
      }
    | { type: "auto_retry_end"; attempt: number; success: boolean }
    | { type: "model_select"; model: string; index: number }
    | { type: "error"; message: string; error?: unknown };

export type EventHandler = (event: AgentEvent) => void;

// ── Agent-loop contract hooks ──────────────────────────────────────────────
// First-class extension points mirroring Pi's richer loop contract. Each is an
// optional async callback on AgentConfig. Returning undefined = no change.

export interface BeforeToolCallContext {
    toolCall: ToolCall;
    args: Record<string, unknown>;
    iteration: number;
}

// Return `{ content }` to short-circuit execution (tool is NOT run; content is
// used as the result). Return `{ args }` to rewrite the tool input before it
// runs. Return both to short-circuit with a rewritten record (content wins).
export interface BeforeToolCallResult {
    content?: string;
    isError?: boolean;
    args?: Record<string, unknown>;
}

export interface AfterToolCallContext {
    toolCall: ToolCall;
    args: Record<string, unknown>;
    result: string;
    isError: boolean;
    iteration: number;
}

// Return `{ content }` and/or `{ isError }` to rewrite the recorded tool result.
// Return `{ terminate: true }` to signal the loop to stop after the current
// tool batch (only effective when ALL tools in the batch set terminate=true).
export interface AfterToolCallResult {
    content?: string;
    isError?: boolean;
    terminate?: boolean;
}

export interface PrepareNextTurnContext {
    messages: Message[];
    iteration: number;
    hadToolCalls: boolean;
    continuationCount: number;
    isContinuation: boolean;
}

// Return rewritten messages to replace the working history before the next
// model call (compaction, steering injection, message rewriting).
export interface PrepareNextTurnResult {
    messages: Message[];
}

export interface ShouldStopAfterTurnContext {
    messages: Message[];
    iteration: number;
    hadToolCalls: boolean;
    continuationCount: number;
    isContinuation: boolean;
}

export interface GetSteeringMessagesContext {
    messages: Message[];
    iteration: number;
}

export interface GetFollowUpMessagesContext {
    messages: Message[];
    iteration: number;
    assistantText: string;
    continuationCount: number;
    maxContinuations: number;
}

export type ToolExecutionMode = "sequential" | "parallel";

/**
 * Controls how many queued user messages are injected when the loop reaches
 * a queue drain point.
 *
 * - "all": drain and inject every queued message at that point.
 * - "one-at-a-time": drain and inject only the oldest queued message, leaving
 *   the rest queued for later drain points.
 */
export type QueueMode = "all" | "one-at-a-time";

/**
 * Thinking/reasoning level for models that support it.
 * "off" = no reasoning. All other levels pass reasoning tokens to the provider.
 */
export type ThinkingLevel =
    | "off"
    | "minimal"
    | "low"
    | "medium"
    | "high"
    | "xhigh";

export interface AgentLoopHooks {
    beforeToolCall?: (
        ctx: BeforeToolCallContext,
    ) => Promise<BeforeToolCallResult | undefined> | BeforeToolCallResult | undefined;
    afterToolCall?: (
        ctx: AfterToolCallContext,
    ) => Promise<AfterToolCallResult | undefined> | AfterToolCallResult | undefined;
    prepareNextTurn?: (
        ctx: PrepareNextTurnContext,
    ) =>
        | Promise<PrepareNextTurnResult | undefined>
        | PrepareNextTurnResult
        | undefined;
    shouldStopAfterTurn?: (
        ctx: ShouldStopAfterTurnContext,
    ) => Promise<boolean | undefined> | boolean | undefined;
    // Pi-style steering: inject queued messages before each assistant response.
    getSteeringMessages?: (
        ctx: GetSteeringMessagesContext,
    ) => Promise<Message[] | undefined> | Message[] | undefined;
    // Pi-style follow-up: when the loop would stop with no tool calls, inject
    // queued messages and continue the outer loop.
    getFollowUpMessages?: (
        ctx: GetFollowUpMessagesContext,
    ) => Promise<Message[] | undefined> | Message[] | undefined;
    // Pi-style next-turn: messages queued to be prepended to the next user
    // prompt. Survives turn boundaries — useful for context that persists
    // across the user's next explicit message.
    getNextTurnMessages?: () => Promise<Message[] | undefined> | Message[] | undefined;
}

export interface ToolCall {
    id: string;
    name: string;
    arguments: string;
}

export interface Tool {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    // Optional compatibility shim for weak function-callers or resumed sessions
    // with older argument shapes. Runs after best-effort parsing and before
    // hooks/tool execution.
    prepareArguments?: (args: unknown) => Record<string, unknown>;
    // Tools that mutate state or depend on ordering should be sequential.
    // Read-only tools may opt into parallel execution when global toolExecution
    // is parallel.
    executionMode?: ToolExecutionMode;
    execute: (
        args: Record<string, unknown>,
        ctx: ToolContext,
    ) => Promise<string>;
}

export interface ToolContext {
    cwd?: string;
    maxOutputChars?: number;
    signal?: AbortSignal;
    onUpdate?: (partialResult: string) => void;
}

export interface AgentConfig {
    baseUrl: string;
    model: string;
    /** Alternative models for cycling. When set, `cycleModel()` switches between them. */
    models?: string[];
    cwd?: string;
    temperature?: number;
    maxTokens?: number;
    chatTemplate?: string;
    stop?: string[];
    maxIterations?: number;
    contextWindowTokens?: number;
    systemPrompt?: string;
    tools?: Tool[];
    onEvent?: EventHandler;
    runtimeHooksEnabled?: boolean;
    hookSessionId?: string;
    hookTranscriptPath?: string;
    hooks?: AgentLoopHooks;
    // Callback invoked with turn_end before the event fires on onEvent.
    // Lets the bridge forward turn_end with the correct turn_id to the TUI.
    turnEndCallback?: (turnId: string) => void;
    // Built-in loop safeguards. Each rides a contract hook.
    guardsEnabled?: boolean; // duplicate + failure-loop guards (default on)
    duplicateToolThreshold?: number;
    toolFailureLoopThreshold?: number;
    budgetStopEnabled?: boolean; // diminishing-returns early stop (default OFF)
    proactiveCompactionEnabled?: boolean; // compact before hitting context wall
    proactiveCompactionFraction?: number; // trigger at this fraction of window
    // Pi-style continuation: resume the agent when it stops with pending todos.
    continuationEnabled?: boolean; // default on
    maxContinuations?: number; // cap per run (default 12)
    toolExecution?: ToolExecutionMode; // default sequential
    // Pi-style queue modes: how steering/follow-up messages are drained.
    steeringQueueMode?: QueueMode; // default "all" (drain all at once)
    followUpQueueMode?: QueueMode; // default "all" (drain all at once)
    // Thinking/reasoning level for models that support it.
    thinkingLevel?: ThinkingLevel; // default "medium"
    // Auto-retry on provider errors (429, 500, 502, 503, 504, timeouts).
    autoRetryEnabled?: boolean; // default on
    maxRetries?: number; // max retry attempts (default 3)
    retryBaseDelayMs?: number; // base delay for exponential backoff (default 1000)
    // Web search backend (SearXNG). When set, the web_search tool is enabled.
    webSearch?: WebSearchConfig;
}

export interface WebSearchConfig {
    /** Base URL of the SearXNG instance (e.g. http://localhost:8090). */
    baseUrl: string;
    /** Max results to return (default 10). */
    maxResults?: number;
}
