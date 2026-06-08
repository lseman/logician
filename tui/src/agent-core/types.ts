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

// ── AgentMessage Abstraction ─────────────────────────────────────────────
// Union of standard LLM messages + custom app messages (notifications,
// status updates, UI-only artifacts). Apps extend via declaration merging
// into CustomAgentMessages (see below).

/** Standard LLM-compatible roles only. */
export type LlmRole = MessageRole;

/** Custom agent message types — extend via declaration merging. */
export type CustomAgentMessages = {};

/** Helper: map custom keys to message shapes. */
export type CustomAgentMessageMap = {
	[K in keyof CustomAgentMessages]: CustomAgentMessages[K] & {
		role: K extends string ? K : never;
	};
};

/** Union of standard Message + custom app messages. */
export type AgentMessage =
	| Message
	| CustomAgentMessageMap[keyof CustomAgentMessageMap & string];

/** Why the model (or loop) ended its turn. Mirrors the provider stop reason,
 *  plus "tool_calls" when the assistant requested tools and "aborted". */
export type StopReason = "stop" | "length" | "tool_calls" | "error" | "aborted";

export type AgentEvent =
	| { type: "agent_start" }
	// Carries the final conversation so a consumer can persist / render the
	// completed transcript without tracking every incremental event.
	| { type: "agent_end"; messages?: Message[] }
	| { type: "turn_start"; turnId: string }
	// stopReason: why the turn ended. message/toolResults: the assistant message
	// produced this turn and any tool results, so the UI can render a completed
	// turn from one event.
	| {
			type: "turn_end";
			turnId: string;
			stopReason?: StopReason;
			message?: Message;
			toolResults?: Message[];
	  }
	| { type: "message_start"; turnId: string; role: MessageRole }
	| { type: "text_start"; turnId: string }
	| { type: "text_delta"; turnId: string; delta: string }
	| { type: "text_end"; turnId: string }
	| { type: "message_update"; turnId: string; message: Message }
	| { type: "message_end"; turnId: string }
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
	| { type: "thinking_delta"; delta: string }
	| {
			type: "tool_call_start";
			toolName: string;
			toolCallId: string;
			args: string;
	  }
	| {
			type: "tool_call_delta";
			toolCallId: string;
			delta: string;
	  }
	| {
			type: "tool_call_end";
			toolName: string;
			toolCallId: string;
			result: string;
			isError?: boolean;
			// Structured metadata returned by the tool alongside its text result.
			details?: Record<string, unknown>;
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
	| { type: "phase"; phase: "thinking" | "tool" | "idle" }
	| {
			type: "auto_retry_start";
			attempt: number;
			maxRetries: number;
			delayMs: number;
			error: string;
	  }
	| { type: "auto_retry_end"; attempt: number; success: boolean }
	| { type: "model_select"; model: string; index: number }
	// Emitted when the loop stops because the safety cap on unproductive turns
	// was reached (not because the agent finished). Lets the UI distinguish a
	// truncated run from a completed one.
	| { type: "max_iterations"; iterations: number; limit: number }
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

export interface TransformContext {
	messages: AgentMessage[];
	iteration: number;
	signal?: AbortSignal;
}

// Fires just before each provider request. Lets a hook inject per-request
// headers (e.g. a freshly-resolved OAuth token) or tune the timeout.
export interface BeforeProviderRequestContext {
	model: string;
	sessionId: string;
	iteration: number;
}

export interface BeforeProviderRequestResult {
	headers?: Record<string, string>;
	timeoutMs?: number;
}

// Fires with the fully-built request payload right before serialization. Lets a
// hook inspect or rewrite the raw body (analytics, A/B params, provider quirks).
export interface BeforeProviderPayloadContext {
	model: string;
	payload: Record<string, unknown>;
}

export interface BeforeProviderPayloadResult {
	payload: Record<string, unknown>;
}

// Return rewritten messages to replace the working context before the LLM call
// (pruning, external-context injection, nextTurn drain). Runs before
// convertToLlm, after steering injection. Return undefined = no change.
export interface TransformContextResult {
	messages: AgentMessage[];
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
	) =>
		| Promise<BeforeToolCallResult | undefined>
		| BeforeToolCallResult
		| undefined;
	afterToolCall?: (
		ctx: AfterToolCallContext,
	) =>
		| Promise<AfterToolCallResult | undefined>
		| AfterToolCallResult
		| undefined;
	prepareNextTurn?: (
		ctx: PrepareNextTurnContext,
	) =>
		| Promise<PrepareNextTurnResult | undefined>
		| PrepareNextTurnResult
		| undefined;
	// Pi-style context transform: rewrite the working AgentMessage[] before each
	// LLM call (before convertToLlm). Used for proactive pruning / external
	// context injection / draining the harness nextTurn queue.
	transformContext?: (
		ctx: TransformContext,
	) =>
		| Promise<TransformContextResult | undefined>
		| TransformContextResult
		| undefined;
	// Provider-boundary hooks: inject per-request headers / tune timeout, and
	// inspect/rewrite the raw request payload before it is sent.
	beforeProviderRequest?: (
		ctx: BeforeProviderRequestContext,
	) =>
		| Promise<BeforeProviderRequestResult | undefined>
		| BeforeProviderRequestResult
		| undefined;
	beforeProviderPayload?: (
		ctx: BeforeProviderPayloadContext,
	) =>
		| Promise<BeforeProviderPayloadResult | undefined>
		| BeforeProviderPayloadResult
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
}

export interface ToolCall {
	id: string;
	name: string;
	arguments: string;
}

// Structured tool result. Tools may return a plain string (back-compat) or this
// shape to attach machine-readable `details` (diffs, line counts, paths) that
// flow through tool_call_end for richer UI without polluting the text the model
// sees.
export interface ToolResult {
	content: string;
	details?: Record<string, unknown>;
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
	// Extra names this tool answers to when matching Claude-Code-style hook
	// matchers (PreToolUse / PostToolUse). e.g. the `bash` tool aliases "Bash".
	// The loop builds the matcher value from the tool's own name + these.
	hookAliases?: string[];
	// Return a string (content only) or a ToolResult (content + structured
	// details). The registry normalizes both to a ToolResult.
	execute: (
		args: Record<string, unknown>,
		ctx: ToolContext,
	) => Promise<string | ToolResult>;
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
	/** Hook event observer for debugging/logging. Receives every hook event
	 *  with its name and context. Unsubscribe via the returned function. */
	onHookEvent?: (event: string, ctx: unknown) => void;
	runtimeHooksEnabled?: boolean;
	hookSessionId?: string;
	hookTranscriptPath?: string;
	hooks?: AgentLoopHooks;
	/** Internal hooks injected by the harness (queue drains). Composed between
	 *  built-ins and user hooks so they share one HookBus rather than a second
	 *  wrapping bus. Not for application use — set `hooks` instead. */
	internalHooks?: AgentLoopHooks;
	/** Custom LLM conversion: filter/transform AgentMessage[] → Message[].
	 *  Default drops non-standard-role messages. Override for custom logic. */
	convertToLlm?: (messages: AgentMessage[]) => Message[];
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
	steeringQueueMode?: QueueMode; // default "one-at-a-time" (interruptible)
	followUpQueueMode?: QueueMode; // default "one-at-a-time" (interruptible)
	// Thinking/reasoning level for models that support it.
	thinkingLevel?: ThinkingLevel; // default "medium"
	// Reasoning mode: runs a structured reasoner on the user prompt before
	// the ReAct loop. The reasoner output is injected as a synthetic assistant
	// message. Set to "none" or undefined for default ReAct.
	reasonerId?: string; // e.g. "ssr", "tot", "reflexion", "none"
	// Auto-retry on provider errors (429, 500, 502, 503, 504, timeouts).
	autoRetryEnabled?: boolean; // default on
	maxRetries?: number; // max retry attempts (default 3)
	retryBaseDelayMs?: number; // base delay for exponential backoff (default 1000)
	// Per-turn timeout to prevent runaway turns. 0 = no timeout.
	turnTimeoutMs?: number; // default 300_000 (5 min)
	// Web search backend (SearXNG). When set, the web_search tool is enabled.
	webSearch?: WebSearchConfig;
	// Cache of tool results (LRU + TTL). Skips re-executing identical calls.
	// Stale entries expire after cacheTtlMs (default 30s).
	cacheSize?: number; // default 1000
	cacheTtlMs?: number; // default 30_000
}

export interface WebSearchConfig {
	/** Base URL of the SearXNG instance (e.g. http://localhost:8090). */
	baseUrl: string;
	/** Max results to return (default 10). */
	maxResults?: number;
}

// ── Structured error types ──────────────────────────────────────────────
// Replace raw Error(message) with typed errors for better error handling
// and debugging.

export enum AgentErrorType {
	// Turn-level errors
	TURN_TIMEOUT = "turn_timeout",
	CONTEXT_FULL = "context_full",
	PROVIDER_ERROR = "provider_error",
	ABORTED = "aborted",

	// Tool-level errors
	TOOL_EXECUTION_FAILED = "tool_execution_failed",
	TOOL_ARGUMENT_ERROR = "tool_argument_error",
	TOOL_DUPLICATE_CALL = "tool_duplicate_call",
	TOOL_FAILURE_LOOP = "tool_failure_loop",

	// Hook errors
	HOOK_FAILED = "hook_failed",

	// Config errors
	INVALID_CONFIG = "invalid_config",
}

export interface AgentErrorOptions {
	type: AgentErrorType;
	message: string;
	cause?: unknown;
	turnId?: string;
	toolName?: string;
	retryable?: boolean;
}

export class AgentError extends Error {
	readonly type: AgentErrorType;
	readonly cause?: unknown;
	readonly turnId?: string;
	readonly toolName?: string;
	readonly retryable: boolean;

	constructor(options: AgentErrorOptions) {
		super(options.message, { cause: options.cause as Error });
		this.name = "AgentError";
		this.type = options.type;
		this.cause = options.cause;
		this.turnId = options.turnId;
		this.toolName = options.toolName;
		this.retryable = options.retryable ?? this.isDefaultRetryable(options.type);
	}

	private isDefaultRetryable(type: AgentErrorType): boolean {
		return (
			type === AgentErrorType.PROVIDER_ERROR ||
			type === AgentErrorType.CONTEXT_FULL
		);
	}
}

/** Create a structured error from a raw Error when type is known. */
export function wrapError(
	type: AgentErrorType,
	original: Error,
	extra?: Partial<AgentErrorOptions>,
): AgentError {
	return new AgentError({
		type,
		message: original.message,
		cause: original,
		...extra,
	});
}
