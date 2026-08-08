// ── Typed extension event system ─────────────────────────────────────────
// Structured events for extensions to subscribe to, mirroring Pi's
// ExtensionEvent contract. Each event has a dedicated result type that
// extensions can return to influence agent behavior.
//
// Events are typed and emitted via ExtensionEventBus. Extensions subscribe
// per-event-type and receive strongly-typed payloads.

import type { Message, StopReason } from "../../agent/types.ts";

// ============================================================================
// Extension lifecycle events
// ============================================================================

/** Fired before the agent starts processing a prompt */
export interface BeforeAgentStartEvent {
	type: "before_agent_start";
	prompt: string;
	systemPrompt: string;
}

export interface BeforeAgentStartResult {
	/** Replace or augment system prompt */
	systemPrompt?: string;
	/** Prepend messages before the user prompt */
	messages?: Message[];
}

/** Fired after the agent loop completes */
export interface AgentEndEvent {
	type: "agent_end";
	messages: Message[];
	outcome?: {
		status: "completed" | "needs_input" | "blocked" | "failed" | "cancelled";
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	};
}

/** No result — extensions may inspect messages for diagnostics */
export type AgentEndResult = {};

/** Fired when the agent starts processing a prompt */
export interface AgentStartEvent {
	type: "agent_start";
}

export type AgentStartResult = {};

/** Fired when an agent-level error occurs */
export interface AgentErrorEvent {
	type: "agent_error";
	message: string;
	phase: "model" | "tool" | "compaction" | "network" | "other";
	recoverable: boolean;
}

export type AgentErrorResult = {};

/** Fired when the agent auto-retries a failed run */
export interface AgentRetryStartEvent {
	type: "agent_retry_start";
	attempt: number;
	maxRetries: number;
	delayMs: number;
	error: string;
	reason: "compaction" | "error" | "overflow" | "rate_limit";
}

export type AgentRetryStartResult = {};

/** Fired after the agent auto-retry completes */
export interface AgentRetryEndEvent {
	type: "agent_retry_end";
	attempt: number;
	success: boolean;
	reason: "compaction" | "error" | "overflow" | "rate_limit";
}

export type AgentRetryEndResult = {};

// ============================================================================
// Tool events
// ============================================================================

/** Fired before a tool executes — extensions can block or modify args */
export interface ToolCallEvent {
	type: "tool_call";
	toolCallId: string;
	toolName: string;
	input: Record<string, unknown>;
}

export interface ToolCallResult {
	/** Return { block: true, reason } to block execution */
	block?: boolean;
	reason?: string;
	/** Return { terminate: true } to hint the agent should stop after this batch */
	terminate?: boolean;
}

/** Fired after a tool finishes — extensions can modify the result */
export interface ToolResultEvent {
	type: "tool_result";
	toolCallId: string;
	toolName: string;
	input: Record<string, unknown>;
	content: Array<{ type: string; text: string }>;
	details?: Record<string, unknown>;
	isError: boolean;
	usage?: Record<string, unknown>;
}

export interface ToolResultResult {
	/** Modify the result content */
	content?: Array<{ type: string; text: string }>;
	details?: Record<string, unknown>;
	isError?: boolean;
	usage?: Record<string, unknown>;
}

// ============================================================================
// Turn events
// ============================================================================

/** Fired at the start of each turn */
export interface TurnStartEvent {
	type: "turn_start";
	turnIndex: number;
}

export type TurnStartResult = {};

/** Fired at the end of each turn */
export interface TurnEndEvent {
	type: "turn_end";
	turnIndex: number;
	stopReason?: StopReason;
	message?: Message;
	toolResults?: Message[];
}

export interface TurnEndResult {
	/** Override the turn-end message (for diagnostics, logging, etc.) */
	message?: Message;
}

// ============================================================================
// Message events
// ============================================================================

/** Fired when a message is added to the conversation */
export interface MessageStartEvent {
	type: "message_start";
	message: Message;
}

export type MessageStartResult = {};

/** Fired when a message is updated (streaming) */
export interface MessageUpdateEvent {
	type: "message_update";
	message: Message;
}

export type MessageUpdateResult = {};

/** Fired when a message is fully written */
export interface MessageEndEvent {
	type: "message_end";
	message: Message;
}

export interface MessageEndResult {
	/** Modify the message content (for post-processing) */
	message?: Message;
}

// ============================================================================
// Tool execution events
// ============================================================================

/** Fired before a tool executes — extensions can short-circuit */
export interface ToolExecutionStartEvent {
	type: "tool_execution_start";
	toolCallId: string;
	toolName: string;
	args: Record<string, unknown>;
}

export interface ToolExecutionStartResult {
	/** Return content to short-circuit: tool will not execute */
	content?: string;
	isError?: boolean;
}

/** Fired during tool execution with partial output */
export interface ToolExecutionUpdateEvent {
	type: "tool_execution_update";
	toolCallId: string;
	toolName: string;
	partialResult: string;
}

export type ToolExecutionUpdateResult = {};

/** Fired after a tool finishes */
export interface ToolExecutionEndEvent {
	type: "tool_execution_end";
	toolCallId: string;
	toolName: string;
	result: string;
	isError: boolean;
	details?: Record<string, unknown>;
}

export interface ToolExecutionEndResult {
	/** Override the tool result */
	content?: string;
	isError?: boolean;
	details?: Record<string, unknown>;
}

// ============================================================================
// Context events
// ============================================================================

/** Fired when context usage changes (token tracking) */
export interface ContextUpdateEvent {
	type: "context_update";
	tokens: number;
	maxTokens?: number;
	compacted?: boolean;
	/** null means the provider supplied no cache telemetry. */
	cachedTokens?: number | null;
	/** Tokens sent to the provider (prompt/input); null means unavailable. */
	promptTokens?: number | null;
	/** Tokens generated by the provider (completion/output); null means unavailable. */
	completionTokens?: number | null;
}

export type ContextUpdateResult = {};

/**
 * Fired before each LLM call, allowing extensions to modify messages.
 * Mirrors Pi's `context` event.
 */
export interface ContextEvent {
	type: "context";
	/** Messages that will be sent to the provider (deep copy, safe to modify). */
	messages: unknown[];
	/** Current system prompt (read-only reference). */
	systemPrompt?: string;
}

export type ContextResult = {
	/** Return modified messages to replace what gets sent to the provider. */
	messages?: unknown[];
};

// ============================================================================
// Session events
// ============================================================================

/** Fired before the session is switched */
export interface SessionBeforeSwitchEvent {
	type: "session_before_switch";
	newSessionId: string;
}

export interface SessionBeforeSwitchResult {
	/** Return true to cancel the switch */
	cancel?: boolean;
}

/** Fired before a branch is forked */
export interface SessionBeforeForkEvent {
	type: "session_before_fork";
	parentEntryId: string;
}

export interface SessionBeforeForkResult {
	cancel?: boolean;
}

/** Fired before compaction */
export interface SessionBeforeCompactEvent {
	type: "session_before_compact";
	tokensBefore: number;
	reason: "auto" | "manual" | "overflow" | "threshold";
}

export interface SessionBeforeCompactResult {
	cancel?: boolean;
}

/** Fired when compaction completes */
export interface SessionCompactEvent {
	type: "session_compact";
	tokensBefore: number;
	tokensAfter: number;
	reason: "auto" | "manual" | "overflow" | "threshold";
}

export type SessionCompactResult = {};

/** Fired when the agent shuts down */
export interface SessionShutdownEvent {
	type: "session_shutdown";
}

export type SessionShutdownResult = {};

/** Fired when the steering/follow-up queue changes */
export interface QueueUpdateEvent {
	type: "queue_update";
	steering: string[];
	followUp: string[];
	nextTurn?: string[];
}

export type QueueUpdateResult = {};

// ============================================================================
// Session lifecycle events
// ============================================================================

/** Fired when a session is started, loaded, or reloaded */
export interface SessionStartEvent {
	type: "session_start";
	reason: "startup" | "reload" | "new" | "resume" | "fork";
	previousSessionFile?: string;
}

export type SessionStartResult = {};

/** Fired when a session is deleted */
export interface SessionDeleteEvent {
	type: "session_delete";
	sessionFile: string;
	sessionId: string;
}

export type SessionDeleteResult = {};

/** Fired after the agent loop completes and no more retries/compaction/follow-ups remain */
export interface AgentSettledEvent {
	type: "agent_settled";
}

export type AgentSettledResult = {};

// ============================================================================
// Model events
// ============================================================================

/** Fired when the model changes */
export interface ModelSelectEvent {
	type: "model_select";
	model: { provider: string; id: string; name?: string };
	previousModel?: { provider: string; id: string; name?: string };
	source: "set" | "cycle" | "restore";
}

export type ModelSelectResult = {};

/** Fired when the thinking level changes */
export interface ThinkingLevelSelectEvent {
	type: "thinking_level_select";
	level: string;
	previousLevel?: string;
}

export type ThinkingLevelSelectResult = {};

// ============================================================================
// Provider events
// ============================================================================

/** Fired before sending a request to the LLM provider */
export interface BeforeProviderRequestEvent {
	type: "before_provider_request";
	model: string;
	sessionId: string;
	iteration: number;
	streamOptions: {
		stream?: boolean;
		maxTokens?: number;
	};
}

export interface BeforeProviderRequestResult {
	headers?: Record<string, string>;
	timeoutMs?: number;
	maxRetries?: number;
}

/** Fired after receiving a response from the LLM provider */
export interface AfterProviderResponseEvent {
	type: "after_provider_response";
	model: string;
	content: string;
	toolCallCount: number;
	stopReason: StopReason;
	usageTokens?: number;
	iteration: number;
}

export type AfterProviderResponseResult = {};

// ============================================================================
// Event union and handler types
// ============================================================================

export type ExtensionEvent =
	| BeforeAgentStartEvent
	| AgentStartEvent
	| AgentEndEvent
	| AgentSettledEvent
	| AgentErrorEvent
	| AgentRetryStartEvent
	| AgentRetryEndEvent
	| ToolCallEvent
	| ToolResultEvent
	| TurnStartEvent
	| TurnEndEvent
	| MessageStartEvent
	| MessageUpdateEvent
	| MessageEndEvent
	| ToolExecutionStartEvent
	| ToolExecutionUpdateEvent
	| ToolExecutionEndEvent
	| ContextUpdateEvent
	| ContextEvent
	| SessionBeforeSwitchEvent
	| SessionBeforeForkEvent
	| SessionBeforeCompactEvent
	| SessionCompactEvent
	| SessionShutdownEvent
	| QueueUpdateEvent
	| SessionStartEvent
	| SessionDeleteEvent
	| ModelSelectEvent
	| ThinkingLevelSelectEvent
	| BeforeProviderRequestEvent
	| AfterProviderResponseEvent;

export type ExtensionEventName = ExtensionEvent["type"];

export type ExtensionEventResult<T extends ExtensionEventName> =
	T extends "before_agent_start"
		? BeforeAgentStartResult | undefined
		: T extends "agent_start"
			? AgentStartResult | undefined
			: T extends "agent_end"
				? AgentEndResult | undefined
				: T extends "agent_settled"
					? AgentSettledResult | undefined
					: T extends "agent_error"
						? AgentErrorResult | undefined
						: T extends "agent_retry_start"
							? AgentRetryStartResult | undefined
							: T extends "agent_retry_end"
								? AgentRetryEndResult | undefined
								: T extends "tool_call"
									? ToolCallResult | undefined
									: T extends "tool_result"
										? ToolResultResult | undefined
										: T extends "turn_start"
											? TurnStartResult | undefined
											: T extends "turn_end"
												? TurnEndResult | undefined
												: T extends "message_start"
													? MessageStartResult | undefined
													: T extends "message_update"
														? MessageUpdateResult | undefined
														: T extends "message_end"
															? MessageEndResult | undefined
															: T extends "tool_execution_start"
																? ToolExecutionStartResult | undefined
																: T extends "tool_execution_update"
																	? ToolExecutionUpdateResult | undefined
																	: T extends "tool_execution_end"
																		? ToolExecutionEndResult | undefined
																		: T extends "context_update"
																			? ContextUpdateResult | undefined
																			: T extends "session_before_switch"
																				? SessionBeforeSwitchResult | undefined
																				: T extends "session_before_fork"
																					? SessionBeforeForkResult | undefined
																					: T extends "session_before_compact"
																						? SessionBeforeCompactResult | undefined
																						: T extends "session_compact"
																							? SessionCompactResult | undefined
																							: T extends "session_shutdown"
																								? SessionShutdownResult | undefined
																								: T extends "queue_update"
																									? QueueUpdateResult | undefined
																									: T extends "session_start"
																									? SessionStartResult | undefined
																									: T extends "session_delete"
																										? SessionDeleteResult | undefined
																										: T extends "model_select"
																											? ModelSelectResult | undefined
																											: T extends "thinking_level_select"
																												? ThinkingLevelSelectResult | undefined
																												: T extends "before_provider_request"
																													? BeforeProviderRequestResult | undefined
																													: T extends "after_provider_response"
																														? AfterProviderResponseResult | undefined
																															: undefined;

export type ExtensionEventHandler<T extends ExtensionEventName> = (
	event: Extract<ExtensionEvent, { type: T }>,
) => Promise<ExtensionEventResult<T>> | ExtensionEventResult<T>;

export type ExtensionErrorHandler = (
	error: Error,
	event: ExtensionEventName,
) => void;
