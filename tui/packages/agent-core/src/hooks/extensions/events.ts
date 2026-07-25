// ── Typed extension event system ─────────────────────────────────────────
// Structured events for extensions to subscribe to, mirroring Pi's
// ExtensionEvent contract. Each event has a dedicated result type that
// extensions can return to influence agent behavior.
//
// Events are typed and emitted via ExtensionEventBus. Extensions subscribe
// per-event-type and receive strongly-typed payloads.

import type { Message, StopReason } from "../../core/types.ts";

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
export interface AgentEndResult {}

// ============================================================================
// Turn events
// ============================================================================

/** Fired at the start of each turn */
export interface TurnStartEvent {
	type: "turn_start";
	turnIndex: number;
}

export interface TurnStartResult {}

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

export interface MessageStartResult {}

/** Fired when a message is updated (streaming) */
export interface MessageUpdateEvent {
	type: "message_update";
	message: Message;
}

export interface MessageUpdateResult {}

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

export interface ToolExecutionUpdateResult {}

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
}

export interface ContextUpdateResult {}

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

export interface SessionCompactResult {}

/** Fired when the agent shuts down */
export interface SessionShutdownEvent {
	type: "session_shutdown";
}

export interface SessionShutdownResult {}

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

export interface AfterProviderResponseResult {}

// ============================================================================
// Event union and handler types
// ============================================================================

export type ExtensionEvent =
	| BeforeAgentStartEvent
	| AgentEndEvent
	| TurnStartEvent
	| TurnEndEvent
	| MessageStartEvent
	| MessageUpdateEvent
	| MessageEndEvent
	| ToolExecutionStartEvent
	| ToolExecutionUpdateEvent
	| ToolExecutionEndEvent
	| ContextUpdateEvent
	| SessionBeforeSwitchEvent
	| SessionBeforeForkEvent
	| SessionBeforeCompactEvent
	| SessionCompactEvent
	| SessionShutdownEvent
	| BeforeProviderRequestEvent
	| AfterProviderResponseEvent;

export type ExtensionEventName = ExtensionEvent["type"];

export type ExtensionEventResult<T extends ExtensionEventName> =
	T extends "before_agent_start"
		? BeforeAgentStartResult | undefined
		: T extends "agent_end"
			? AgentEndResult | undefined
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
