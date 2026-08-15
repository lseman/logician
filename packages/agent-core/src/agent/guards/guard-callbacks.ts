// ── GuardCallbacks ──────────────────────────────────────────────────────────
// Callback-based guardrail system — the pi-style pattern.
//
// Instead of a monolithic GuardEngine class, we expose a set of optional
// callbacks that the loop runner calls at the right moments. Each callback
// receives context and returns a decision (or undefined to pass through).
//
// This is simpler, more testable, and easier to reason about than the old
// GuardEngine which bundled loop detection, output guard, failure tracking,
// and recovery memory into one class.
//
// The loop runner decides what to do with the callback results.

import type { LoopDetector } from "./loop-detector.ts";
import type { OutputGuard } from "./output-guard.ts";

// ── Types ───────────────────────────────────────────────────────────────────

/** Decision returned from `onToolCall`. Return { block: true } to prevent execution. */
export interface ToolCallDecision {
	block: boolean;
	message?: string;
	/** Which guard tripped — lets callers report/emit without parsing message text. */
	guard?: "duplicate" | "failure";
}

/** Decision returned from `onToolResult`. Can modify results or signal termination. */
export interface ToolResultDecision {
	/** Override the tool result content. */
	content?: string;
	/** Override tool result details. */
	details?: Record<string, unknown>;
	/** Override whether the result is an error. */
	isError?: boolean;
	/** Signal that the agent should stop after this result. */
	terminate?: boolean;
}

/** Decision returned from `onTurnComplete`. No longer used for loop detection. */
export interface TurnDecision {
	/** Reserved for future use. */
	_?: never;
}

/** Decision returned from `onError` for backend/provider errors. */
export interface ErrorDecision {
	/** What to do about the error. */
	action: "proceed" | "retry" | "compact" | "abort" | "compact_then_retry";
	/** Retry delay in ms (only for "retry" / "compact_then_retry"). */
	retryDelayMs?: number;
	/** Retry attempt number (1-based). */
	attempt?: number;
	/** Max retries configured. */
	maxRetries?: number;
	/** Error message for the user. */
	message?: string;
	/** Whether the original error is retryable. */
	isRetryable?: boolean;
}

/** Decision returned from `onResponse` for model response analysis. */
export interface ResponseDecision {
	/** What to do about the response. */
	action: "proceed" | "retry" | "compact" | "abort" | "compact_then_retry";
	/** Message for the user. */
	message?: string;
}

/** Decision returned from `onBudget` for token budget tracking. */
export interface BudgetDecision {
	/** What to do about budget usage. */
	action: "proceed" | "budget_exhausted";
}

// ── Callback Interfaces ─────────────────────────────────────────────────────

/** Called before each tool call. Return a decision to block or pass through. */
export type OnToolCall = (context: {
	toolName: string;
	args: string;
}) => ToolCallDecision | undefined;

/** Called after each tool call. Can modify results or signal termination. */
export type OnToolResult = (context: {
	toolName: string;
	args: string;
	result: string;
	isError: boolean;
}) => ToolResultDecision | undefined;

/** Called after each turn. Return a decision if a loop is detected. */
export type OnTurnComplete = (context: {
	assistantContent: string;
	toolCalls: Array<{ name: string; args: string; result: string }>;
}) => TurnDecision | undefined;

/** Called when a backend error occurs. Return a decision on how to handle it. */
export type OnError = (error: unknown) => ErrorDecision;

/** Called after each model response. Return a decision on the response. */
export type OnResponse = (context: {
	content: string | null;
	toolCallsCount: number;
}) => ResponseDecision | undefined;

/** Called with token usage. Return a budget decision. */
export type OnBudget = (context: {
	tokensUsed: number;
	maxTokens: number;
}) => BudgetDecision | undefined;

// ── GuardCallbacks Interface ────────────────────────────────────────────────

/**
 * A set of optional callbacks that the loop runner calls at the right moments.
 * Each callback can return a decision or undefined (pass-through).
 *
 * The loop runner is responsible for acting on the decisions — the callbacks
 * only observe and decide. This separation of concerns makes the system
 * simpler, more testable, and easier to extend.
 */
export interface GuardCallbacks {
	/** The underlying OutputGuard — exposed for the runner to use directly. */
	outputGuard: OutputGuard | null;
	/** Called before each tool call. */
	onToolCall?: OnToolCall;
	/** Called after each tool call. */
	onToolResult?: OnToolResult;
	/** Called after each turn. No longer used for loop detection. */
	onTurnComplete?: OnTurnComplete;
	/** Called when a backend error occurs. */
	onError?: OnError;
	/** Called after each model response. */
	onResponse?: OnResponse;
	/** Called with token usage. */
	onBudget?: OnBudget;
	/** Reset all internal state. */
	reset: () => void;
}

// ── Factory ───────────────────────────────────────────────────────────────────

export interface GuardCallbacksConfig {
	/** Loop detector instance (optional). When provided, powers onToolCall and onTurnComplete. */
	loopDetector?: LoopDetector | null;
	/** Output guard instance (optional). When provided, powers onError and onResponse. */
	outputGuard?: OutputGuard | null;
}

/**
 * Create a GuardCallbacks instance from a loop detector and output guard.
 * This is the factory that wires the standalone utilities into the callback interface.
 */
export function createGuardCallbacks(config: GuardCallbacksConfig = {}): GuardCallbacks {
	const { loopDetector, outputGuard } = config;

	return {
		get outputGuard() {
			return outputGuard ?? null;
		},

		onToolCall: loopDetector
			? (context) => loopDetector.checkToolCall(context.toolName, context.args)
			: undefined,

		onToolResult: outputGuard
			? (context) => {
					// Record failures/successes for the loop detector
					if (context.isError) {
						loopDetector?.recordFailure(context.toolName, context.args, context.result);
					} else {
						loopDetector?.recordSuccess(context.toolName, context.args);
					}
					return undefined;
				}
			: undefined,

		// onTurnComplete kept as a hook point; no longer performs detection.
		onTurnComplete: undefined,

		onError: outputGuard ? (error) => outputGuard.handleError(error) : undefined,

		onResponse: outputGuard
			? (context) => outputGuard.checkResponse(context.content, context.toolCallsCount)
			: undefined,

		onBudget: outputGuard
			? (context) => outputGuard.processResponse(context.tokensUsed, context.maxTokens)
			: undefined,

		reset: () => {
			loopDetector?.reset();
			outputGuard?.reset();
		},
	};
}
