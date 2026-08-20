// ── OutputGuard ──────────────────────────────────────────────────────────────
// Watches model responses mid-loop for degenerate patterns and recovers.
//
// Detects:
// 1. Context-full errors from backend → auto-compact + retry
// 2. Model returning empty/no-content responses repeatedly → abort turn
// 3. Excessive tool calls without progress (delegated to LoopDetector)
// 4. Provider errors that are retryable (rate_limit, transient) → backoff retry
//
// Unlike LoopDetector (which watches turn-level patterns), OutputGuard
// operates at the response boundary and can trigger structural actions:
// - compaction
// - retry with backoff
// - turn abortion
//
// Usage: create one per harness/loop invocation, feed each response + error.

import {
	BackendError,
	type BackendErrorCategory,
} from "../provider/backend.ts";
import type { EventHandler } from "../types/types-messages.ts";
import type { LoopDetector } from "./loop-detector.ts";

export interface OutputGuardConfig {
	/** Max retry attempts for transient/provider errors (default 3). */
	maxRetries?: number;
	/** Base delay in ms before first retry (default 500). */
	retryBaseDelayMs?: number;
	/** Max delay cap for retries (default 15000). */
	maxRetryDelayMs?: number;
	/** Whether to auto-compact on context_full (default true). */
	autoCompactOnContextFull?: boolean;
	/** Max consecutive empty assistant responses before aborting (default 3). */
	maxEmptyResponses?: number;
	/** Max consecutive non-committal assistant responses before aborting (default 3). */
	maxNonCommittalResponses?: number;
	/** Context-usage fraction that triggers budget_exhausted (default 0.95). */
	budgetThreshold?: number;
	/** Max consecutive context_full→compact_then_retry cycles before aborting (default 3). */
	maxConsecutiveCompactions?: number;
	/** Emit events to the UI/event bus. */
	onEvent?: EventHandler;
	/** The loop detector for turn-level patterns (optional). */
	loopDetector?: LoopDetector;
}

export interface OutputGuardResult {
	/** What the guard decided to do. */
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

/** Default config values. */
const DEFAULT_CONFIG: Required<
	Omit<OutputGuardConfig, "onEvent" | "loopDetector">
> = {
	maxRetries: 3,
	retryBaseDelayMs: 500,
	maxRetryDelayMs: 15000,
	autoCompactOnContextFull: true,
	maxEmptyResponses: 3,
	maxNonCommittalResponses: 3,
	budgetThreshold: 0.95,
	maxConsecutiveCompactions: 3,
};

export class OutputGuard {
	private config: Required<Omit<OutputGuardConfig, "onEvent" | "loopDetector">>;
	private readonly onEvent: OutputGuardConfig["onEvent"];
	// loopDetector field kept for type compatibility but no longer used for turn detection.
	private retryCount = 0;
	private consecutiveEmptyResponses = 0;
	private consecutiveCompactions = 0;
	// Set by processResponse when context usage crossed budgetThreshold on the
	// last successful response. A subsequent rate_limit/quota error is often
	// the provider's own context-size cap rather than a transient throttle —
	// backoff-retrying the same oversized request would just fail again, so
	// handleError compacts first in that case instead of blindly retrying.
	private contextWasNearFull = false;

	constructor(config: OutputGuardConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
		this.onEvent = config.onEvent;
	}

	/**
	 * Process a backend error. Returns the guard's decision on what to do.
	 * Called from the loop runner when the backend throws/categorizes an error.
	 */
	handleError(error: unknown): OutputGuardResult {
		// Abort is an intentional cancellation — abort immediately, no retry.
		if (error instanceof Error && error.name === "AbortError") {
			return {
				action: "abort",
				message: "Operation aborted",
				isRetryable: false,
			};
		}

		// Classify the error
		const backendErr =
			error instanceof Error ? this.extractBackendError(error) : null;

		// Special handling for malformed assistant message errors.
		// When the API rejects with 400 "Assistant message must contain either
		// 'content' or 'tool_calls'!", the conversation history contains a
		// malformed assistant message (empty content, no tool_calls).  The
		// fix is to compact (which drops the bad message) and retry.
		if (this.isMalformedAssistantMessageError(error)) {
			if (this.config.maxRetries === 0) {
				return {
					action: "abort",
					message:
						"Malformed assistant message and automatic retries are disabled.",
					isRetryable: false,
				};
			}
			this.retryCount = 0;
			this.emitEvent({
				type: "agent_retry_start",
				attempt: 1,
				maxRetries: 1,
				delayMs: 0,
				error: error instanceof Error ? error.message : String(error),
			});
			return {
				action: "compact_then_retry",
				attempt: 1,
				maxRetries: 1,
				isRetryable: true,
				message:
					"Malformed assistant message in history — compaction triggered to drop it.",
			};
		}

		const category = backendErr?.category ?? "unknown";

		// Poisoned history: a stored tool_call has unparseable arguments. Retrying
		// resends the identical history and fails identically every time, and
		// compaction can't help since it never inspects individual tool_calls —
		// so unlike context_full, there is nothing productive left to try here.
		if (category === "poisoned_history") {
			this.emitEvent({
				type: "error",
				message:
					"A tool call in this conversation's history has malformed arguments " +
					"and the backend can't parse it — the request can't be retried or " +
					"compacted around. Start a new conversation to continue.",
			});
			return {
				action: "abort",
				message:
					"A tool call in this conversation's history has malformed arguments " +
					"and the backend can't parse it — the request can't be retried or " +
					"compacted around. Start a new conversation to continue.",
				isRetryable: false,
			};
		}

		// Context-full: auto-compact and retry
		if (category === "context_full") {
			this.retryCount = 0; // Reset retry count for context errors
			if (this.config.autoCompactOnContextFull && this.config.maxRetries > 0) {
				this.consecutiveCompactions++;
				if (
					this.consecutiveCompactions > this.config.maxConsecutiveCompactions
				) {
					this.emitEvent({
						type: "agent_retry_end",
						attempt: this.consecutiveCompactions,
						success: false,
					});
					return {
						action: "abort",
						message:
							`Context kept overflowing after ${this.config.maxConsecutiveCompactions} compaction attempts — ` +
							"likely a single oversized tool call. Aborting instead of compacting again.",
						isRetryable: false,
					};
				}
				this.emitEvent({
					type: "agent_retry_start",
					attempt: 1,
					maxRetries: 1,
					delayMs: 0,
					error: error instanceof Error ? error.message : String(error),
				});
				return {
					action: "compact_then_retry",
					attempt: 1,
					maxRetries: 1,
					isRetryable: true,
					message: "Context too long — compaction triggered.",
				};
			}
			return {
				action: "abort",
				message: "Context full and auto-compaction disabled.",
				isRetryable: false,
			};
		}

		// Rate-limit/quota errors that follow a near-full context are treated
		// like context_full: the provider's 429 is plausibly its own
		// context-size cap, not a transient throttle, and retrying the same
		// oversized request would just repeat the failure. Compact first.
		if (
			category === "rate_limit" &&
			this.contextWasNearFull &&
			this.config.autoCompactOnContextFull &&
			this.config.maxRetries > 0
		) {
			this.contextWasNearFull = false;
			this.consecutiveCompactions++;
			if (this.consecutiveCompactions > this.config.maxConsecutiveCompactions) {
				this.emitEvent({
					type: "agent_retry_end",
					attempt: this.consecutiveCompactions,
					success: false,
				});
				return {
					action: "abort",
					message:
						`Rate-limited after ${this.config.maxConsecutiveCompactions} compaction attempts — ` +
						"likely a single oversized tool call. Aborting instead of compacting again.",
					isRetryable: false,
				};
			}
			this.emitEvent({
				type: "agent_retry_start",
				attempt: 1,
				maxRetries: 1,
				delayMs: 0,
				error: error instanceof Error ? error.message : String(error),
			});
			return {
				action: "compact_then_retry",
				attempt: 1,
				maxRetries: 1,
				isRetryable: true,
				message:
					"Rate limited near context capacity — compaction triggered before retry.",
			};
		}

		// Retryable errors (rate_limit, transient): backoff retry
		// Note: agent_retry_start is emitted by the loop runner after handleError returns,
		// to avoid duplicate events. The OutputGuard only emits agent_retry_end for exhausted retries.
		if (backendErr?.retryable && this.retryCount < this.config.maxRetries) {
			this.retryCount++;
			const delay = this.computeRetryDelay(backendErr);
			return {
				action: "retry",
				retryDelayMs: delay,
				attempt: this.retryCount,
				maxRetries: this.config.maxRetries,
				isRetryable: true,
				message: `${category} error — retry ${this.retryCount}/${this.config.maxRetries}.`,
			};
		}

		// Non-retryable or exhausted retries
		if (backendErr?.retryable && this.retryCount >= this.config.maxRetries) {
			this.emitEvent({
				type: "agent_retry_end",
				attempt: this.retryCount,
				success: false,
			});
			return {
				action: "abort",
				message: `${category} error after ${this.config.maxRetries} retries.`,
				isRetryable: true,
			};
		}

		// Unknown/unclassified error: single retry as safety net
		if (!backendErr && this.retryCount === 0 && this.config.maxRetries > 0) {
			this.retryCount = 1;
			return {
				action: "retry",
				retryDelayMs: this.config.retryBaseDelayMs,
				attempt: 1,
				maxRetries: this.config.maxRetries,
				isRetryable: true,
				message: "Unknown error — retrying once.",
			};
		}

		// Final fallback
		this.emitEvent({
			type: "agent_retry_end",
			attempt: this.retryCount,
			success: false,
		});
		return {
			action: "abort",
			message: error instanceof Error ? error.message : String(error),
			isRetryable: false,
		};
	}

	/**
	 * Process a successful model response. Checks for empty responses.
	 * Returns "abort" if the model keeps returning nothing.
	 */
	checkResponse(
		content: string | null | undefined,
		toolCallsCount: number,
	): OutputGuardResult {
		// Track empty responses (no content AND no tool calls)
		const isEmpty = !content || content.trim().length === 0;
		const hasNoTools = toolCallsCount === 0;

		if (isEmpty && hasNoTools) {
			this.consecutiveEmptyResponses++;
			if (this.consecutiveEmptyResponses >= this.config.maxEmptyResponses) {
				this.emitEvent({
					type: "error",
					message: `Model returned ${this.config.maxEmptyResponses} consecutive empty responses. Aborting turn.`,
				});
				return {
					action: "abort",
					message: `Model returned ${this.config.maxEmptyResponses} consecutive empty responses.`,
					isRetryable: false,
				};
			}
		} else {
			this.consecutiveEmptyResponses = 0;
		}

		this.consecutiveCompactions = 0;
		return { action: "proceed" };
	}

	/**
	 * Process a successful backend response (for context tracking).
	 * Extracts token usage if available and emits context_update.
	 */
	processResponse(
		tokensUsed?: number,
		maxTokens?: number,
	): { action: "proceed" } | { action: "budget_exhausted" } {
		if (tokensUsed !== undefined && maxTokens !== undefined) {
			this.emitEvent({
				type: "context_update",
				tokens: tokensUsed,
				maxTokens,
			});

			// Budget guard: if usage exceeds threshold, stop
			if (tokensUsed > maxTokens * (this.config.budgetThreshold ?? 0.95)) {
				this.contextWasNearFull = true;
				this.emitEvent({
					type: "budget_exhausted",
					usedTokens: tokensUsed,
					limitTokens: maxTokens,
				});
				return { action: "budget_exhausted" };
			}
			this.contextWasNearFull = false;
		}
		return { action: "proceed" };
	}

	/**
	 * Reset guard state after a completed turn (successful or aborted).
	 */
	reset(): void {
		this.retryCount = 0;
		this.consecutiveEmptyResponses = 0;
		this.consecutiveCompactions = 0;
		this.contextWasNearFull = false;
	}

	/**
	 * Get the current retry count for diagnostics.
	 */
	getRetryCount(): number {
		return this.retryCount;
	}

	// ── Internals ──────────────────────────────────────────────────────────

	/** Detect the specific "Assistant message must contain either 'content' or 'tool_calls'" error. */
	private isMalformedAssistantMessageError(error: unknown): boolean {
		const msg = error instanceof Error ? error.message : String(error);
		return (
			msg.includes("Assistant message must contain") &&
			msg.includes("tool_calls")
		);
	}

	private extractBackendError(error: Error): BackendError | null {
		if (error instanceof BackendError) return error;
		// Try to extract category from message for errors that didn't go through classifyHttpError
		const lower = error.message.toLowerCase();
		const categories: Array<{
			patterns: string[];
			category: BackendErrorCategory;
		}> = [
			{
				patterns: [
					"failed to parse tool call arguments",
					"failed to parse tool calls",
					"invalid tool call arguments",
				],
				category: "poisoned_history",
			},
			{
				patterns: [
					"context",
					"too long",
					"too many tokens",
					"maximum context",
					"reduce the length",
				],
				category: "context_full",
			},
			{ patterns: ["rate limit", "429", "retry"], category: "rate_limit" },
			{
				patterns: ["500", "502", "503", "504", "server error", "upstream"],
				category: "transient",
			},
		];
		for (const { patterns, category } of categories) {
			if (patterns.some(p => lower.includes(p))) {
				return new BackendError({ category, message: error.message });
			}
		}
		return null;
	}

	private computeRetryDelay(backendErr: BackendError): number {
		// Use Retry-After header if present
		if (backendErr.retryAfterMs && backendErr.retryAfterMs > 0) {
			return Math.min(backendErr.retryAfterMs, this.config.maxRetryDelayMs);
		}
		// Exponential backoff: base * 2^(attempt-1), capped
		const exponential =
			this.config.retryBaseDelayMs * 2 ** (this.retryCount - 1);
		return Math.min(exponential, this.config.maxRetryDelayMs);
	}

	private emitEvent(
		event: Parameters<NonNullable<OutputGuardConfig["onEvent"]>>[0],
	): void {
		this.onEvent?.(event);
	}
}
