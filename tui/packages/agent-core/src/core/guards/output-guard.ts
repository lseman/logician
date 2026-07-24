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

import type { EventHandler } from "../types.ts";
import { BackendError, type BackendErrorCategory } from "../backend.ts";
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
	/** Hook to trigger compaction. Returns tokens saved, or null if no compaction. */
	onCompact?: () => Promise<number | null>;
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
	Omit<OutputGuardConfig, "onCompact" | "onEvent" | "loopDetector">
> = {
	maxRetries: 3,
	retryBaseDelayMs: 500,
	maxRetryDelayMs: 15000,
	autoCompactOnContextFull: true,
	maxEmptyResponses: 3,
	maxNonCommittalResponses: 3,
	budgetThreshold: 0.95,
};

export class OutputGuard {
	private config: Required<
		Omit<OutputGuardConfig, "onCompact" | "onEvent" | "loopDetector">
	>;
	private readonly onEvent: OutputGuardConfig["onEvent"];
	private readonly loopDetector: OutputGuardConfig["loopDetector"];
	private retryCount = 0;
	private consecutiveEmptyResponses = 0;
	private consecutiveNonCommittalResponses = 0;

	constructor(config: OutputGuardConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
		this.onEvent = config.onEvent;
		this.loopDetector = config.loopDetector;
	}

	/**
	 * Process a backend error. Returns the guard's decision on what to do.
	 * Called from the loop runner when the backend throws/categorizes an error.
	 */
	handleError(error: unknown): OutputGuardResult {
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
					message: "Malformed assistant message and automatic retries are disabled.",
					isRetryable: false,
				};
			}
			this.retryCount = 0;
			this.emitEvent({
				type: "auto_retry_start",
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

		// Context-full: auto-compact and retry
		if (category === "context_full") {
			this.retryCount = 0; // Reset retry count for context errors
			if (
				this.config.autoCompactOnContextFull &&
				this.config.maxRetries > 0
			) {
				this.emitEvent({
					type: "auto_retry_start",
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

		// Retryable errors (rate_limit, transient): backoff retry
		if (backendErr?.retryable && this.retryCount < this.config.maxRetries) {
			this.retryCount++;
			const delay = this.computeRetryDelay(backendErr);
			this.emitEvent({
				type: "auto_retry_start",
				attempt: this.retryCount,
				maxRetries: this.config.maxRetries,
				delayMs: delay,
				error: error instanceof Error ? error.message : String(error),
			});
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
				type: "auto_retry_end",
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
		if (
			!backendErr &&
			this.retryCount === 0 &&
			this.config.maxRetries > 0
		) {
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
			type: "auto_retry_end",
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
	 * Process a successful model response. Checks for empty/degenerate patterns.
	 * Returns "abort" if the model keeps returning nothing or is stuck in non-committal mode.
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

		// Track non-committal responses (has content but no tool calls, matches vague patterns)
		if (content && content.trim().length > 0 && hasNoTools) {
			const nonCommittalPatterns = [
				/\b(i\s+(need|should|have|might|could|will)\s+(to\s+)?(?:check|look|think|consider|analyze|investigate|examine|review|verify))\b/i,
				/\b(let\s+me\s+(think|see|check|try|consider))\b/i,
				/\b(i'm\s+(going\s+to|thinking\s+about|not\s+sure|still\s+considering))\b/i,
				/\b(i'll\s+(try|check|look|see|think))\b/i,
				/\b(need\s+to\s+(check|think|verify|confirm))\b/i,
				/\b(however|but|although)\s+(i\s+(need|should|have|might))\b/i,
				/\b(this\s+(requires|needs|demands|warrants)\s+(further|more|additional))\b/i,
				/\b(i\s+(don't|do\s+not)\s+(know|think\s+|certain))\b/i,
				/\blet(?:'s|\s+me)\s+(?:step\s+back|circle\s+back|reconsider)\b/i,
				/\b(at\s+this\s+point|so\s+far)\s+(i\s+(have|can|see)|we\s+(need|should))\b/i,
			];

			const isNonCommittal = nonCommittalPatterns.some((p) => p.test(content));
			if (isNonCommittal) {
				this.consecutiveNonCommittalResponses++;
				if (
					this.consecutiveNonCommittalResponses >=
					this.config.maxNonCommittalResponses
				) {
					this.emitEvent({
						type: "error",
						message: `Model returned ${this.config.maxNonCommittalResponses} consecutive non-committal responses without tool calls. Aborting turn.`,
					});
					return {
						action: "abort",
						message: `Model stuck in non-committal mode. No tool calls after ${this.config.maxNonCommittalResponses} responses.`,
						isRetryable: false,
					};
				}
			} else {
				this.consecutiveNonCommittalResponses = 0;
			}
		}

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
				this.emitEvent({
					type: "budget_exhausted",
					usedTokens: tokensUsed,
					limitTokens: maxTokens,
				});
				return { action: "budget_exhausted" };
			}
		}
		return { action: "proceed" };
	}

	/**
	 * Reset guard state after a completed turn (successful or aborted).
	 */
	reset(): void {
		this.retryCount = 0;
		this.consecutiveEmptyResponses = 0;
		this.consecutiveNonCommittalResponses = 0;
	}

	/**
	 * Get the current retry count for diagnostics.
	 */
	getRetryCount(): number {
		return this.retryCount;
	}

	/**
	 * Check if loop detector found a loop in the current turn.
	 * Returns true if loop detected (and emits event).
	 */
	checkLoopDetection(
		assistantContent: string,
		toolCalls: Array<{ name: string; args: string; result: string }>,
	): boolean {
		if (!this.loopDetector) return false;
		const isLooping = this.loopDetector.recordAndDetect(
			assistantContent,
			toolCalls,
		);
		if (isLooping) {
			const diag = this.loopDetector.getLoopDiagnostic();
			if (diag) {
				this.emitEvent({
					type: "loop_detected",
					message: diag,
					attempt: this.retryCount,
				});
			}
		}
		return isLooping;
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
			if (patterns.some((p) => lower.includes(p))) {
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
