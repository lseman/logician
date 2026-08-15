// ── GuardEngine ──────────────────────────────────────────────────────────
// Minimal guard engine: wraps tool-call guards + output guard only.
//
// Two orthogonal concerns:
//   1. Tool-call guards (pre-execution) — blocks duplicate tool calls and
//      repeated failures.
//   2. Output guard (response level) — handles backend errors (retry/abort),
//      empty responses, and token budget exhaustion.
//
// No multi-signal fusion, no graduated intervention, no composite scoring.
// Callers check tool-call guards and output guard results directly.

import type { LoopDetector } from "./loop-detector.ts";
import type { OutputGuard } from "./output-guard.ts";
import type { EventHandler } from "../types.ts";

// ── Config ────────────────────────────────────────────────────────────────

export interface GuardEngineConfig {
	// ── Tool guards ────────────────────────────────────────────────────────
	guardsEnabled?: boolean;
	duplicateGuardEnabled?: boolean;
	failureGuardEnabled?: boolean;
	loopDuplicateThreshold?: number;
	loopFailureThreshold?: number;

	// ── Output guard ───────────────────────────────────────────────────────
	outputMaxRetries?: number;
	outputRetryBaseDelayMs?: number;
	outputMaxRetryDelayMs?: number;
	outputAutoCompactOnContextFull?: boolean;
	outputMaxEmptyResponses?: number;
	outputMaxNonCommittalResponses?: number;
	outputBudgetThreshold?: number;
	outputMaxConsecutiveCompactions?: number;

	// ── Events ─────────────────────────────────────────────────────────────
	onEvent?: EventHandler;
	onCompact?: () => Promise<number | null>;
}

// ── Public API ──────────────────────────────────────────────────────────────

export interface GuardEngine {
	/** The internal OutputGuard — used by the loop runner for error/response handling. */
	outputGuard: OutputGuard;

	// ── Tool guard (pre-execution) ─────────────────────────────────────────
	/** Check a tool call before execution. Returns block=true with a message. */
	checkToolCall(name: string, args: string): { block: boolean; message?: string; guard?: "duplicate" | "failure" };

	/** Record a failed tool call. */
	recordFailure(name: string, args: string, result: string): void;

	/** Record a successful tool call (decays failure state). */
	recordSuccess(name: string, args: string): void;



	// ── Output guard ───────────────────────────────────────────────────────
	/** Process a backend error. Returns guard decision. */
	handleError(error: unknown): {
		action: "proceed" | "retry" | "compact" | "abort" | "compact_then_retry";
		retryDelayMs?: number;
		attempt?: number;
		message?: string;
		isRetryable?: boolean;
	};

	/** Process a successful model response for empty/degenerate patterns. */
	checkResponse(content: string | null | undefined, toolCallsCount: number): {
		action: "proceed" | "retry" | "compact" | "abort" | "compact_then_retry";
		message?: string;
	};

	/** Process token usage for budget tracking. */
	processResponse(tokensUsed?: number, maxTokens?: number): { action: "proceed" | "budget_exhausted" };

	// ── Recovery memory (simple failure tracking) ──────────────────────────
	/** Record a failure/nudge event. Returns similar past entries. */
	recordFailureEntry(
		failureType: string,
		approach: string,
		outcome: string,
		suggestedAlternative?: string,
	): { entryId: string; similarEntries: Array<{ approach: string; outcome: string; repeatCount: number }> };

	/** Get warnings for a new approach. */
	getFailureWarnings(approach: string, failureType: string): string[];

	/** Look up recovery warnings for this failure and record it, atomically. */
	checkAndRecordFailure(name: string, args: string, result: string): { warnings: string[] };

	// ── Lifecycle ──────────────────────────────────────────────────────────
	/** Reset all guard state. */
	reset(): void;

	/** Get a summary of all guard stats for diagnostics. */
	getStats(): Record<string, unknown>;
}

// ── Factory ─────────────────────────────────────────────────────────────────

export function createGuardEngine(config: GuardEngineConfig = {}): GuardEngine {
	const {
		onEvent,
		onCompact,
		// Tool guard defaults
		guardsEnabled = true,
		duplicateGuardEnabled = true,
		failureGuardEnabled = false,
		loopDuplicateThreshold = 3,
		loopFailureThreshold = 3,
		// Output guard defaults
		outputMaxRetries = 3,
		outputRetryBaseDelayMs = 500,
		outputMaxRetryDelayMs = 15_000,
		outputAutoCompactOnContextFull = true,
		outputMaxEmptyResponses = 3,
		outputMaxNonCommittalResponses = 3,
		outputBudgetThreshold = 0.95,
		outputMaxConsecutiveCompactions = 3,
	} = config;

	// ── Lazy instantiation ──────────────────────────────────────────────
	let _loopDetector: LoopDetector | null = null;
	let _outputGuard: OutputGuard | null = null;

	const getLoopDetector = (): LoopDetector => {
		if (!_loopDetector) {
			const { LoopDetector: LD } = require("./loop-detector.ts");
			_loopDetector = new LD({
				duplicateThreshold: loopDuplicateThreshold,
				failureThreshold: loopFailureThreshold,
			});
		}
		return _loopDetector!;
	};

	const getOutputGuard = (): OutputGuard => {
		if (!_outputGuard) {
			const { OutputGuard: OG } = require("./output-guard.ts");
			_outputGuard = new OG({
				maxRetries: outputMaxRetries,
				retryBaseDelayMs: outputRetryBaseDelayMs,
				maxRetryDelayMs: outputMaxRetryDelayMs,
				autoCompactOnContextFull: outputAutoCompactOnContextFull,
				maxEmptyResponses: outputMaxEmptyResponses,
				maxNonCommittalResponses: outputMaxNonCommittalResponses,
				budgetThreshold: outputBudgetThreshold,
				maxConsecutiveCompactions: outputMaxConsecutiveCompactions,
				onEvent,
				onCompact,
				loopDetector: getLoopDetector(),
			});
		}
		return _outputGuard!;
	};

	// ── Simple recovery memory (in-memory map of past failures) ─────────
	const failureHistory: Array<{
		approach: string;
		failureType: string;
		outcome: string;
		repeatCount: number;
		suggestedAlternative?: string;
	}> = [];

	const findSimilar = (approach: string, failureType: string) =>
		failureHistory.filter(
			(e) =>
				e.approach === approach &&
				e.failureType === failureType,
		);

	// ── Public API ──────────────────────────────────────────────────────
	const duplicateGuardOn = guardsEnabled && duplicateGuardEnabled;
	const failureGuardOn = guardsEnabled && failureGuardEnabled;

	return {
		get outputGuard() {
			return getOutputGuard();
		},

		checkToolCall(name, args) {
			if (!duplicateGuardOn && !failureGuardOn) return { block: false };
			return getLoopDetector().checkToolCall(name, args);
		},

		recordFailure(name, args, result) {
			if (duplicateGuardOn || failureGuardOn) {
				getLoopDetector().recordFailure(name, args, result);
			}
			// Record in simple recovery memory
			const approach = `${name} ${args.slice(0, 200)}`;
			const failureType = result.includes("Error:")
				? result.slice(0, 100).replace(/^Error:\s*/i, "").slice(0, 60)
				: "unknown";
			const similar = findSimilar(approach, failureType);
			if (similar.length > 0) {
				similar[0].repeatCount++;
			} else {
				failureHistory.push({ approach, failureType, outcome: result.slice(0, 300), repeatCount: 1 });
			}
		},

		recordSuccess(name, args) {
			if (duplicateGuardOn || failureGuardOn) {
				getLoopDetector().recordSuccess(name, args);
			}
		},



		handleError(error) {
			return getOutputGuard().handleError(error);
		},

		checkResponse(content, toolCallsCount) {
			return getOutputGuard().checkResponse(content, toolCallsCount);
		},

		processResponse(tokensUsed, maxTokens) {
			return getOutputGuard().processResponse(tokensUsed, maxTokens);
		},

		recordFailureEntry(failureType, approach, outcome, suggestedAlternative) {
			const entryId = `failure-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
			const similar = findSimilar(approach, failureType);
			if (similar.length > 0) {
				similar[0].repeatCount++;
			} else {
				failureHistory.push({ approach, failureType, outcome, repeatCount: 1, suggestedAlternative });
			}
			return {
				entryId,
				similarEntries: similar.map(e => ({
					approach: e.approach,
					outcome: e.outcome,
					repeatCount: e.repeatCount,
				})),
			};
		},

		getFailureWarnings(approach, failureType) {
			const similar = findSimilar(approach, failureType);
			return similar.filter(e => e.repeatCount >= 2).map(
				e => `[recovery-memory] ${e.outcome.slice(0, 150)}. ${e.suggestedAlternative ? `Try: ${e.suggestedAlternative}` : "Consider a different strategy."}`,
			);
		},

		checkAndRecordFailure(name, args, result) {
			if (duplicateGuardOn || failureGuardOn) {
				getLoopDetector().recordFailure(name, args, result);
			}
			const approach = `${name} ${args.slice(0, 200)}`;
			const failureType = result.includes("Error:")
				? result.slice(0, 100).replace(/^Error:\s*/i, "").slice(0, 60)
				: "unknown";
			const similar = findSimilar(approach, failureType);
			if (similar.length > 0) {
				similar[0].repeatCount++;
			} else {
				failureHistory.push({ approach, failureType, outcome: result.slice(0, 300), repeatCount: 1 });
			}
			return {
				warnings: similar.filter(e => e.repeatCount >= 2).map(
					e => `[recovery-memory] This approach has failed ${e.repeatCount} times. Last: ${e.outcome.slice(0, 150)}`,
				),
			};
		},

		reset() {
			getLoopDetector().reset();
			getOutputGuard().reset();
			failureHistory.length = 0;
		},

		getStats() {
			return {
				outputRetryCount: getOutputGuard().getRetryCount(),
				failureHistoryCount: failureHistory.length,
			};
		},
	};
}
