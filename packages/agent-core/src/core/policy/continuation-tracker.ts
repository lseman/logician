// ── Continuation tracking ────────────────────────────────────────────────
// In-memory replacement for the durable Run Kernel's cross-run counters.
// Tracks how many autonomous continuation runs a task has gone through and
// whether recent runs made semantic progress, so the harness can pause an
// autonomous loop that's spinning without needing a crash-durable ledger.

import { randomUUID } from "node:crypto";
import type { RunOutcomeStatus } from "./execution-policy.ts";

export interface ContinuationLimits {
	maxRuns: number;
	maxNoProgressRuns: number;
	maxElapsedMs: number;
}

export const DEFAULT_CONTINUATION_LIMITS: ContinuationLimits = {
	maxRuns: 8,
	maxNoProgressRuns: 3,
	maxElapsedMs: 30 * 60_000,
};

export type TaskStatus = "idle" | "active" | RunOutcomeStatus;

export interface ContinuationState {
	taskId?: string;
	status: TaskStatus;
	createdAt?: number;
	updatedAt?: number;
	continuationRuns: number;
	noProgressRuns: number;
	lastProgressFingerprint: string;
	compactionGeneration: number;
	terminalReason?: string;
}

export interface RunBudgetStatus {
	continuationsRemaining: number;
	noProgressRemaining: number;
	elapsedMs: number;
	timeRemainingMs: number;
}

export type ContinuationDecision =
	| { action: "continue"; state: ContinuationState }
	| { action: "pause"; reason: string; state: ContinuationState };

export function initialContinuationState(): ContinuationState {
	return {
		status: "idle",
		continuationRuns: 0,
		noProgressRuns: 0,
		lastProgressFingerprint: "",
		compactionGeneration: 0,
	};
}

/** Tracks continuation budget across an autonomous task's runs, entirely in memory. */
export class ContinuationTracker {
	private state: ContinuationState = initialContinuationState();
	private readonly limits: ContinuationLimits;

	constructor(limits: Partial<ContinuationLimits> = {}) {
		this.limits = { ...DEFAULT_CONTINUATION_LIMITS, ...limits };
	}

	snapshot(): ContinuationState {
		return { ...this.state };
	}

	startTask(progressFingerprint = ""): ContinuationState {
		const now = Date.now();
		this.state = {
			taskId: randomUUID(),
			status: "active",
			createdAt: now,
			updatedAt: now,
			continuationRuns: 0,
			noProgressRuns: 0,
			lastProgressFingerprint: progressFingerprint,
			compactionGeneration: 0,
		};
		return this.snapshot();
	}

	requestContinuation(progressFingerprint: string): ContinuationDecision {
		const prior = this.state;
		if (prior.status === "blocked" || prior.status === "needs_input")
			return {
				action: "pause",
				reason: prior.terminalReason ?? "run is paused",
				state: this.snapshot(),
			};
		if (
			!prior.taskId ||
			prior.status === "idle" ||
			prior.status === "failed" ||
			prior.status === "cancelled"
		)
			this.startTask(progressFingerprint);

		const current = this.state;
		current.continuationRuns++;
		if (
			progressFingerprint &&
			progressFingerprint !== current.lastProgressFingerprint
		) {
			current.lastProgressFingerprint = progressFingerprint;
			current.noProgressRuns = 0;
		} else {
			current.noProgressRuns++;
		}
		current.updatedAt = Date.now();

		const elapsedMs = Math.max(
			0,
			current.updatedAt - (current.createdAt ?? current.updatedAt),
		);
		let reason: string | undefined;
		if (current.continuationRuns > this.limits.maxRuns)
			reason = `continuation run budget exhausted (${this.limits.maxRuns})`;
		else if (current.noProgressRuns >= this.limits.maxNoProgressRuns)
			reason = `no semantic progress across ${current.noProgressRuns} continuation runs`;
		else if (elapsedMs > this.limits.maxElapsedMs)
			reason = `continuation time budget exhausted (${this.limits.maxElapsedMs}ms)`;

		if (!reason) return { action: "continue", state: this.snapshot() };
		this.finish("blocked", reason);
		return { action: "pause", reason, state: this.snapshot() };
	}

	recordCompaction(): void {
		if (!this.state.taskId) return;
		this.state.compactionGeneration++;
		this.state.updatedAt = Date.now();
	}

	finish(status: RunOutcomeStatus, reason?: string): void {
		if (!this.state.taskId) return;
		this.state = {
			...this.state,
			status,
			terminalReason: reason,
			updatedAt: Date.now(),
		};
	}

	budgetStatus(now = Date.now()): RunBudgetStatus | undefined {
		const state = this.state;
		if (!state.taskId || state.createdAt === undefined) return undefined;
		const elapsedMs = Math.max(0, now - state.createdAt);
		return {
			continuationsRemaining: Math.max(
				0,
				this.limits.maxRuns - state.continuationRuns,
			),
			noProgressRemaining: Math.max(
				0,
				this.limits.maxNoProgressRuns - state.noProgressRuns,
			),
			elapsedMs,
			timeRemainingMs: Math.max(0, this.limits.maxElapsedMs - elapsedMs),
		};
	}
}
