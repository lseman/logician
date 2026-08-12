import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import type { ExplicitTaskState } from "./tasks/task-state-controller.ts";

export interface ContinuationLimits {
	maxRuns: number;
	maxNoProgressRuns: number;
	maxElapsedMs: number;
}

export interface DurableRunState {
	version: 1;
	sessionId: string;
	runId: string;
	rootPrompt: string;
	createdAt: number;
	updatedAt: number;
	status: "active" | "completed" | "paused" | "failed" | "cancelled";
	continuationRuns: number;
	noProgressRuns: number;
	lastProgressFingerprint: string;
	lastCause: string;
	taskState?: ExplicitTaskState;
	outcome?: {
		status: "completed" | "needs_input" | "blocked" | "failed" | "cancelled";
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	};
	terminalReason?: string;
	compactionGeneration: number;
	lastEventSequence: number;
}

export type ContinuationDecision =
	| { action: "continue"; state: DurableRunState }
	| { action: "pause"; reason: string; state: DurableRunState };

export interface RunBudgetStatus {
	continuationsRemaining: number;
	noProgressRemaining: number;
	elapsedMs: number;
	timeRemainingMs: number;
}

type RunStateEvent =
	| { type: "run_started"; state: Omit<DurableRunState, "lastEventSequence"> }
	| {
			type: "continuation_requested";
			cause: string;
			progressFingerprint: string;
	  }
	| { type: "task_state_updated"; state: ExplicitTaskState }
	| { type: "compaction_committed" }
	| { type: "run_outcome"; outcome: NonNullable<DurableRunState["outcome"]> }
	| { type: "run_paused"; reason: string }
	| { type: "run_failed"; reason?: string };

interface JournalEntry {
	version: 1;
	sequence: number;
	timestamp: number;
	sessionId: string;
	runId: string;
	event: RunStateEvent;
}

const DEFAULT_LIMITS: ContinuationLimits = {
	maxRuns: 8,
	maxNoProgressRuns: 3,
	maxElapsedMs: 30 * 60_000,
};

function safeId(value: string): string {
	return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

/** Durable, replayable owner for one user task and all of its continuations. */
export class RunStateController {
	private state?: DurableRunState;
	private sessionId: string;
	private readonly limits: ContinuationLimits;

	constructor(
		private readonly cwd: string,
		sessionId: string,
		limits: Partial<ContinuationLimits> = {},
	) {
		this.sessionId = sessionId;
		this.limits = { ...DEFAULT_LIMITS, ...limits };
		this.load();
	}

	useSession(sessionId: string): void {
		this.sessionId = sessionId;
		this.state = undefined;
		this.load();
	}

	start(rootPrompt: string, progressFingerprint = ""): DurableRunState {
		const now = Date.now();
		const initial: Omit<DurableRunState, "lastEventSequence"> = {
			version: 1,
			sessionId: this.sessionId,
			runId: randomUUID(),
			rootPrompt,
			createdAt: now,
			updatedAt: now,
			status: "active",
			continuationRuns: 0,
			noProgressRuns: 0,
			lastProgressFingerprint: progressFingerprint,
			lastCause: "user_prompt",
			compactionGeneration: 0,
		};
		this.state = { ...initial, lastEventSequence: 0 };
		this.append({ type: "run_started", state: initial });
		return this.snapshot() as DurableRunState;
	}

	requestContinuation(
		cause: string,
		progressFingerprint: string,
	): ContinuationDecision {
		if (this.state?.status === "paused") {
			return {
				action: "pause",
				reason: this.state.terminalReason ?? "run is paused",
				state: this.snapshot() as DurableRunState,
			};
		}
		if (
			!this.state ||
			this.state.status === "failed" ||
			this.state.status === "cancelled"
		) {
			this.start("restored continuation", progressFingerprint);
		}
		this.append({ type: "continuation_requested", cause, progressFingerprint });
		const state = this.state as DurableRunState;
		let reason: string | undefined;
		if (state.continuationRuns > this.limits.maxRuns) {
			reason = `continuation run budget exhausted (${this.limits.maxRuns})`;
		} else if (state.noProgressRuns >= this.limits.maxNoProgressRuns) {
			reason = `no semantic progress across ${state.noProgressRuns} continuation runs`;
		} else if (state.updatedAt - state.createdAt > this.limits.maxElapsedMs) {
			reason = `continuation time budget exhausted (${this.limits.maxElapsedMs}ms)`;
		}
		if (reason) {
			this.append({ type: "run_paused", reason });
			return {
				action: "pause",
				reason,
				state: this.snapshot() as DurableRunState,
			};
		}
		return { action: "continue", state: this.snapshot() as DurableRunState };
	}

	applyTaskState(state: ExplicitTaskState): void {
		if (this.state) this.append({ type: "task_state_updated", state });
	}

	applyOutcome(outcome: NonNullable<DurableRunState["outcome"]>): void {
		if (this.state) this.append({ type: "run_outcome", outcome });
	}

	recordCompaction(): void {
		if (this.state) this.append({ type: "compaction_committed" });
	}

	fail(reason?: string): void {
		if (this.state) this.append({ type: "run_failed", reason });
	}

	snapshot(): DurableRunState | undefined {
		return this.state ? structuredClone(this.state) : undefined;
	}

	budgetStatus(now = Date.now()): RunBudgetStatus | undefined {
		if (!this.state) return undefined;
		const elapsedMs = Math.max(0, now - this.state.createdAt);
		return {
			continuationsRemaining: Math.max(
				0,
				this.limits.maxRuns - this.state.continuationRuns,
			),
			noProgressRemaining: Math.max(
				0,
				this.limits.maxNoProgressRuns - this.state.noProgressRuns,
			),
			elapsedMs,
			timeRemainingMs: Math.max(0, this.limits.maxElapsedMs - elapsedMs),
		};
	}

	private journalPath(): string {
		return path.join(
			this.cwd,
			".logician",
			"runtime",
			`${safeId(this.sessionId)}.jsonl`,
		);
	}

	private append(event: RunStateEvent): void {
		if (!this.state) return;
		const entry: JournalEntry = {
			version: 1,
			sequence: this.state.lastEventSequence + 1,
			timestamp: Date.now(),
			sessionId: this.sessionId,
			runId: this.state.runId,
			event,
		};
		this.reduce(entry);
		if (this.sessionId.startsWith("tui_")) return;
		const file = this.journalPath();
		mkdirSync(path.dirname(file), { recursive: true });
		appendFileSync(file, `${JSON.stringify(entry)}\n`, "utf8");
	}

	private load(): void {
		if (this.sessionId.startsWith("tui_")) return;
		const file = this.journalPath();
		if (!existsSync(file)) return;
		for (const line of readFileSync(file, "utf8").split("\n")) {
			if (!line.trim()) continue;
			try {
				const entry = JSON.parse(line) as JournalEntry;
				if (entry.version === 1 && entry.sessionId === this.sessionId)
					this.reduce(entry);
			} catch {
				// Ignore a truncated final record; prior append-only history remains valid.
			}
		}
	}

	private reduce(entry: JournalEntry): void {
		const { event } = entry;
		if (event.type === "run_started") {
			this.state = {
				...event.state,
				compactionGeneration: event.state.compactionGeneration ?? 0,
				lastEventSequence: entry.sequence,
			};
			return;
		}
		if (!this.state || entry.runId !== this.state.runId) return;
		this.state.updatedAt = entry.timestamp;
		this.state.lastEventSequence = entry.sequence;
		if (event.type === "continuation_requested") {
			this.state.status = "active";
			this.state.terminalReason = undefined;
			this.state.continuationRuns++;
			this.state.lastCause = event.cause;
			if (
				event.progressFingerprint &&
				event.progressFingerprint !== this.state.lastProgressFingerprint
			) {
				this.state.lastProgressFingerprint = event.progressFingerprint;
				this.state.noProgressRuns = 0;
			} else this.state.noProgressRuns++;
		} else if (event.type === "task_state_updated")
			this.state.taskState = event.state;
		else if (event.type === "compaction_committed")
			this.state.compactionGeneration++;
		else if (event.type === "run_outcome") {
			this.state.outcome = event.outcome;
			if (event.outcome.status === "completed") this.state.status = "completed";
			else if (event.outcome.status === "failed") this.state.status = "failed";
			else if (event.outcome.status === "cancelled")
				this.state.status = "cancelled";
			else if (
				event.outcome.status === "needs_input" ||
				event.outcome.status === "blocked"
			) {
				this.state.status = "paused";
				this.state.terminalReason =
					event.outcome.summary ?? event.outcome.status;
			}
		} else if (event.type === "run_paused") {
			this.state.status = "paused";
			this.state.terminalReason = event.reason;
		} else if (event.type === "run_failed") {
			this.state.status = "failed";
			this.state.terminalReason = event.reason;
		}
	}
}
