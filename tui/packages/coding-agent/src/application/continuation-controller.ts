import { randomUUID } from "node:crypto";
import {
	existsSync,
	mkdirSync,
	readFileSync,
	renameSync,
	writeFileSync,
} from "node:fs";
import path from "node:path";

export interface ContinuationLimits {
	maxRuns: number;
	maxNoProgressRuns: number;
	maxElapsedMs: number;
}

export interface ContinuationLease {
	version: 1;
	id: string;
	sessionId: string;
	rootPrompt: string;
	createdAt: number;
	updatedAt: number;
	runs: number;
	noProgressRuns: number;
	lastProgressFingerprint: string;
	lastCause: string;
	status: "active" | "completed" | "paused" | "failed";
	terminalReason?: string;
}

export type ContinuationDecision =
	| { action: "continue"; lease: ContinuationLease }
	| { action: "pause"; reason: string; lease: ContinuationLease };

const DEFAULT_LIMITS: ContinuationLimits = {
	maxRuns: 8,
	maxNoProgressRuns: 3,
	maxElapsedMs: 30 * 60_000,
};

function safeSessionId(value: string): string {
	return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

/**
 * Owns the complete lifecycle of an internal continuation chain. Callers only
 * start a user task and request the next continuation; budgeting, persistence,
 * progress accounting, and terminal decisions remain behind this interface.
 */
export class ContinuationController {
	private lease?: ContinuationLease;
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
		this.lease = undefined;
		this.load();
	}

	start(rootPrompt: string, progressFingerprint = ""): ContinuationLease {
		const now = Date.now();
		this.lease = {
			version: 1,
			id: randomUUID(),
			sessionId: this.sessionId,
			rootPrompt,
			createdAt: now,
			updatedAt: now,
			runs: 0,
			noProgressRuns: 0,
			lastProgressFingerprint: progressFingerprint,
			lastCause: "user_prompt",
			status: "active",
		};
		this.persist();
		return structuredClone(this.lease);
	}

	request(cause: string, progressFingerprint: string): ContinuationDecision {
		if (!this.lease || this.lease.status !== "active") {
			this.start("restored continuation", progressFingerprint);
		}
		const lease = this.lease as ContinuationLease;
		lease.runs++;
		lease.updatedAt = Date.now();
		lease.lastCause = cause;
		if (
			progressFingerprint &&
			progressFingerprint !== lease.lastProgressFingerprint
		) {
			lease.lastProgressFingerprint = progressFingerprint;
			lease.noProgressRuns = 0;
		} else {
			lease.noProgressRuns++;
		}

		let reason: string | undefined;
		if (lease.runs > this.limits.maxRuns) {
			reason = `continuation run budget exhausted (${this.limits.maxRuns})`;
		} else if (lease.noProgressRuns >= this.limits.maxNoProgressRuns) {
			reason = `no semantic progress across ${lease.noProgressRuns} continuation runs`;
		} else if (lease.updatedAt - lease.createdAt > this.limits.maxElapsedMs) {
			reason = `continuation time budget exhausted (${this.limits.maxElapsedMs}ms)`;
		}

		if (reason) {
			lease.status = "paused";
			lease.terminalReason = reason;
			this.persist();
			return { action: "pause", reason, lease: structuredClone(lease) };
		}
		this.persist();
		return { action: "continue", lease: structuredClone(lease) };
	}

	finish(status: "completed" | "failed", reason?: string): void {
		if (!this.lease) return;
		this.lease.status = status;
		this.lease.terminalReason = reason;
		this.lease.updatedAt = Date.now();
		this.persist();
	}

	snapshot(): ContinuationLease | undefined {
		return this.lease ? structuredClone(this.lease) : undefined;
	}

	private statePath(): string {
		return path.join(
			this.cwd,
			".logician",
			"continuations",
			`${safeSessionId(this.sessionId)}.json`,
		);
	}

	private load(): void {
		if (this.sessionId.startsWith("tui_")) return;
		const statePath = this.statePath();
		if (!existsSync(statePath)) return;
		try {
			const parsed = JSON.parse(
				readFileSync(statePath, "utf8"),
			) as ContinuationLease;
			if (parsed.version === 1 && parsed.sessionId === this.sessionId) {
				this.lease = parsed;
			}
		} catch {
			// A corrupt lease must not prevent the user from starting a new task.
		}
	}

	private persist(): void {
		if (!this.lease || this.sessionId.startsWith("tui_")) return;
		const statePath = this.statePath();
		mkdirSync(path.dirname(statePath), { recursive: true });
		const temporaryPath = `${statePath}.${process.pid}.tmp`;
		writeFileSync(temporaryPath, JSON.stringify(this.lease, null, 2), "utf8");
		renameSync(temporaryPath, statePath);
	}
}
