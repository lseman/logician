import { randomUUID } from "node:crypto";
import {
	closeSync,
	existsSync,
	fsyncSync,
	mkdirSync,
	openSync,
	readFileSync,
	statSync,
	unlinkSync,
	writeSync,
} from "node:fs";
import path from "node:path";
import {
	initialRunKernelState,
	isRunEventEnvelope,
	RUN_KERNEL_SCHEMA_VERSION,
	type RunEventEnvelope,
	type RunKernelEvent,
	type RunKernelReduction,
	type RunKernelViolation,
	reduceRunKernel,
} from "./run-kernel-events.ts";
import type { ExplicitTaskState } from "./tasks/task-state-controller.ts";

export interface ContinuationLimits {
	maxRuns: number;
	maxNoProgressRuns: number;
	maxElapsedMs: number;
}

export interface RunBudgetStatus {
	continuationsRemaining: number;
	noProgressRemaining: number;
	elapsedMs: number;
	timeRemainingMs: number;
}

export type ContinuationDecision =
	| { action: "continue"; state: RunKernelReduction["state"] }
	| { action: "pause"; reason: string; state: RunKernelReduction["state"] };

const DEFAULT_CONTINUATION_LIMITS: ContinuationLimits = {
	maxRuns: 8,
	maxNoProgressRuns: 3,
	maxElapsedMs: 30 * 60_000,
};

function safeId(value: string): string {
	return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

export interface RunKernelAppendOptions {
	taskId: string;
	runId: string;
	operationId?: string;
	leaseEpoch?: number;
	timestamp?: number;
}

export interface RunKernelLease {
	ownerId: string;
	epoch: number;
	expiresAt: number;
}

export interface RunKernelDoctorReport {
	file: string;
	events: number;
	lastValidSequence: number;
	truncatedFinalRecord: boolean;
	parseErrors: Array<{ line: number; message: string }>;
	violations: RunKernelViolation[];
	incompleteOperations: Array<{
		operationId: string;
		toolCallId?: string;
		toolName: string;
		arguments?: Record<string, unknown>;
		idempotencyKey: string;
		receipt?: string;
		result?: string;
		status: string;
		recovery: string;
		recommendedAction:
			| "retry"
			| "reuse_result"
			| "reconcile_receipt"
			| "quarantine"
			| "none";
	}>;
	orphanedSubagents: Array<{
		agentId: string;
		agent: string;
		task: string;
		lastEventType?: string;
	}>;
}

/** Sole persistence boundary for the versioned execution ledger. */
export class RunKernel {
	private reduction: RunKernelReduction = {
		state: initialRunKernelState(),
		violations: [],
	};
	private sessionId: string;
	private observedFileSize = 0;
	private readonly continuationLimits: ContinuationLimits;

	constructor(
		private readonly cwd: string,
		sessionId: string,
		continuationLimits: Partial<ContinuationLimits> = {},
	) {
		this.sessionId = sessionId;
		this.continuationLimits = {
			...DEFAULT_CONTINUATION_LIMITS,
			...continuationLimits,
		};
		this.reload();
	}

	useSession(sessionId: string): void {
		this.sessionId = sessionId;
		this.reload();
	}

	get filePath(): string {
		return path.join(
			this.cwd,
			".logician",
			"run-kernel",
			`${safeId(this.sessionId)}.jsonl`,
		);
	}

	snapshot(): RunKernelReduction {
		return structuredClone(this.reduction);
	}

	loadEvents(): RunEventEnvelope[] {
		return this.read().events;
	}

	startTask(
		rootPrompt: string,
		progressFingerprint = "",
	): RunKernelReduction["state"] {
		const prior = this.snapshot().state;
		if (prior.taskId && prior.runId && prior.status === "active") {
			this.append(
				{
					type: "run_finished",
					status: "cancelled",
					summary: "superseded by a new user task",
					source: "runtime",
				},
				{
					taskId: prior.taskId,
					runId: prior.runId,
					leaseEpoch: prior.leaseEpoch,
				},
			);
		}
		const runId = randomUUID();
		const options = {
			taskId: runId,
			runId,
			leaseEpoch: Math.max(1, prior.leaseEpoch),
		};
		this.append(
			{
				type: "task_started",
				rootPrompt,
				createdAt: Date.now(),
				progressFingerprint,
			},
			options,
		);
		this.append({ type: "run_started", cause: "prompt" }, options);
		return this.snapshot().state;
	}

	requestContinuation(
		cause: string,
		progressFingerprint: string,
	): ContinuationDecision {
		const prior = this.snapshot().state;
		if (prior.status === "blocked" || prior.status === "needs_input")
			return {
				action: "pause",
				reason: prior.terminalReason ?? "run is paused",
				state: prior,
			};
		if (
			!prior.taskId ||
			!prior.runId ||
			prior.status === "failed" ||
			prior.status === "cancelled"
		)
			this.startTask("restored continuation", progressFingerprint);
		const current = this.snapshot().state;
		if (!current.taskId || !current.runId)
			throw new Error("Run Kernel failed to initialize continuation task");
		this.append(
			{ type: "continuation_requested", cause, progressFingerprint },
			{
				taskId: current.taskId,
				runId: current.runId,
				leaseEpoch: current.leaseEpoch,
			},
		);
		const state = this.snapshot().state;
		const elapsedMs = Math.max(
			0,
			(state.updatedAt ?? Date.now()) - (state.createdAt ?? Date.now()),
		);
		let reason: string | undefined;
		if (state.continuationRuns > this.continuationLimits.maxRuns)
			reason = `continuation run budget exhausted (${this.continuationLimits.maxRuns})`;
		else if (state.noProgressRuns >= this.continuationLimits.maxNoProgressRuns)
			reason = `no semantic progress across ${state.noProgressRuns} continuation runs`;
		else if (elapsedMs > this.continuationLimits.maxElapsedMs)
			reason = `continuation time budget exhausted (${this.continuationLimits.maxElapsedMs}ms)`;
		if (!reason) return { action: "continue", state };
		this.finish("blocked", reason, "runtime");
		return { action: "pause", reason, state: this.snapshot().state };
	}

	budgetStatus(now = Date.now()): RunBudgetStatus | undefined {
		const state = this.snapshot().state;
		if (!state.taskId || state.createdAt === undefined) return undefined;
		const elapsedMs = Math.max(0, now - state.createdAt);
		return {
			continuationsRemaining: Math.max(
				0,
				this.continuationLimits.maxRuns - state.continuationRuns,
			),
			noProgressRemaining: Math.max(
				0,
				this.continuationLimits.maxNoProgressRuns - state.noProgressRuns,
			),
			elapsedMs,
			timeRemainingMs: Math.max(
				0,
				this.continuationLimits.maxElapsedMs - elapsedMs,
			),
		};
	}

	updateTaskState(state: ExplicitTaskState): void {
		this.appendForActive({ type: "task_state_updated", state });
	}
	recordCompaction(): void {
		this.appendForActive({
			type: "compaction_committed",
			generation: this.snapshot().state.compactionGeneration + 1,
		});
	}
	finish(
		status: import("./run-kernel-events.ts").RunTerminalStatus,
		summary?: string,
		source?: "structured" | "heuristic" | "runtime",
	): void {
		this.appendForActive({ type: "run_finished", status, summary, source });
	}
	recordTrajectory(
		kind: "run_start" | "agent_event" | "run_finish",
		operationId: string,
		payload: Record<string, unknown>,
		runId?: string,
	): void {
		const state = this.snapshot().state;
		if (!state.taskId || !state.runId || this.sessionId.startsWith("tui_"))
			return;
		this.append(
			{ type: "trajectory_recorded", kind, operationId, payload },
			{
				taskId: state.taskId,
				runId: runId ?? state.runId,
				operationId,
				leaseEpoch: state.leaseEpoch,
			},
		);
	}

	private appendForActive(event: RunKernelEvent): void {
		const state = this.snapshot().state;
		if (!state.taskId || !state.runId) return;
		this.append(event, {
			taskId: state.taskId,
			runId: state.runId,
			leaseEpoch: state.leaseEpoch,
		});
	}

	acquireLease(
		ownerId: string,
		options: {
			taskId: string;
			runId: string;
			ttlMs?: number;
			now?: number;
			force?: boolean;
		},
	): RunKernelLease {
		this.refreshIfChanged();
		const now = options.now ?? Date.now();
		const state = this.reduction.state;
		if (
			state.leaseOwnerId &&
			state.leaseOwnerId !== ownerId &&
			(state.leaseExpiresAt ?? 0) >= now &&
			options.force !== true
		)
			throw new Error(
				`Run Kernel lease is held by ${state.leaseOwnerId} until ${state.leaseExpiresAt}`,
			);
		const sameLiveOwner =
			state.leaseOwnerId === ownerId && (state.leaseExpiresAt ?? 0) >= now;
		const epoch = sameLiveOwner ? state.leaseEpoch : state.leaseEpoch + 1;
		const expiresAt = now + (options.ttlMs ?? 60 * 60_000);
		this.append(
			{ type: "lease_acquired", ownerId, expiresAt },
			{
				taskId: options.taskId,
				runId: options.runId,
				leaseEpoch: epoch,
				timestamp: now,
			},
		);
		return { ownerId, epoch, expiresAt };
	}

	append(
		event: RunKernelEvent,
		options: RunKernelAppendOptions,
	): RunEventEnvelope {
		if (this.sessionId.startsWith("tui_"))
			return this.appendUnlocked(event, options);
		return this.withFileLock(() => this.appendUnlocked(event, options));
	}

	private appendUnlocked(
		event: RunKernelEvent,
		options: RunKernelAppendOptions,
	): RunEventEnvelope {
		this.refreshIfChanged();
		const envelope: RunEventEnvelope = {
			schemaVersion: RUN_KERNEL_SCHEMA_VERSION,
			sequence: this.reduction.state.lastSequence + 1,
			eventId: randomUUID(),
			sessionId: this.sessionId,
			taskId: options.taskId,
			runId: options.runId,
			operationId: options.operationId,
			leaseEpoch: options.leaseEpoch ?? this.reduction.state.leaseEpoch,
			timestamp: options.timestamp ?? Date.now(),
			event,
		};
		const next = reduceRunKernel(this.reduction.state, envelope);
		if (next.violations.length) {
			throw new Error(
				`Run Kernel rejected ${event.type}: ${next.violations.map(item => item.message).join("; ")}`,
			);
		}
		if (!this.sessionId.startsWith("tui_")) {
			mkdirSync(path.dirname(this.filePath), { recursive: true });
			const line = `${JSON.stringify(envelope)}\n`;
			const existed = existsSync(this.filePath);
			const descriptor = openSync(this.filePath, "a");
			try {
				writeSync(descriptor, line, undefined, "utf8");
				fsyncSync(descriptor);
			} finally {
				closeSync(descriptor);
			}
			if (!existed) {
				try {
					const directory = openSync(path.dirname(this.filePath), "r");
					try {
						fsyncSync(directory);
					} finally {
						closeSync(directory);
					}
				} catch {
					// Some platforms cannot fsync directories; the file itself is durable.
				}
			}
			this.observedFileSize += Buffer.byteLength(line);
		}
		this.reduction = {
			state: next.state,
			violations: [...this.reduction.violations, ...next.violations],
		};
		return structuredClone(envelope);
	}

	private withFileLock<T>(operation: () => T): T {
		mkdirSync(path.dirname(this.filePath), { recursive: true });
		const lockPath = `${this.filePath}.lock`;
		let descriptor: number | undefined;
		for (let attempt = 0; attempt < 2; attempt++) {
			try {
				descriptor = openSync(lockPath, "wx");
				break;
			} catch (error) {
				const code = (error as NodeJS.ErrnoException).code;
				if (code !== "EEXIST") throw error;
				const stale = Date.now() - statSync(lockPath).mtimeMs > 30_000;
				if (!stale || attempt > 0)
					throw new Error(`Run Kernel ledger is busy: ${this.filePath}`);
				unlinkSync(lockPath);
			}
		}
		if (descriptor === undefined)
			throw new Error(`Unable to acquire Run Kernel lock: ${this.filePath}`);
		try {
			return operation();
		} finally {
			closeSync(descriptor);
			if (existsSync(lockPath)) unlinkSync(lockPath);
		}
	}

	doctor(): RunKernelDoctorReport {
		const read = this.read();
		let state = initialRunKernelState();
		const violations: RunKernelViolation[] = [];
		for (const event of read.events) {
			const next = reduceRunKernel(state, event);
			violations.push(...next.violations);
			state = next.state;
		}
		const incompleteOperations = Object.values(state.operations)
			.filter(operation => operation.status !== "committed")
			.map(operation => ({
				operationId: operation.operationId,
				toolCallId: operation.toolCallId,
				toolName: operation.toolName,
				arguments: operation.arguments,
				idempotencyKey: operation.idempotencyKey,
				receipt: operation.receipt,
				result: operation.result,
				status: operation.status,
				recovery: operation.recovery,
				recommendedAction:
					operation.status === "quarantined"
						? ("none" as const)
						: operation.status === "result_recorded"
							? ("reuse_result" as const)
							: operation.recovery === "pure" ||
									operation.recovery === "idempotent"
								? ("retry" as const)
								: operation.recovery === "receipt_recoverable"
									? ("reconcile_receipt" as const)
									: ("quarantine" as const),
			}));
		return {
			file: this.filePath,
			events: read.events.length,
			lastValidSequence: state.lastSequence,
			truncatedFinalRecord: read.truncatedFinalRecord,
			parseErrors: read.parseErrors,
			violations,
			incompleteOperations,
			orphanedSubagents: Object.values(state.subagents)
				.filter(child => child.status === "running")
				.map(child => ({
					agentId: child.agentId,
					agent: child.agent,
					task: child.task,
					lastEventType: child.lastEventType,
				})),
		};
	}

	private reload(): void {
		let state = initialRunKernelState();
		const violations: RunKernelViolation[] = [];
		for (const event of this.read().events) {
			const next = reduceRunKernel(state, event);
			violations.push(...next.violations);
			state = next.state;
		}
		this.reduction = { state, violations };
		this.observedFileSize = existsSync(this.filePath)
			? statSync(this.filePath).size
			: 0;
	}

	private refreshIfChanged(): void {
		if (this.sessionId.startsWith("tui_")) return;
		const size = existsSync(this.filePath) ? statSync(this.filePath).size : 0;
		if (size !== this.observedFileSize) this.reload();
	}

	private read(): {
		events: RunEventEnvelope[];
		truncatedFinalRecord: boolean;
		parseErrors: Array<{ line: number; message: string }>;
	} {
		if (!existsSync(this.filePath))
			return { events: [], truncatedFinalRecord: false, parseErrors: [] };
		const lines = readFileSync(this.filePath, "utf8").split("\n");
		const events: RunEventEnvelope[] = [];
		const parseErrors: Array<{ line: number; message: string }> = [];
		let truncatedFinalRecord = false;
		for (let index = 0; index < lines.length; index++) {
			const line = lines[index]?.trim();
			if (!line) continue;
			try {
				const parsed: unknown = JSON.parse(line);
				if (!isRunEventEnvelope(parsed))
					parseErrors.push({
						line: index + 1,
						message: "invalid event envelope",
					});
				else events.push(parsed);
			} catch (error) {
				const isFinalContent = lines
					.slice(index + 1)
					.every(item => !item.trim());
				if (isFinalContent) truncatedFinalRecord = true;
				else
					parseErrors.push({
						line: index + 1,
						message: error instanceof Error ? error.message : String(error),
					});
			}
		}
		return { events, truncatedFinalRecord, parseErrors };
	}
}
