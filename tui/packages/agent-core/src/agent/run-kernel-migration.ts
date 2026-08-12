import { existsSync, mkdirSync, readFileSync, renameSync } from "node:fs";
import path from "node:path";
import type { RunKernel } from "./run-kernel.ts";
import type { RunTerminalStatus } from "./run-kernel-events.ts";
import type { ExplicitTaskState } from "./tasks/task-state-controller.ts";

interface LegacyRunProjection {
	runId: string;
	rootPrompt: string;
	createdAt: number;
	status: string;
	continuationRuns: number;
	lastProgressFingerprint: string;
	lastCause: string;
	taskState?: ExplicitTaskState;
	outcome?: {
		status: RunTerminalStatus;
		summary?: string;
		source?: "structured" | "heuristic" | "runtime";
	};
	terminalReason?: string;
	compactionGeneration: number;
}

interface LegacyTrajectoryEntry {
	version: 1;
	timestamp: number;
	runId: string;
	operationId: string;
	kind: "run_start" | "agent_event" | "run_finish";
	payload: Record<string, unknown>;
}

function isLegacyTrajectoryEntry(
	entry: Record<string, unknown>,
): entry is Record<string, unknown> & LegacyTrajectoryEntry {
	return (
		entry.version === 1 &&
		typeof entry.runId === "string" &&
		typeof entry.operationId === "string" &&
		["run_start", "agent_event", "run_finish"].includes(String(entry.kind)) &&
		Boolean(entry.payload) &&
		typeof entry.payload === "object" &&
		!Array.isArray(entry.payload) &&
		typeof entry.timestamp === "number"
	);
}

function safeId(value: string): string {
	return value.replace(/[^a-zA-Z0-9._-]/g, "_");
}

function records(file: string): Record<string, unknown>[] {
	if (!existsSync(file)) return [];
	const result: Record<string, unknown>[] = [];
	for (const line of readFileSync(file, "utf8").split("\n")) {
		if (!line.trim()) continue;
		try {
			const value: unknown = JSON.parse(line);
			if (value && typeof value === "object" && !Array.isArray(value))
				result.push(value as Record<string, unknown>);
		} catch {
			// A torn tail does not invalidate the replayable prefix.
		}
	}
	return result;
}

function loadLegacyRun(
	file: string,
	sessionId: string,
): LegacyRunProjection | undefined {
	let state: LegacyRunProjection | undefined;
	for (const entry of records(file)) {
		if (entry.version !== 1 || entry.sessionId !== sessionId) continue;
		const event = entry.event as Record<string, unknown> | undefined;
		if (!event || typeof event.type !== "string") continue;
		if (event.type === "run_started") {
			const initial = event.state as LegacyRunProjection | undefined;
			if (initial?.runId && typeof initial.createdAt === "number")
				state = structuredClone(initial);
			continue;
		}
		if (!state || entry.runId !== state.runId) continue;
		if (event.type === "continuation_requested") {
			state.continuationRuns++;
			state.lastCause = String(event.cause ?? "legacy_import");
			const fingerprint = String(event.progressFingerprint ?? "");
			if (fingerprint) state.lastProgressFingerprint = fingerprint;
		} else if (event.type === "task_state_updated")
			state.taskState = event.state as ExplicitTaskState;
		else if (event.type === "compaction_committed")
			state.compactionGeneration++;
		else if (event.type === "run_outcome")
			state.outcome = event.outcome as LegacyRunProjection["outcome"];
		else if (event.type === "run_paused") {
			state.status = "paused";
			state.terminalReason = String(event.reason ?? "paused");
		} else if (event.type === "run_failed") {
			state.status = "failed";
			state.terminalReason =
				typeof event.reason === "string" ? event.reason : undefined;
		}
	}
	return state;
}

function archive(cwd: string, source: string): void {
	if (!existsSync(source)) return;
	const directory = path.join(cwd, ".logician", "migrations", "v1-archive");
	mkdirSync(directory, { recursive: true });
	const target = path.join(
		directory,
		path.basename(path.dirname(source)),
		path.basename(source),
	);
	mkdirSync(path.dirname(target), { recursive: true });
	if (!existsSync(target)) renameSync(source, target);
}

/** One-time, recoverable import of pre-kernel execution journals. */
export function migrateLegacyRunData(
	kernel: RunKernel,
	cwd: string,
	sessionId: string,
): boolean {
	if (sessionId.startsWith("tui_") || kernel.snapshot().state.lastSequence > 0)
		return false;
	const id = safeId(sessionId);
	const runFile = path.join(cwd, ".logician", "runtime", `${id}.jsonl`);
	const trajectoryFile = path.join(
		cwd,
		".logician",
		"trajectories",
		`${id}.jsonl`,
	);
	const legacy = loadLegacyRun(runFile, sessionId);
	if (!legacy) return false;
	const options = { taskId: legacy.runId, runId: legacy.runId, leaseEpoch: 1 };
	kernel.append(
		{
			type: "task_started",
			rootPrompt: legacy.rootPrompt,
			createdAt: legacy.createdAt,
			progressFingerprint: "",
		},
		options,
	);
	kernel.append({ type: "run_started", cause: "resume" }, options);
	for (let index = 0; index < legacy.continuationRuns; index++)
		kernel.append(
			{
				type: "continuation_requested",
				cause:
					index === legacy.continuationRuns - 1
						? legacy.lastCause
						: "legacy_import",
				progressFingerprint:
					index === legacy.continuationRuns - 1
						? legacy.lastProgressFingerprint
						: "",
			},
			options,
		);
	if (legacy.taskState)
		kernel.append(
			{ type: "task_state_updated", state: legacy.taskState },
			options,
		);
	for (
		let generation = 1;
		generation <= legacy.compactionGeneration;
		generation++
	)
		kernel.append({ type: "compaction_committed", generation }, options);
	const trajectory = records(trajectoryFile).filter(isLegacyTrajectoryEntry);
	for (const entry of trajectory)
		kernel.append(
			{
				type: "trajectory_recorded",
				kind: entry.kind,
				operationId: entry.operationId,
				payload: entry.payload,
			},
			{
				...options,
				runId: entry.runId,
				operationId: entry.operationId,
				timestamp: entry.timestamp,
			},
		);
	if (legacy.outcome)
		kernel.append(
			{
				type: "run_finished",
				status: legacy.outcome.status,
				summary: legacy.outcome.summary,
				source: legacy.outcome.source,
			},
			options,
		);
	else if (legacy.status === "failed" || legacy.status === "cancelled")
		kernel.append(
			{
				type: "run_finished",
				status: legacy.status,
				summary: legacy.terminalReason,
				source: "runtime",
			},
			options,
		);
	else if (legacy.status === "paused")
		kernel.append(
			{
				type: "run_finished",
				status: "blocked",
				summary: legacy.terminalReason,
				source: "runtime",
			},
			options,
		);
	archive(cwd, runFile);
	archive(cwd, trajectoryFile);
	return true;
}
