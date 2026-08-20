// ── task_status state ────────────────────────────────────────────────────────
// Run-scoped record of the structured end-of-task declaration. The loop's
// continuation logic reads this directly to decide when a run has ended
// cleanly, so it lives in core rather than alongside the task_status Tool
// (which is in @logician/agent-blocks).

export interface TaskStatusRecord {
	status: "done" | "blocked" | "needs_input" | "failed";
	summary: string;
	ts: number;
}

import { currentRunTaskState } from "./run-task-state.ts";

/** The status recorded this run, or null. Reset at each run start. */
export function getTaskStatus(): TaskStatusRecord | null {
	return currentRunTaskState().taskStatus;
}

export function resetTaskStatus(): void {
	currentRunTaskState().taskStatus = null;
}

export function recordTaskStatus(record: TaskStatusRecord): void {
	currentRunTaskState().taskStatus = record;
}
