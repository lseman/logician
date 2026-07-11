// ── task_status state ────────────────────────────────────────────────────────
// Module-level record of the structured end-of-task declaration. The loop's
// continuation logic reads this directly to decide when a run has ended
// cleanly, so it lives in core rather than alongside the task_status Tool
// (which is in @logician/agent-capabilities).

export interface TaskStatusRecord {
	status: "done" | "blocked" | "needs_input" | "failed";
	summary: string;
	ts: number;
}

let current: TaskStatusRecord | null = null;

/** The status recorded this run, or null. Reset at each run start. */
export function getTaskStatus(): TaskStatusRecord | null {
	return current;
}

export function resetTaskStatus(): void {
	current = null;
}

export function recordTaskStatus(record: TaskStatusRecord): void {
	current = record;
}
