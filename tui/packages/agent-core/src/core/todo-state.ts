// ── todo state ────────────────────────────────────────────────────────────
// Shared task-list store. Core reads it directly (builtin-hooks nudges the
// model while tasks remain); the todo Tool (defined in @logician/agent-capabilities)
// owns the mutations. Lives in core so both sides can depend on one copy
// without a circular package dependency.

export type TaskStatus = "pending" | "in_progress" | "completed" | "deleted";

export interface Task {
	id: number;
	subject: string;
	description?: string;
	activeForm?: string;
	status: TaskStatus;
	blockedBy?: number[];
	owner?: string;
	metadata?: Record<string, unknown>;
}

let tasks: Task[] = [];
let nextId = 1;
const listeners = new Set<(tasks: Task[]) => void>();

function snapshotTasks(): Task[] {
	return tasks.map((task) => ({
		...task,
		blockedBy: task.blockedBy ? [...task.blockedBy] : undefined,
		metadata: task.metadata ? { ...task.metadata } : undefined,
	}));
}

export function getTasks(): Task[] {
	return snapshotTasks();
}

export function onTodosChanged(cb: (tasks: Task[]) => void): () => void {
	listeners.add(cb);
	return () => listeners.delete(cb);
}

/** Read the live (mutable) task array. For use by the todo Tool's action handlers only. */
export function getTasksMutable(): Task[] {
	return tasks;
}

export function allocateTaskId(): number {
	return nextId++;
}

export function replaceTasks(next: Task[]): void {
	tasks = next;
	nextId = 1;
}

export function notifyTodosChanged(): void {
	const snapshot = snapshotTasks();
	for (const cb of listeners) cb(snapshot);
}
