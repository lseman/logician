// Shared todo state. The todo tool owns mutations; core observes the list to
// decide whether an optional continuation nudge is useful.

export type TaskStatus = "pending" | "in_progress" | "completed" | "deleted";
export type TaskState = TaskStatus;

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
let nextTaskId = 1;
const listeners = new Set<(tasks: Task[]) => void>();

function snapshotTasks(): Task[] {
	return tasks.map(task => ({
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

/** Mutable list reserved for the todo tool's action handlers. */
export function getTasksMutable(): Task[] {
	return tasks;
}

export function allocateTaskId(): number {
	return nextTaskId++;
}

export function replaceTasks(next: Task[]): void {
	tasks = next;
	nextTaskId = 1;
}

export function notifyTodosChanged(): void {
	const snapshot = snapshotTasks();
	for (const cb of listeners) cb(snapshot);
}
