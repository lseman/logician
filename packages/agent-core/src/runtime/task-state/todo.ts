// ── todo state ────────────────────────────────────────────────────────────
// Shared task-list store. Core reads it directly (builtin-hooks nudges the
// model while tasks remain); the todo Tool (defined in @logician/agent-blocks)
// owns the mutations. Lives in core so both sides can depend on one copy
// without a circular package dependency.

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

import { currentRunTaskState } from "./context.ts";

const listeners = new Set<(tasks: Task[]) => void>();

function snapshotTasks(): Task[] {
	return currentRunTaskState().tasks.map(task => ({
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
	return currentRunTaskState().tasks;
}

export function allocateTaskId(): number {
	return currentRunTaskState().nextTaskId++;
}

export function replaceTasks(next: Task[]): void {
	const state = currentRunTaskState();
	state.tasks = next;
	state.nextTaskId = 1;
}

export function notifyTodosChanged(): void {
	const snapshot = snapshotTasks();
	for (const cb of listeners) cb(snapshot);
}
