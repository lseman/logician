// ── Per-run task state ──────────────────────────────────────────────────────
// Tools live as process-wide objects, but their mutable state must not. Async
// context keeps todo/task_status isolated across concurrent parent and child
// agent loops without changing the public Tool execution API.

import { AsyncLocalStorage } from "node:async_hooks";
import type { Task } from "./todo-state.ts";
import type { TaskStatusRecord } from "./task-status-state.ts";

export interface RunTaskState {
	tasks: Task[];
	nextTaskId: number;
	taskStatus: TaskStatusRecord | null;
}

const storage = new AsyncLocalStorage<RunTaskState>();
const fallback: RunTaskState = {
	tasks: [],
	nextTaskId: 1,
	taskStatus: null,
};

export function currentRunTaskState(): RunTaskState {
	return storage.getStore() ?? fallback;
}

export function runWithTaskState<T>(fn: () => Promise<T>): Promise<T> {
	return storage.run(
		{ tasks: [], nextTaskId: 1, taskStatus: null },
		fn,
	);
}
