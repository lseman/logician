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

function cloneTasks(source: readonly Task[] = tasks): Task[] {
	return source.map(task => ({
		...task,
		blockedBy: task.blockedBy ? [...task.blockedBy] : undefined,
		metadata: task.metadata ? { ...task.metadata } : undefined,
	}));
}

export function getTasks(): Task[] {
	return cloneTasks();
}

export function onTodosChanged(cb: (tasks: Task[]) => void): () => void {
	listeners.add(cb);
	return () => listeners.delete(cb);
}

export interface TaskMutationContext {
	tasks: Task[];
	allocateId: () => number;
	resetIds: () => void;
}

export interface TaskMutationResult<T> {
	value: T;
	changed: boolean;
}

/**
 * Apply one atomic task-state transaction. Failed/no-op actions discard the
 * draft and allocated ids; successful actions publish one immutable snapshot.
 */
export function mutateTasks<T>(
	mutation: (context: TaskMutationContext) => TaskMutationResult<T>,
): T {
	const draft = cloneTasks();
	let draftNextTaskId = nextTaskId;
	const result = mutation({
		tasks: draft,
		allocateId: () => draftNextTaskId++,
		resetIds: () => {
			draftNextTaskId = 1;
		},
	});
	if (!result.changed) return result.value;

	tasks = draft;
	nextTaskId = draftNextTaskId;
	const snapshot = cloneTasks();
	for (const listener of listeners) {
		try {
			listener(cloneTasks(snapshot));
		} catch (error) {
			console.error("[todo] change listener failed:", error);
		}
	}
	return result.value;
}
