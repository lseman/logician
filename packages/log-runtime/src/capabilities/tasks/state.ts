// Shared todo state. The todo tool owns mutations; core observes the list to
// decide whether an optional continuation nudge is useful.

/** Status values for phased todo tasks. */
export type TaskStatus = "pending" | "in_progress" | "completed" | "abandoned";

/** A single task in the phased todo list. */
export interface Task {
	content: string;
	status: TaskStatus;
	blocker?: string;
}

/** A named phase grouping tasks together. */
export interface TaskPhase {
	name: string;
	tasks: Task[];
}

let phases: TaskPhase[] = [];
let nextTaskId = 1;
const listeners = new Set<(phases: TaskPhase[]) => void>();

function clonePhases(source: readonly TaskPhase[] = phases): TaskPhase[] {
	return source.map(phase => ({
		name: phase.name,
		tasks: phase.tasks.map(task => ({ ...task })),
	}));
}

export function getTasks(): TaskPhase[] {
	return clonePhases();
}

export function onTodosChanged(cb: (phases: TaskPhase[]) => void): () => void {
	listeners.add(cb);
	return () => listeners.delete(cb);
}

export interface TaskMutationContext {
	phases: TaskPhase[];
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
	const draft = clonePhases();
	let draftNextTaskId = nextTaskId;
	const result = mutation({
		phases: draft,
		allocateId: () => draftNextTaskId++,
		resetIds: () => {
			draftNextTaskId = 1;
		},
	});
	if (!result.changed) return result.value;

	phases = draft;
	nextTaskId = draftNextTaskId;
	const snapshot = clonePhases();
	for (const listener of listeners) {
		try {
			listener(clonePhases(snapshot));
		} catch (error) {
			console.error("[todo] change listener failed:", error);
		}
	}
	return result.value;
}
