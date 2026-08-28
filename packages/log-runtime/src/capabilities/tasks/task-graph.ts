import type { Task } from "./state.ts";

export function hasDependencyCycle(
	tasks: readonly Task[],
	startTaskId: number,
): boolean {
	const byId = new Map(tasks.map(task => [task.id, task]));
	const visited = new Set<number>();
	const active = new Set<number>();

	const visit = (taskId: number): boolean => {
		if (active.has(taskId)) return true;
		if (visited.has(taskId)) return false;
		visited.add(taskId);
		active.add(taskId);
		for (const dependencyId of byId.get(taskId)?.blockedBy ?? []) {
			if (visit(dependencyId)) return true;
		}
		active.delete(taskId);
		return false;
	};

	return visit(startTaskId);
}

/** Dependents whose complete blocker set is now satisfied. */
export function resolvedDependents(
	tasks: readonly Task[],
	completedTaskId: number,
): Task[] {
	const byId = new Map(tasks.map(task => [task.id, task]));
	return tasks.filter(
		task =>
			task.status === "in_progress" &&
			task.blockedBy?.includes(completedTaskId) === true &&
			task.blockedBy.every(
				dependencyId => byId.get(dependencyId)?.status === "completed",
			),
	);
}
