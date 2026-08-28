// ── todo tool — full task tracking ────────────────────────────────────────────
// Actions: create, update, list, get, delete, clear
// State machine: pending → in_progress → completed, plus deleted tombstone
// Dependencies: blockedBy with cycle detection and auto-resolve

import type { Tool } from "@logician/log-core";
import {
	getTasks,
	mutateTasks,
	type Task,
	type TaskMutationResult,
	type TaskStatus,
} from "./state.ts";
import { hasDependencyCycle, resolvedDependents } from "./task-graph.ts";

export type { Task, TaskStatus } from "./state.ts";
export { getTasks, onTodosChanged } from "./state.ts";

export type TaskAction =
	| "create"
	| "update"
	| "list"
	| "get"
	| "delete"
	| "clear";

// ── Validation ───────────────────────────────────────────────────────────────

const VALID_STATUSES: ReadonlySet<TaskStatus> = new Set([
	"pending",
	"in_progress",
	"completed",
	"deleted",
]);
const ANSI_CSI_SEQUENCE = new RegExp(
	`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`,
	"g",
);

function validTransition(from: TaskStatus, to: TaskStatus): boolean {
	// pending → in_progress, pending → completed (finish without starting)
	// in_progress → completed, in_progress → pending (pause)
	// completed → pending (restart)
	// deleted can come from any status
	if (to === "deleted") return true;
	if (from === "pending" && to === "in_progress") return true;
	if (from === "pending" && to === "completed") return true;
	if (from === "in_progress" && to === "completed") return true;
	if (from === "in_progress" && to === "pending") return true;
	if (from === "completed" && to === "pending") return true;
	return false;
}

function unchanged(value: string): TaskMutationResult<string> {
	return { value, changed: false };
}

function changed(value: string): TaskMutationResult<string> {
	return { value, changed: true };
}

// ── Actions ──────────────────────────────────────────────────────────────────

function stripNewlines(s: string): string {
	return s
		.replace(ANSI_CSI_SEQUENCE, "")
		.replace(/[\p{Cc}\p{Cf}]/gu, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function actionCreate(params: Record<string, unknown>): string {
	const subject = stripNewlines(String(params.subject ?? ""));
	if (!subject) return "Error: subject is required for create.";
	if (
		params.metadata !== undefined &&
		(!params.metadata ||
			typeof params.metadata !== "object" ||
			Array.isArray(params.metadata))
	) {
		return "Error: metadata must be an object.";
	}

	return mutateTasks(({ tasks, allocateId }) => {
		const existing = tasks.find(
			task =>
				task.status !== "deleted" &&
				task.subject.toLowerCase() === subject.toLowerCase(),
		);
		if (existing) {
			return unchanged(
				`Task already on the list: #${existing.id} - ${existing.subject} (${existing.status})`,
			);
		}

		const task: Task = {
			id: allocateId(),
			subject,
			status: "pending",
		};
		if (params.description) task.description = String(params.description);
		if (params.activeForm)
			task.activeForm = stripNewlines(String(params.activeForm));
		if (params.owner) task.owner = stripNewlines(String(params.owner));
		if (params.metadata)
			task.metadata = { ...(params.metadata as Record<string, unknown>) };

		if (Array.isArray(params.blockedBy)) {
			const blockedBy: number[] = [];
			for (const dependencyId of params.blockedBy) {
				if (typeof dependencyId !== "number") continue;
				const dependency = tasks.find(item => item.id === dependencyId);
				if (!dependency)
					return unchanged(`Error: blockedBy #${dependencyId} not found.`);
				if (dependency.status === "deleted")
					return unchanged(`Error: blockedBy #${dependencyId} is deleted.`);
				if (!blockedBy.includes(dependencyId)) blockedBy.push(dependencyId);
			}
			if (blockedBy.length) task.blockedBy = blockedBy;
			if (hasDependencyCycle([...tasks, task], task.id)) {
				return unchanged("Error: blockedBy would create a dependency cycle.");
			}
		}

		tasks.push(task);
		return changed(`Created task #${task.id}: ${subject}`);
	});
}

function actionUpdate(params: Record<string, unknown>): string {
	const id = params.id as number | undefined;
	if (id === undefined) return "Error: id is required for update.";
	const hasMutation =
		params.subject !== undefined ||
		params.description !== undefined ||
		params.activeForm !== undefined ||
		params.status !== undefined ||
		params.owner !== undefined ||
		params.metadata !== undefined ||
		params.addBlockedBy ||
		params.removeBlockedBy;
	if (!hasMutation) return "Error: update requires at least one mutable field.";
	if (
		params.metadata !== undefined &&
		(!params.metadata ||
			typeof params.metadata !== "object" ||
			Array.isArray(params.metadata))
	) {
		return "Error: metadata must be an object.";
	}

	return mutateTasks(({ tasks }) => {
		const index = tasks.findIndex(task => task.id === id);
		if (index === -1) return unchanged(`Error: task #${id} not found.`);
		const task = tasks[index];
		let newStatus = task.status;

		if (params.status !== undefined) {
			const targetStatus = params.status as TaskStatus;
			if (!VALID_STATUSES.has(targetStatus)) {
				return unchanged(
					`Error: invalid status '${targetStatus}'. Use: pending, in_progress, completed, deleted.`,
				);
			}
			if (!validTransition(task.status, targetStatus)) {
				return unchanged(
					`Error: illegal status transition ${task.status} → ${targetStatus}.`,
				);
			}
			newStatus = targetStatus;
		}

		let newBlockedBy = task.blockedBy ? [...task.blockedBy] : [];
		if (Array.isArray(params.removeBlockedBy)) {
			const remove = new Set(params.removeBlockedBy);
			newBlockedBy = newBlockedBy.filter(
				dependencyId => !remove.has(dependencyId),
			);
		}
		if (Array.isArray(params.addBlockedBy)) {
			for (const dependencyId of params.addBlockedBy) {
				if (typeof dependencyId !== "number") continue;
				if (dependencyId === id)
					return unchanged(`Error: cannot block #${id} on itself.`);
				const dependency = tasks.find(item => item.id === dependencyId);
				if (!dependency)
					return unchanged(`Error: addBlockedBy #${dependencyId} not found.`);
				if (dependency.status === "deleted")
					return unchanged(`Error: addBlockedBy #${dependencyId} is deleted.`);
				if (!newBlockedBy.includes(dependencyId))
					newBlockedBy.push(dependencyId);
			}
			const candidate = { ...task, blockedBy: newBlockedBy };
			const candidateTasks = [...tasks];
			candidateTasks[index] = candidate;
			if (hasDependencyCycle(candidateTasks, id)) {
				return unchanged(
					"Error: addBlockedBy would create a dependency cycle.",
				);
			}
		}

		if (params.subject !== undefined) {
			const newSubject = stripNewlines(String(params.subject));
			if (!newSubject) return unchanged("Error: subject cannot be empty.");
			task.subject = newSubject;
		}
		if (params.description !== undefined)
			task.description = String(params.description);
		if (params.activeForm !== undefined)
			task.activeForm = stripNewlines(String(params.activeForm));
		if (params.owner !== undefined)
			task.owner = stripNewlines(String(params.owner));

		if (params.metadata !== undefined) {
			const merged = { ...(task.metadata ?? {}) };
			for (const [key, value] of Object.entries(
				params.metadata as Record<string, unknown>,
			)) {
				if (value === null) delete merged[key];
				else merged[key] = value;
			}
			task.metadata = Object.keys(merged).length ? merged : undefined;
		}

		const statusWas = task.status;
		task.status = newStatus;
		task.blockedBy = newBlockedBy.length ? newBlockedBy : undefined;
		const dependents =
			statusWas !== "completed" && newStatus === "completed"
				? resolvedDependents(tasks, id)
				: [];

		const lines = [`Updated task #${id}: ${task.subject}`];
		if (statusWas !== newStatus)
			lines.push(`  Status: ${statusWas} → ${newStatus}`);
		for (const dependent of dependents) {
			lines.push(`  → Unblocked #${dependent.id}: ${dependent.subject}`);
			dependent.status = "pending";
		}

		return changed(lines.join("\n"));
	});
}

function actionList(params: Record<string, unknown>): string {
	const tasks = getTasks();
	let filtered = [...tasks];
	const status = params.status as TaskStatus | undefined;
	const includeDeleted = params.includeDeleted === true;

	if (status) {
		filtered = filtered.filter(t => t.status === status);
	}
	if (!includeDeleted) {
		filtered = filtered.filter(t => t.status !== "deleted");
	}

	if (filtered.length === 0) return "No tasks.";

	const groups: Record<TaskStatus, Task[]> = {
		pending: [],
		in_progress: [],
		completed: [],
		deleted: [],
	};
	for (const t of filtered) {
		if (!t.subject?.trim()) continue;
		groups[t.status].push(t);
	}

	const lines: string[] = [];
	const fmt = (task: Task) => {
		const mark = {
			pending: "○",
			in_progress: "◐",
			completed: "✓",
			deleted: "✗",
		}[task.status];
		const dep = task.blockedBy ? ` [blocks: #${task.blockedBy.join(",")}]` : "";
		const active = task.activeForm ? ` (${task.activeForm})` : "";
		return `  #${task.id} ${mark} ${task.subject}${active}${dep}`;
	};

	if (groups.in_progress.length > 0) {
		lines.push("── In Progress ──");
		for (const task of groups.in_progress) lines.push(fmt(task));
	}
	if (groups.pending.length > 0) {
		lines.push("── Pending ──");
		for (const task of groups.pending) lines.push(fmt(task));
	}
	if (groups.completed.length > 0 && includeDeleted) {
		lines.push("── Completed ──");
		for (const task of groups.completed) lines.push(fmt(task));
	}
	if (groups.deleted.length > 0) {
		lines.push("── Deleted ──");
		for (const task of groups.deleted) lines.push(fmt(task));
	}

	return lines.join("\n");
}

function actionGet(params: Record<string, unknown>): string {
	const tasks = getTasks();
	const id = params.id as number | undefined;
	if (id === undefined) return "Error: id is required for get.";
	const task = tasks.find(t => t.id === id);
	if (!task) return `Error: task #${id} not found.`;

	const lines: string[] = [];
	lines.push(`# ${task.subject}`);
	if (task.description) lines.push(`  desc: ${task.description}`);
	lines.push(`  status: ${task.status}`);
	if (task.activeForm) lines.push(`  active: ${task.activeForm}`);
	if (task.owner) lines.push(`  owner: ${task.owner}`);
	if (task.blockedBy?.length)
		lines.push(`  blockedBy: #${task.blockedBy.join(", ")}`);
	if (task.metadata) lines.push(`  metadata: ${JSON.stringify(task.metadata)}`);

	return lines.join("\n");
}

function actionDelete(params: Record<string, unknown>): string {
	const id = params.id as number | undefined;
	if (id === undefined) return "Error: id is required for delete.";
	return mutateTasks(({ tasks }) => {
		const task = tasks.find(item => item.id === id);
		if (!task) return unchanged(`Error: task #${id} not found.`);
		if (task.status === "deleted")
			return unchanged(`Error: task #${id} is already deleted.`);

		task.status = "deleted";
		for (const dependent of tasks) {
			if (dependent.blockedBy?.includes(id)) {
				dependent.blockedBy = dependent.blockedBy.filter(
					dependencyId => dependencyId !== id,
				);
				if (dependent.blockedBy.length === 0) dependent.blockedBy = undefined;
			}
		}
		return changed(`Deleted task #${id}: ${task.subject}`);
	});
}

function actionClear(): string {
	return mutateTasks(({ tasks, resetIds }) => {
		const count = tasks.length;
		tasks.length = 0;
		resetIds();
		return { value: `Cleared ${count} task(s).`, changed: count > 0 };
	});
}

// ── Tool definition ──────────────────────────────────────────────────────────

const normalizeInput = (
	raw: unknown,
): { action?: TaskAction; params?: Record<string, unknown> } => {
	if (typeof raw === "string") {
		try {
			return JSON.parse(raw);
		} catch (_e: unknown) {
			return {};
		}
	}
	if (!raw || typeof raw !== "object") return {};
	return raw as Record<string, unknown>;
};

export const todo_tool: Tool = {
	readOnly: false,
	executionMode: "sequential",
	name: "todo",
	label: "Todo",
	hookAliases: ["Todo"],
	description:
		"Manage a task list for tracking multi-step progress. Actions: create (new task), update (change status/fields/dependencies), list (all tasks, optionally filtered by status), get (single task details), delete (tombstone), clear (reset all). " +
		"Status: pending → in_progress → completed, plus deleted tombstone. " +
		"Use this to plan and track multi-step work like research, design, and implementation. " +
		"Use blockedBy to express dependencies (A is blocked by B). On create, pass blockedBy as the initial set. On update, use addBlockedBy / removeBlockedBy (additive merge — do not resend the full array). Cycles are rejected. " +
		"list hides tombstoned (deleted) tasks by default; pass includeDeleted:true to see them. Pass status to filter by a single status. " +
		"Subject must be short and imperative (e.g. 'Research existing tool'); description is for long-form detail. " +
		"activeForm is a present-continuous label shown while in_progress (e.g. 'writing tests').",
	promptSnippet: "Manage task list with status tracking and dependencies",
	promptGuidelines: [
		"Use todo to track multi-step progress; mark in_progress before work, completed immediately when done",
	],
	parameters: {
		type: "object",
		properties: {
			action: {
				type: "string",
				enum: ["create", "update", "list", "get", "delete", "clear"],
				description: "Action to perform",
			},
			subject: {
				type: "string",
				description: "Task subject line (required for create)",
			},
			description: {
				type: "string",
				description: "Long-form task description",
			},
			activeForm: {
				type: "string",
				description:
					"Present-continuous spinner label shown while status is in_progress",
			},
			status: {
				type: "string",
				enum: ["pending", "in_progress", "completed", "deleted"],
				description: "Target status (update) or list filter (list)",
			},
			blockedBy: {
				type: "array",
				items: { type: "number" },
				description: "Initial blockedBy ids (create only)",
			},
			addBlockedBy: {
				type: "array",
				items: { type: "number" },
				description:
					"Task ids to add to blockedBy (update only, additive merge)",
			},
			removeBlockedBy: {
				type: "array",
				items: { type: "number" },
				description:
					"Task ids to remove from blockedBy (update only, additive merge)",
			},
			owner: {
				type: "string",
				description: "Agent/owner assigned to this task",
			},
			metadata: {
				type: "object",
				description:
					"Arbitrary metadata; pass null value for a key to delete that key on update",
			},
			id: {
				type: "number",
				description: "Task id (required for update, get, delete)",
			},
			includeDeleted: {
				type: "boolean",
				description:
					"If true, list action returns deleted (tombstoned) tasks as well. Default: false.",
			},
		},
		required: ["action"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		const parsed = normalizeInput(raw);
		return parsed;
	},
	execute: async (args): Promise<string> => {
		const { action, ...params } = normalizeInput(args) as {
			action?: TaskAction;
		} & Record<string, unknown>;

		if (!action)
			return "Error: action is required. Use: create, update, list, get, delete, clear.";

		let result: string;
		switch (action) {
			case "create":
				result = actionCreate(params);
				break;
			case "update":
				result = actionUpdate(params);
				break;
			case "list":
				result = actionList(params);
				break;
			case "get":
				result = actionGet(params);
				break;
			case "delete":
				result = actionDelete(params);
				break;
			case "clear":
				result = actionClear();
				break;
			default:
				return `Error: unknown action '${action}'. Use: create, update, list, get, delete, clear.`;
		}

		return result;
	},
};
