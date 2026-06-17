// ── todo tool — full task tracking (pi-compatible) ────────────────────────────
// Actions: create, update, list, get, delete, clear
// State machine: pending → in_progress → completed, plus deleted tombstone
// Dependencies: blockedBy with cycle detection and auto-resolve

import type { Tool } from "../../core/types.ts";

// ── Types ────────────────────────────────────────────────────────────────────

export type TaskStatus = "pending" | "in_progress" | "completed" | "deleted";
export type TaskAction =
	| "create"
	| "update"
	| "list"
	| "get"
	| "delete"
	| "clear";

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

// ── Store ────────────────────────────────────────────────────────────────────

let tasks: Task[] = [];
let nextId = 1;
const listeners = new Set<(tasks: Task[]) => void>();

export function getTasks(): Task[] {
	return tasks;
}

export function onTodosChanged(cb: (tasks: Task[]) => void): () => void {
	listeners.add(cb);
	return () => listeners.delete(cb);
}

// ── Validation ───────────────────────────────────────────────────────────────

const VALID_STATUSES: ReadonlySet<TaskStatus> = new Set([
	"pending",
	"in_progress",
	"completed",
	"deleted",
]);

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

// Cycle detection via DFS
function detectCycle(tasks: Task[], taskId: number): boolean {
	const visited = new Set<number>();
	const stack = new Set<number>();

	function dfs(id: number): boolean {
		if (stack.has(id)) return true;
		if (visited.has(id)) return false;
		visited.add(id);
		stack.add(id);
		const task = tasks.find((t) => t.id === id);
		if (task?.blockedBy) {
			for (const dep of task.blockedBy) {
				if (dfs(dep)) return true;
			}
		}
		stack.delete(id);
		return false;
	}

	return dfs(taskId);
}

// ── Actions ──────────────────────────────────────────────────────────────────

function actionCreate(params: Record<string, unknown>): string {
	const subject = String(params.subject ?? "").trim();
	if (!subject) return "Error: subject is required for create.";

	const task: Task = {
		id: nextId++,
		subject,
		status: "pending",
	};

	if (params.description) task.description = String(params.description);
	if (params.activeForm) task.activeForm = String(params.activeForm);
	if (params.owner) task.owner = String(params.owner);
	if (params.metadata)
		task.metadata = params.metadata as Record<string, unknown>;

	if (Array.isArray(params.blockedBy)) {
		const blockedBy: number[] = [];
		for (const depId of params.blockedBy) {
			if (typeof depId !== "number") continue;
			const dep = tasks.find((t) => t.id === depId);
			if (!dep) return `Error: blockedBy #${depId} not found.`;
			if (dep.status === "deleted")
				return `Error: blockedBy #${depId} is deleted.`;
			blockedBy.push(depId);
		}
		if (blockedBy.length) {
			task.blockedBy = blockedBy;
			// Reject if adding these deps would create a cycle
			if (detectCycle([...tasks, task], task.id)) {
				delete task.blockedBy;
				return "Error: blockedBy would create a dependency cycle.";
			}
		}
	}

	tasks.push(task);
	return `Created task #${task.id}: ${subject}`;
}

function actionUpdate(params: Record<string, unknown>): string {
	const id = params.id as number | undefined;
	if (id === undefined) return "Error: id is required for update.";
	const idx = tasks.findIndex((t) => t.id === id);
	if (idx === -1) return `Error: task #${id} not found.`;

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

	const task = tasks[idx];
	let newStatus = task.status;

	if (params.status !== undefined) {
		const to = params.status as TaskStatus;
		if (!VALID_STATUSES.has(to)) {
			return `Error: invalid status '${to}'. Use: pending, in_progress, completed, deleted.`;
		}
		if (!validTransition(task.status, to)) {
			return `Error: illegal status transition ${task.status} → ${to}.`;
		}
		newStatus = to;
	}

	let newBlockedBy: number[] = task.blockedBy ? [...task.blockedBy] : [];

	if (params.removeBlockedBy && Array.isArray(params.removeBlockedBy)) {
		const removeSet = new Set(params.removeBlockedBy);
		newBlockedBy = newBlockedBy.filter((d) => !removeSet.has(d));
	}

	if (params.addBlockedBy && Array.isArray(params.addBlockedBy)) {
		for (const depId of params.addBlockedBy) {
			if (typeof depId !== "number") continue;
			if (depId === id) return `Error: cannot block #${id} on itself.`;
			const dep = tasks.find((t) => t.id === depId);
			if (!dep) return `Error: addBlockedBy #${depId} not found.`;
			if (dep.status === "deleted")
				return `Error: addBlockedBy #${depId} is deleted.`;
			if (!newBlockedBy.includes(depId)) newBlockedBy.push(depId);
		}
		// Check cycles
		const candidate = { ...task, blockedBy: newBlockedBy };
		if (
			detectCycle(
				[...tasks.slice(0, idx), candidate, ...tasks.slice(idx + 1)],
				id,
			)
		) {
			return "Error: addBlockedBy would create a dependency cycle.";
		}
	}

	// Apply mutations
	if (params.subject !== undefined) task.subject = String(params.subject);
	if (params.description !== undefined)
		task.description = String(params.description);
	if (params.activeForm !== undefined)
		task.activeForm = String(params.activeForm);
	if (params.owner !== undefined) task.owner = String(params.owner);

	// Metadata merge (null removes key)
	if (params.metadata !== undefined) {
		const merged = { ...(task.metadata ?? {}) };
		for (const [k, v] of Object.entries(
			params.metadata as Record<string, unknown>,
		)) {
			if (v === null) delete merged[k];
			else merged[k] = v;
		}
		task.metadata = Object.keys(merged).length ? merged : undefined;
	}

	// Update status and blockedBy
	task.status = newStatus;
	if (newBlockedBy.length) task.blockedBy = newBlockedBy;
	else delete task.blockedBy;

	// Auto-complete dependent tasks: if this task was completed and some tasks
	// blocked by it have no other blockers, unblock them.
	const dependents = tasks.filter(
		(t) =>
			t.status === "in_progress" &&
			t.blockedBy?.includes(id) &&
			t.blockedBy.every((d) => {
				const dep = tasks.find((x) => x.id === d);
				return dep?.status === "completed" || dep?.id === id;
			}),
	);

	const lines: string[] = [];
	lines.push(`Updated task #${id}: ${task.subject}`);
	if (task.status !== newStatus && task.status !== newStatus) {
		lines.push(`  Status: ${task.status} → ${newStatus}`);
	}
	if (dependents.length) {
		for (const dep of dependents) {
			lines.push(`  → Unblocked #${dep.id}: ${dep.subject}`);
			dep.status = "pending";
		}
	}

	return lines.join("\n");
}

function actionList(params: Record<string, unknown>): string {
	let filtered = [...tasks];
	const status = params.status as TaskStatus | undefined;
	const includeDeleted = params.includeDeleted === true;

	if (status) {
		filtered = filtered.filter((t) => t.status === status);
	}
	if (!includeDeleted) {
		filtered = filtered.filter((t) => t.status !== "deleted");
	}

	if (filtered.length === 0) return "No tasks.";

	const groups: Record<TaskStatus, Task[]> = {
		pending: [],
		in_progress: [],
		completed: [],
		deleted: [],
	};
	for (const t of filtered) {
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
		groups.in_progress.forEach((t) => lines.push(fmt(t)));
	}
	if (groups.pending.length > 0) {
		lines.push("── Pending ──");
		groups.pending.forEach((t) => lines.push(fmt(t)));
	}
	if (groups.completed.length > 0 && includeDeleted) {
		lines.push("── Completed ──");
		groups.completed.forEach((t) => lines.push(fmt(t)));
	}
	if (groups.deleted.length > 0) {
		lines.push("── Deleted ──");
		groups.deleted.forEach((t) => lines.push(fmt(t)));
	}

	return lines.join("\n");
}

function actionGet(params: Record<string, unknown>): string {
	const id = params.id as number | undefined;
	if (id === undefined) return "Error: id is required for get.";
	const task = tasks.find((t) => t.id === id);
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
	const idx = tasks.findIndex((t) => t.id === id);
	if (idx === -1) return `Error: task #${id} not found.`;
	if (tasks[idx].status === "deleted")
		return `Error: task #${id} is already deleted.`;

	tasks[idx].status = "deleted";
	// Remove from any other task's blockedBy
	for (const t of tasks) {
		if (t.blockedBy?.includes(id)) {
			t.blockedBy = t.blockedBy.filter((d) => d !== id);
		}
	}

	return `Deleted task #${id}: ${tasks[idx].subject}`;
}

function actionClear(): string {
	const count = tasks.length;
	tasks = [];
	nextId = 1;
	return `Cleared ${count} task(s).`;
}

// ── Tool definition ──────────────────────────────────────────────────────────

const normalizeInput = (
	raw: unknown,
): { action?: TaskAction; params?: Record<string, unknown> } => {
	if (typeof raw === "string") {
		try {
			return JSON.parse(raw);
		} catch {
			return {};
		}
	}
	if (!raw || typeof raw !== "object") return {};
	return raw as Record<string, unknown>;
};

export const todo_tool: Tool = {
	readOnly: false,
	name: "todo",
	hookAliases: ["Todo"],
	description:
		"Manage a task list for tracking multi-step progress. Actions: create (new task), update (change status/fields/dependencies), list (all tasks, optionally filtered by status), get (single task details), delete (tombstone), clear (reset all). " +
		"Status: pending → in_progress → completed, plus deleted tombstone. " +
		"Use this to plan and track multi-step work like research, design, and implementation. " +
		"Use blockedBy to express dependencies (A is blocked by B). On create, pass blockedBy as the initial set. On update, use addBlockedBy / removeBlockedBy (additive merge — do not resend the full array). Cycles are rejected. " +
		"list hides tombstoned (deleted) tasks by default; pass includeDeleted:true to see them. Pass status to filter by a single status. " +
		"Subject must be short and imperative (e.g. 'Research existing tool'); description is for long-form detail. " +
		"activeForm is a present-continuous label shown while in_progress (e.g. 'writing tests').",
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

		// Notify listeners
		for (const cb of listeners) cb(tasks);
		return result;
	},
};
