// ── todo tool — phased task tracking ──────────────────────────────────────────
// Operations: init, start, done, rm, drop, block, unblock, append, view
// Phases group tasks; completion transitions track what changed per update.

import type { Tool } from "@logician/log-core";
/** Read-only task state supplied by an optional capability package. */
interface TaskLedger {
	snapshot(): readonly { id: number; subject: string; status: string }[];
}

import { getTasks, mutateTasks } from "./state.ts";
import type { Task, TaskPhase } from "./state.ts";

export { getTasks } from "./state.ts";
export type { Task, TaskPhase, TaskStatus } from "./state.ts";
export { onTodosChanged } from "./state.ts";

export type TodoOp =
	| "init"
	| "start"
	| "done"
	| "rm"
	| "drop"
	| "block"
	| "unblock"
	| "append"
	| "view";

const ANSI_CSI_SEQUENCE = new RegExp(
	`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`,
	"g",
);

function stripNewlines(s: string): string {
	return s
		.replace(ANSI_CSI_SEQUENCE, "")
		.replace(/[\p{Cc}\p{Cf}]/gu, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function findPhase(phases: TaskPhase[], name: string): TaskPhase | undefined {
	return phases.find(p => p.name === name);
}

function findTaskByContent(
	phases: TaskPhase[],
	content: string,
): { phase: TaskPhase; task: Task } | undefined {
	for (const phase of phases) {
		const task = phase.tasks.find(t => t.content === content);
		if (task) return { phase, task };
	}
	return undefined;
}

function opInit(entry: Record<string, unknown>): string {
	const items = entry.items as
		| Array<string | { phase: string; items: string[] }>
		| undefined;
	if (!items || !Array.isArray(items))
		return "Error: init requires 'items' array.";

	const errors: string[] = [];
	const newPhases: TaskPhase[] = [];
	const phaseNames = new Set<string>();
	const allTasks: Array<{ phaseName: string; content: string }> = [];

	for (const item of items) {
		if (typeof item === "string") {
			const name = stripNewlines(item);
			if (!name) continue;
			if (phaseNames.has(name)) {
				errors.push(`Duplicate phase '${name}'.`);
				continue;
			}
			phaseNames.add(name);
			newPhases.push({ name, tasks: [] });
		} else if (typeof item === "object" && item && "phase" in item) {
			const name = stripNewlines(String(item.phase));
			if (!name) continue;
			if (phaseNames.has(name)) {
				errors.push(`Duplicate phase '${name}'.`);
				continue;
			}
			phaseNames.add(name);
			const taskNames = Array.isArray(item.items) ? item.items : [];
			const tasks: Task[] = [];
			for (const t of taskNames) {
				const content = stripNewlines(String(t));
				if (!content) continue;
				tasks.push({ content, status: "pending" });
				allTasks.push({ phaseName: name, content });
			}
			newPhases.push({ name, tasks });
		}
	}

	if (errors.length > 0) return errors.join("\n");

	return mutateTasks(ctx => {
		ctx.phases = newPhases;
		return {
			value: `Created ${newPhases.length} phase(s): ${newPhases
				.map(p => p.name)
				.join(", ")} (${allTasks.length} tasks)`,
			changed: true,
		};
	});
}

function opStart(entry: Record<string, unknown>): string {
	const phase = stripNewlines(String(entry.phase || ""));
	const task = stripNewlines(String(entry.task || ""));
	if (!phase || !task)
		return "Error: both 'phase' and 'task' are required for start.";

	return mutateTasks(ctx => {
		const result = findTaskByContent(ctx.phases, task);
		if (!result)
			return {
				value: `Error: task '${task}' not found.`,
				changed: false,
			};
		const { task: targetTask } = result;
		if (targetTask.status === "in_progress")
			return { value: "Task already in progress.", changed: false };
		if (targetTask.status === "completed")
			return { value: "Task already completed.", changed: false };
		targetTask.status = "in_progress";
		return {
			value: `Started '${targetTask.content}' in ${result.phase.name}.`,
			changed: true,
		};
	});
}

function opDone(entry: Record<string, unknown>): string {
	const task = stripNewlines(String(entry.task || ""));
	if (!task) return "Error: 'task' is required for done.";

	return mutateTasks(ctx => {
		const result = findTaskByContent(ctx.phases, task);
		if (!result)
			return { value: `Error: task '${task}' not found.`, changed: false };
		const { task: targetTask, phase } = result;
		targetTask.status = "completed";
		return {
			value: `Completed '${targetTask.content}' in ${phase.name}.`,
			changed: true,
		};
	});
}

function opRm(entry: Record<string, unknown>): string {
	const phase = stripNewlines(String(entry.phase || ""));
	const task = stripNewlines(String(entry.task || ""));
	if (!phase || !task)
		return "Error: both 'phase' and 'task' are required for rm.";

	return mutateTasks(ctx => {
		const target = findPhase(ctx.phases, phase);
		if (!target)
			return {
				value: `Error: phase '${phase}' not found.`,
				changed: false,
			};
		const idx = target.tasks.findIndex(t => t.content === task);
		if (idx === -1)
			return {
				value: `Error: task '${task}' not in phase '${phase}'.`,
				changed: false,
			};
		target.tasks.splice(idx, 1);
		return { value: `Removed '${task}' from ${phase}.`, changed: true };
	});
}

function opDrop(entry: Record<string, unknown>): string {
	const phase = stripNewlines(String(entry.phase || ""));
	if (!phase) return "Error: 'phase' is required for drop.";

	return mutateTasks(ctx => {
		const idx = ctx.phases.findIndex(p => p.name === phase);
		if (idx === -1)
			return {
				value: `Error: phase '${phase}' not found.`,
				changed: false,
			};
		const count = ctx.phases[idx].tasks.length;
		ctx.phases.splice(idx, 1);
		return {
			value: `Dropped phase '${phase}' (${count} task(s)).`,
			changed: true,
		};
	});
}

function opBlock(entry: Record<string, unknown>): string {
	const phase = stripNewlines(String(entry.phase || ""));
	const task = stripNewlines(String(entry.task || ""));
	const blocker = stripNewlines(String(entry.blocker || ""));
	if (!phase || !task || !blocker)
		return "Error: 'phase', 'task', and 'blocker' are required for block.";

	return mutateTasks(ctx => {
		const result = findTaskByContent(ctx.phases, task);
		if (!result)
			return { value: `Error: task '${task}' not found.`, changed: false };
		const { task: targetTask } = result;
		targetTask.blocker = blocker;
		targetTask.status = "pending";
		return {
			value: `Blocked '${task}' by '${blocker}'.`,
			changed: true,
		};
	});
}

function opUnblock(entry: Record<string, unknown>): string {
	const task = stripNewlines(String(entry.task || ""));
	if (!task) return "Error: 'task' is required for unblock.";

	return mutateTasks(ctx => {
		const result = findTaskByContent(ctx.phases, task);
		if (!result)
			return { value: `Error: task '${task}' not found.`, changed: false };
		const { task: targetTask } = result;
		targetTask.blocker = undefined;
		return { value: `Unblocked '${task}'.`, changed: true };
	});
}

function opAppend(entry: Record<string, unknown>): string {
	const phase = stripNewlines(String(entry.phase || ""));
	const taskList = entry.tasks as string[] | undefined;
	if (!phase || !taskList || !Array.isArray(taskList))
		return "Error: 'phase' and 'tasks' (array) are required for append.";

	const newTasks: Task[] = [];
	for (const content of taskList) {
		const cleaned = stripNewlines(String(content));
		if (!cleaned) continue;
		newTasks.push({ content: cleaned, status: "pending" });
	}
	if (newTasks.length === 0) return "No valid tasks to append.";

	return mutateTasks(ctx => {
		const existingPhases = ctx.phases;
		const target = findPhase(existingPhases, phase);
		if (!target) {
			existingPhases.push({ name: phase, tasks: newTasks });
			return {
				value: `Created phase '${phase}' and added ${newTasks.length} task(s).`,
				changed: true,
			};
		}
		let added = 0;
		for (const t of newTasks) {
			if (!target.tasks.find(ex => ex.content === t.content)) {
				target.tasks.push(t);
				added++;
			}
		}
		return {
			value: `Added ${added} task(s) to ${phase}.`,
			changed: added > 0,
		};
	});
}

function opView(): string {
	const phases = getTasks();
	if (phases.length === 0) return "No tasks.";

	const lines: string[] = [];
	for (const phase of phases) {
		const done = phase.tasks.filter(
			t => t.status === "completed",
		).length;
		const active = phase.tasks.filter(
			t => t.status === "in_progress",
		).length;
		lines.push(
			`Phase: ${phase.name} (${done} done, ${active} active, ${phase.tasks.length} total)`,
		);
		for (const t of phase.tasks) {
			const mark: Record<string, string> = {
				pending: "○",
				in_progress: "→",
				completed: "✓",
				abandoned: "✕",
			};
			const dep = t.blocker
				? ` [blocked by: ${t.blocker}]`
				: "";
			lines.push(`  ${mark[t.status]} ${t.content}${dep}`);
		}
		lines.push("");
	}
	return lines.join("\n");
}

// ── Tool definition ──────────────────────────────────────────────────────────

export const todo_tool: Tool = {
	readOnly: false,
	executionMode: "sequential",
	name: "todo",
	label: "Todo",
	hookAliases: ["Todo"],
	description:
		"Manage a phased task list for tracking multi-step progress. " +
		"Operations: init (create phases with tasks), start (mark in_progress), done (mark completed), " +
		"rm (remove task), drop (remove phase), block/unblock (set/clear blocker), append (add tasks to phase), view (list all). " +
		"Init takes { phase: string, items: string[] } entries or just phase name strings. " +
		"Use phases to organize work (e.g., 'Foundation', 'Implementation', 'Verification').",
	promptSnippet: "Manage phased task list with status tracking",
	promptGuidelines: [
		"Use todo to track multi-step progress; mark in_progress before work, completed immediately when done",
	],
	parameters: {
		type: "object",
		properties: {
			op: {
				type: "string",
				enum: [
					"init",
					"start",
					"done",
					"rm",
					"drop",
					"block",
					"unblock",
					"append",
					"view",
				],
				description: "Operation to apply",
			},
			phase: {
				type: "string",
				description:
					"Phase name (required for most operations)",
			},
			task: {
				type: "string",
				description:
					"Task content (required for start/done/rm/block/unblock)",
			},
			blocker: {
				type: "string",
				description: "Blocker description (required for block)",
			},
			tasks: {
				type: "array",
				items: { type: "string" },
				description:
					"Task content strings (for append or init items)",
			},
			items: {
				type: "array",
				description:
					"Init items: phase names or { phase, items } objects",
			},
		},
		required: ["op"],
	},
	prepareArguments: (raw): Record<string, unknown> => {
		if (typeof raw === "string") {
			try {
				return JSON.parse(raw);
			} catch {
				return {};
			}
		}
		if (!raw || typeof raw !== "object") return {};
		return raw as Record<string, unknown>;
	},
	execute: async (args): Promise<string> => {
		const entry = (raw => {
			if (typeof raw === "string") {
				try { return JSON.parse(raw); } catch { return {}; }
			}
			if (!raw || typeof raw !== "object") return {};
			return raw as Record<string, unknown>;
		})(args) as Record<string, unknown>;
		const op = (entry.op as TodoOp) || (entry.action as TodoOp);

		if (!op)
			return "Error: 'op' is required. Use: init, start, done, rm, drop, block, unblock, append, view.";

		switch (op) {
			case "init":
				return opInit(entry);
			case "start":
				return opStart(entry);
			case "done":
				return opDone(entry);
			case "rm":
				return opRm(entry);
			case "drop":
				return opDrop(entry);
			case "block":
				return opBlock(entry);
			case "unblock":
				return opUnblock(entry);
			case "append":
				return opAppend(entry);
			case "view":
				return opView();
			default:
				return `Error: unknown op '${op}'. Use: init, start, done, rm, drop, block, unblock, append, view.`;
		}
	},
};

// ── TaskLedger adapter ────────────────────────────────────────────────────────
// Converts phased tasks to the flat TaskLedgerEntry format expected by the
// autonomous policy and continuation logic.

const PHASED_TASK_LEDGER: TaskLedger = {
	snapshot: () => {
		const phases = getTasks();
		const entries: Array<{
			id: number;
			subject: string;
			status: string;
		}> = [];
		let idx = 0;
		for (const phase of phases) {
			for (const task of phase.tasks) {
				entries.push({
					id: idx++,
					subject: `${phase.name}: ${task.content}`,
					status: task.status,
				});
			}
		}
		return entries;
	},
};

export { PHASED_TASK_LEDGER as taskLedger };
