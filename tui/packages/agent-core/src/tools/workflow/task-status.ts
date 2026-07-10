// ── task_status tool ─────────────────────────────────────────────────────────
// Structured end-of-task declaration. The model calls this instead of writing
// "task complete" prose; the loop's continuation logic checks the recorded
// status instead of regex-sniffing the assistant text, and the afterToolCall
// hook terminates the run cleanly.

import type { Tool } from "../../core/types.ts";

export interface TaskStatusRecord {
	status: "done" | "blocked";
	summary: string;
	ts: number;
}

let current: TaskStatusRecord | null = null;

/** The status recorded this run, or null. Reset at each run start. */
export function getTaskStatus(): TaskStatusRecord | null {
	return current;
}

export function resetTaskStatus(): void {
	current = null;
}

export const task_status: Tool = {
	name: "task_status",
	label: "Task Status",
	readOnly: true,
	executionMode: "sequential",
	description:
		"Declare the task finished or blocked. Call this exactly once, as your " +
		"final action, when everything requested is complete (status=done) or " +
		"you cannot proceed (status=blocked). It ends the run cleanly — do not " +
		"call it while work remains.",
	promptSnippet: "Get task status records for todo items",
	parameters: {
		type: "object",
		properties: {
			status: {
				type: "string",
				enum: ["done", "blocked"],
				description:
					"done = all requested work complete; blocked = cannot proceed.",
			},
			summary: {
				type: "string",
				description:
					"One or two sentences: what was accomplished, or what blocks progress.",
			},
		},
		required: ["status", "summary"],
	},
	execute: async (args) => {
		const status = args.status === "blocked" ? "blocked" : "done";
		const summary = typeof args.summary === "string" ? args.summary : "";
		current = { status, summary, ts: Date.now() };
		return `Recorded: ${status}${summary ? ` — ${summary}` : ""}`;
	},
};
