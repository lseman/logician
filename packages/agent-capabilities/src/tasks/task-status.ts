// ── task_status tool ─────────────────────────────────────────────────────────
// Structured end-of-task declaration. The model calls this instead of writing
// "task complete" prose; the loop's continuation logic checks the recorded
// status instead of regex-sniffing the assistant text, and the afterToolCall
// hook terminates the run cleanly.

import {
	recordTaskStatus,
	type TaskStatusRecord,
} from "@logician/agent-core/agent/tasks/task-status-state.ts";
import type { Tool } from "@logician/agent-core/agent/types/index.ts";

export const task_status: Tool = {
	name: "task_status",
	label: "Task Status",
	readOnly: true,
	executionMode: "sequential",
	description:
		"Declare the task finished or blocked. Call this exactly once, as your " +
		"final action, when everything requested is complete, blocked, requires " +
		"user input, or has failed. It ends the run cleanly — do not " +
		"call it while work remains.",
	promptSnippet: "Get task status records for todo items",
	parameters: {
		type: "object",
		properties: {
			status: {
				type: "string",
				enum: ["done", "blocked", "needs_input", "failed"],
				description:
					"done = complete; blocked = externally blocked; needs_input = a user decision is required; failed = unrecoverable failure.",
			},
			summary: {
				type: "string",
				description:
					"One or two sentences: what was accomplished, or what blocks progress.",
			},
		},
		required: ["status", "summary"],
	},
	execute: async args => {
		const allowed = new Set(["done", "blocked", "needs_input", "failed"]);
		const status = allowed.has(String(args.status))
			? (args.status as TaskStatusRecord["status"])
			: "done";
		const summary = typeof args.summary === "string" ? args.summary : "";
		recordTaskStatus({ status, summary, ts: Date.now() });
		return `Recorded: ${status}${summary ? ` — ${summary}` : ""}`;
	},
};
