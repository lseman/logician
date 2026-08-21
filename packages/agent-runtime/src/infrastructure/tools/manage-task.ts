// ── manage_task tool ────────────────────────────────────────────────────────
// Inspect and interact with background tasks spawned by bash.
// Actions: list, status, send_input, kill.

import type { Tool, ToolResult } from "@logician/agent-core";
import { defaultTaskManager, type TaskManager } from "./utils/task-manager.ts";

const manageTaskSchema = {
	type: "object",
	properties: {
		action: {
			type: "string",
			enum: ["list", "status", "kill", "send_input"],
			description:
				"The action to perform: 'list' (list all tasks), 'status' (inspect task status and recent logs), 'send_input' (send stdin to running task), 'kill' (cancel/terminate task).",
		},
		taskId: {
			type: "string",
			description: "The task ID to manage. Required for 'status', 'kill', and 'send_input'.",
		},
		input: {
			type: "string",
			description: "The input string to send to the task stdin. Required when action is 'send_input'.",
		},
		maxLines: {
			type: "number",
			description: "Maximum number of recent log lines to include for 'status' (default: 50).",
		},
	},
	required: ["action"],
} as const;

export interface ManageTaskArgs {
	action: "list" | "status" | "kill" | "send_input";
	taskId?: string;
	input?: string;
	maxLines?: number;
}

export function createManageTaskTool(
	taskManager: TaskManager = defaultTaskManager,
): Tool {
	return {
		name: "manage_task",
		label: "Manage Task",
		hookAliases: ["ManageTask", "Task"],
		description:
			"Manage background tasks. List running tasks, inspect logs and exit status, send input to stdin, or terminate background processes.",
		promptSnippet:
			"Manage background tasks (list, status, send_input, kill)",
		promptGuidelines: [
			"Use manage_task with action='list' to inspect all background jobs",
			"Use manage_task with action='status' to read recent task logs and exit codes",
			"Use manage_task with action='kill' to terminate background jobs when done",
		],
		parameters: manageTaskSchema,
		execute: async (args): Promise<string | ToolResult> => {
			const parsed = args as ManageTaskArgs;
			const { action, taskId, input, maxLines = 50 } = parsed;

			if (action === "list") {
				const tasks = taskManager.listTasks();
				if (tasks.length === 0) {
					return {
						content: "No background tasks currently registered.",
						details: { tasks: [] },
					};
				}

				const formatted = tasks
					.map(task => {
						const durSec = (task.durationMs / 1000).toFixed(1);
						const exitStr =
							task.exitCode !== null
								? ` (exit: ${task.exitCode})`
								: task.signal
									? ` (signal: ${task.signal})`
									: "";
						return `- [${task.id}] ${task.status.toUpperCase()}${exitStr} | PID: ${task.pid ?? "none"} | Duration: ${durSec}s | Command: ${task.command}\n  Log: ${task.logFilePath}`;
					})
					.join("\n\n");

				return {
					content: `Background tasks (${tasks.length}):\n\n${formatted}`,
					details: { tasks },
				};
			}

			if (action === "status") {
				if (!taskId) {
					return {
						content: "Error: taskId is required for 'status' action.",
						isError: true,
					};
				}

				const status = taskManager.getTaskStatus(taskId, maxLines);
				if (!status) {
					return {
						content: `Error: Task "${taskId}" was not found.`,
						isError: true,
					};
				}

				const durSec = (status.durationMs / 1000).toFixed(1);
				const exitStr =
					status.exitCode !== null
						? `\nExit Code: ${status.exitCode}`
						: status.signal
							? `\nTerminated By Signal: ${status.signal}`
							: "";

				const outputBlock = status.recentOutput
					? `\n\nRecent output (last ${Math.min(maxLines, status.totalLines)} of ${status.totalLines} lines):\n\`\`\`\n${status.recentOutput}\n\`\`\``
					: "\n\n(no output recorded yet)";

				return {
					content: `Task: ${status.id}\nCommand: ${status.command}\nStatus: ${status.status.toUpperCase()}\nPID: ${status.pid ?? "unknown"}\nDuration: ${durSec}s${exitStr}\nLog file: ${status.logFilePath}${outputBlock}`,
					details: status,
				};
			}

			if (action === "send_input") {
				if (!taskId) {
					return {
						content: "Error: taskId is required for 'send_input' action.",
						isError: true,
					};
				}
				if (input === undefined) {
					return {
						content: "Error: input is required for 'send_input' action.",
						isError: true,
					};
				}

				const res = taskManager.sendInput(taskId, input);
				return {
					content: res.message,
					isError: !res.success,
				};
			}

			if (action === "kill") {
				if (!taskId) {
					return {
						content: "Error: taskId is required for 'kill' action.",
						isError: true,
					};
				}

				const res = taskManager.killTask(taskId);
				return {
					content: res.message,
					isError: !res.success,
				};
			}

			return {
				content: `Error: Unknown action "${action}". Supported actions: list, status, send_input, kill.`,
				isError: true,
			};
		},
	};
}

export const manage_task: Tool = createManageTaskTool();
