// ── Todo bar component ─────────────────────────────────────────────────────────
// Pinned task list shown directly above the input bar. Renders nothing when the
// list is empty, so it only takes vertical space while there are active tasks.
//
// Displays: task ID, status mark, subject, activeForm (if in_progress),
//           blockedBy dependencies (as [blocked: #N])
//
// Groups tasks by status: in_progress first, then pending, then completed.

import { type Component, visibleWidth } from "../../tui-core.ts";
import { theme } from "../../theme.ts";

const RESET = "\x1b[0m";

export interface TaskItem {
	id: number;
	subject: string;
	description?: string;
	activeForm?: string;
	status: "pending" | "in_progress" | "completed" | "deleted";
	blockedBy?: number[];
	owner?: string;
	metadata?: Record<string, unknown>;
}

const MARK: Record<TaskItem["status"], () => string> = {
	completed: () => theme.fg("success", "") + "✓" + RESET,
	in_progress: () => theme.fg("warning", "") + "◐" + RESET,
	pending: () => theme.fg("dim", "") + "○" + RESET,
	deleted: () => theme.fg("dim", "") + "✗" + RESET,
};

const MAX_ROWS = 5;

export class TodoBar implements Component {
	private tasks: TaskItem[] = [];
	private onInvalidate: (() => void) | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	setTodos(tasks: TaskItem[]): void {
		this.tasks = tasks;
		this.onInvalidate?.();
	}

	invalidate(): void {
		this.onInvalidate?.();
	}

	render(width: number): string[] {
		if (this.tasks.length === 0) return [];

		// Filter out deleted tasks for display
		const visible = this.tasks.filter((t) => t.status !== "deleted");
		if (visible.length === 0) return [];

		const done = visible.filter((t) => t.status === "completed").length;
		const header = `${theme.fg("muted", "")}Tasks ${done}/${visible.length}${RESET}`;
		const rows: string[] = [pad(header, width)];

		// Group by status for better readability
		const groups = new Map<string, TaskItem[]>();
		for (const t of visible) {
			if (!groups.has(t.status)) groups.set(t.status, []);
			groups.get(t.status)?.push(t);
		}

		// Show in order: in_progress, pending, completed
		const order = ["in_progress", "pending", "completed"];
		let shown = 0;

		for (const status of order) {
			const group = groups.get(status);
			if (!group || shown >= MAX_ROWS) continue;

			// Add section header if we have multiple groups and room
			if (groups.size > 1 && status === "pending" && shown > 0) {
				rows.push(pad("", width)); // spacer
			}

			for (const t of group) {
				if (shown >= MAX_ROWS) break;

				// Build the display line
				const mark = MARK[t.status]();
				const active = t.activeForm ? ` (${t.activeForm})` : "";
				const deps =
					t.blockedBy && t.blockedBy.length > 0
						? ` [blocked: #${t.blockedBy.join(",")}]`
						: "";

				const subject = `${t.id} ${t.subject}`;
				const line = `${mark} ${subject}${active}${deps}`;
				rows.push(pad(clamp(line, width), width));
				shown++;
			}
		}

		const hidden = visible.length - shown;
		if (hidden > 0) {
			rows.push(pad(`${theme.fg("dim", "")} … ${hidden} more${RESET}`, width));
		}

		return rows;
	}
}

function clamp(line: string, width: number): string {
	if (visibleWidth(line) <= width) return line;
	let out = "";
	let w = 0;
	for (const ch of line) {
		const cw = visibleWidth(ch);
		if (w + cw > width - 1) break;
		out += ch;
		w += cw;
	}
	return `${out}…\x1b[0m`;
}

function pad(line: string, width: number): string {
	const w = visibleWidth(line);
	return w < width ? line + " ".repeat(width - w) : line;
}
