// ── Todo bar component ─────────────────────────────────────────────────────────
// Pinned task list shown directly above the input bar. Renders nothing when the
// list is empty, so it only takes vertical space while there are active tasks.
//
// Displays: compact task list with status marks, grouped by status.
//           in_progress first, then pending, then completed.

import { type Component, visibleWidth } from "../terminal/core.ts";
import { type ThemeColor, theme } from "../terminal/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";

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

// ── Status marks ──────────────────────────────────────────────────────────────

const STATUS: Record<TaskItem["status"], { sym: string; color: ThemeColor }> = {
	completed: { sym: "✓", color: "success" },
	in_progress: { sym: "▸", color: "warning" },
	pending: { sym: "○", color: "dim" },
	deleted: { sym: "✗", color: "dim" },
};

// Fill-in frames shown briefly when a task's status just changed, e.g.
// pending → in_progress plays ○ ◔ ◑ ◕ ▸, in_progress → completed plays ▸ ◕ ● ✓.
const TRANSITION_FRAMES: Record<TaskItem["status"], string[]> = {
	pending: ["○"],
	in_progress: ["◔", "◑", "◕", "▸"],
	completed: ["◕", "●", "✓"],
	deleted: ["✗"],
};
const TRANSITION_TICKS = 4;
const TRANSITION_INTERVAL_MS = 90;
const ANSI_CSI_SEQUENCE = new RegExp(
	`${String.fromCharCode(27)}\\[[0-?]*[ -/]*[@-~]`,
	"g",
);

function statusMark(status: TaskItem["status"], frame?: number): string {
	const s = STATUS[status];
	if (frame !== undefined) {
		const frames = TRANSITION_FRAMES[status];
		const sym = frames[Math.min(frame, frames.length - 1)];
		return ` ${theme.fg(s.color, sym)}${RESET}`;
	}
	return ` ${theme.fg(s.color, s.sym)}${RESET}`;
}

// ── Task list ─────────────────────────────────────────────────────────────────

const MAX_ROWS = 5;

export class TodoBar implements Component {
	private tasks: TaskItem[] = [];
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private cachedCount = -1;
	private onInvalidate: (() => void) | null = null;

	// Transition animation state: last-seen status per task id, and remaining
	// frame count for tasks currently mid-transition.
	private prevStatus = new Map<number, TaskItem["status"]>();
	private transitionFrame = new Map<number, number>();
	private timer: ReturnType<typeof setInterval> | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	setTodos(tasks: TaskItem[]): void {
		this.tasks = tasks.flatMap(task => {
			const subject = normalizeLabel(task?.subject);
			if (!subject) return [];
			return [
				{
					...task,
					subject,
					activeForm: normalizeLabel(task.activeForm) || undefined,
					blockedBy: Array.isArray(task.blockedBy)
						? task.blockedBy.filter(Number.isFinite)
						: undefined,
				},
			];
		});

		let anyTransition = false;
		const liveIds = new Set(this.tasks.map(t => t.id));
		for (const t of this.tasks) {
			const prev = this.prevStatus.get(t.id);
			if (prev !== undefined && prev !== t.status) {
				this.transitionFrame.set(t.id, 0);
				anyTransition = true;
			}
			this.prevStatus.set(t.id, t.status);
		}
		for (const id of this.prevStatus.keys()) {
			if (!liveIds.has(id)) {
				this.prevStatus.delete(id);
				this.transitionFrame.delete(id);
			}
		}
		if (anyTransition) this.startAnimation();

		this.cachedLines = null;
		this.cachedCount = -1;
		this.onInvalidate?.();
	}

	invalidate(): void {
		this.cachedLines = null;
		this.cachedCount = -1;
		this.onInvalidate?.();
	}

	private startAnimation(): void {
		if (this.timer) return;
		this.timer = setInterval(() => {
			let stillRunning = false;
			for (const [id, frame] of this.transitionFrame) {
				const next = frame + 1;
				if (next >= TRANSITION_TICKS) {
					this.transitionFrame.delete(id);
				} else {
					this.transitionFrame.set(id, next);
					stillRunning = true;
				}
			}
			this.cachedLines = null;
			this.onInvalidate?.();
			if (!stillRunning) this.stopAnimation();
		}, TRANSITION_INTERVAL_MS);
	}

	private stopAnimation(): void {
		if (this.timer) {
			clearInterval(this.timer);
			this.timer = null;
		}
	}

	dispose(): void {
		this.stopAnimation();
		this.onInvalidate = null;
	}

	render(width: number): string[] {
		const countKey = this.tasks.filter(t => t.status !== "deleted").length;

		if (
			width === this.cachedWidth &&
			countKey === this.cachedCount &&
			this.cachedLines !== null &&
			this.transitionFrame.size === 0
		) {
			return this.cachedLines;
		}

		const lines = renderRaw(width, this.tasks, this.transitionFrame);
		this.cachedWidth = width;
		this.cachedCount = countKey;
		this.cachedLines = lines;
		return lines;
	}
}

// ── Render (pure function, no closure over `this`) ────────────────────────────

function renderRaw(
	width: number,
	tasks: TaskItem[],
	transitionFrame: Map<number, number>,
): string[] {
	const visible = tasks.filter(t => t.status !== "deleted");
	if (visible.length === 0) return [];

	const done = visible.filter(t => t.status === "completed").length;
	const total = visible.length;
	const lines: string[] = [];

	// Header
	const header = `${theme.fg("muted", "")}Tasks ${done}/${total}${RESET}`;
	lines.push(pad(clampLine(header, width), width));

	// Group by status: in_progress → pending → completed
	const groups = new Map<string, TaskItem[]>();
	for (const t of visible) {
		if (!groups.has(t.status)) groups.set(t.status, []);
		groups.get(t.status)?.push(t);
	}

	const order: TaskItem["status"][] = ["in_progress", "pending", "completed"];
	let shown = 0;

	for (const status of order) {
		const group = groups.get(status);
		if (!group || shown >= MAX_ROWS) continue;

		for (const t of group) {
			if (shown >= MAX_ROWS) break;

			const mark = statusMark(t.status, transitionFrame.get(t.id));
			const line = buildTaskLine(t, mark);
			lines.push(pad(clampLine(line, width), width));
			shown++;
		}
	}

	// Hidden count
	const hidden = total - shown;
	if (hidden > 0) {
		const hint = `${DIM}… ${hidden} more${RESET}`;
		lines.push(pad(clampLine(`   ${hint}`, width), width));
	}

	return lines;
}

function buildTaskLine(t: TaskItem, mark: string): string {
	let text = `${mark} ${t.subject}`;

	if (t.activeForm) {
		text += ` ${DIM}— ${t.activeForm}${RESET}`;
	}

	if (t.blockedBy?.length) {
		const deps = t.blockedBy
			.map(id => ` ${theme.fg("muted", `[→ #${id}]`)}${RESET}`)
			.join("");
		text += deps;
	}

	return text;
}

function clampLine(text: string, maxW: number): string {
	if (visibleWidth(text) <= maxW) return text;
	let out = "";
	let w = 0;
	for (const ch of text) {
		const cw = visibleWidth(ch);
		if (w + cw > maxW) break;
		out += ch;
		w += cw;
	}
	return out;
}

function pad(line: string, width: number): string {
	const w = visibleWidth(line);
	return w < width ? line + " ".repeat(width - w) : line;
}

function normalizeLabel(value: unknown): string {
	return String(value ?? "")
		.replace(ANSI_CSI_SEQUENCE, "")
		.replace(/[\p{Cc}\p{Cf}]/gu, " ")
		.replace(/\s+/g, " ")
		.trim();
}
