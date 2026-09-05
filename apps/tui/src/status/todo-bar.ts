// ── Todo bar component ────────────────────────────────────────────────────────
// Phased task list rendered above the input bar. Shows phase headers with Roman
// numerals, progress counts, and task status marks.
//
// Rendering strategy:
//   - Phases touched in the latest update are rendered fully.
//   - Phases not touched show a collapsed summary line: "I. Foundation · 2/5"
//   - The phase with the in_progress task is always touched (active attention).
//
// Status marks:
//   → in_progress (accent), ✓ completed (success + strikethrough),
//   ○ pending (dim), ✕ abandoned (error + strikethrough)

import { type Component, visibleWidth } from "../terminal/core.ts";
import { type ThemeColor, theme } from "../terminal/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const STRIKE = "\x1b[9m";

// ── Phased data types ─────────────────────────────────────────────────────────

export interface TodoPhase {
	name: string;
	tasks: TodoTask[];
}

export interface TodoTask {
	content: string;
	status: "pending" | "in_progress" | "completed" | "abandoned";
	blocker?: string;
}

/** Tracks which task completed in the last update (for strikethrough animation). */
export interface CompletionTransition {
	phase: string;
	content: string;
}

// ── Status marks ──────────────────────────────────────────────────────────────

const STATUS: Record<TodoTask["status"], { sym: string; color: ThemeColor }> = {
	completed: { sym: "✓", color: "success" },
	in_progress: { sym: "→", color: "accent" },
	pending: { sym: "○", color: "dim" },
	abandoned: { sym: "✕", color: "error" },
};

// ── Roman numeral display ─────────────────────────────────────────────────────

const ROMAN_PAIRS: ReadonlyArray<readonly [number, string]> = [
	[1000, "M"],
	[900, "CM"],
	[500, "D"],
	[400, "CD"],
	[100, "C"],
	[90, "XC"],
	[50, "L"],
	[40, "XL"],
	[10, "X"],
	[9, "IX"],
	[5, "V"],
	[4, "IV"],
	[1, "I"],
];

function roman(n: number): string {
	let remaining = n;
	let result = "";
	for (const [value, numeral] of ROMAN_PAIRS) {
		while (remaining >= value) {
			result += numeral;
			remaining -= value;
		}
	}
	return result;
}

function formatPhaseDisplayName(name: string, oneBasedIndex: number): string {
	return `${roman(oneBasedIndex)}. ${name}`;
}

// ── Strikethrough animation ───────────────────────────────────────────────────

const STRIKE_HOLD_FRAMES = 2;
const STRIKE_REVEAL_FRAMES = 12;
const STRIKE_TOTAL_FRAMES = STRIKE_HOLD_FRAMES + STRIKE_REVEAL_FRAMES;
const STRIKE_START = "\x1b[9m";
const STRIKE_END = "\x1b[29m";

function partialStrikethrough(text: string, visibleChars: number): string {
	if (visibleChars <= 0) return text;
	const fullStrike = STRIKE + text + STRIKE_END;
	if (visibleChars >= text.length) return fullStrike;
	return STRIKE + text.slice(0, visibleChars) + STRIKE_END + text.slice(visibleChars);
}

function strikeRevealCount(totalChars: number, frame: number): number {
	if (frame < STRIKE_HOLD_FRAMES) return totalChars;
	return Math.round(
		((frame - STRIKE_HOLD_FRAMES) / STRIKE_REVEAL_FRAMES) * totalChars,
	);
}

// ── Phase-aware rendering ─────────────────────────────────────────────────────

const MAX_ROWS = 6;

function computeTouchedPhases(
	completedTasks: CompletionTransition[] | undefined,
): Set<string> | null {
	if (!completedTasks || completedTasks.length === 0) return null;
	const touched = new Set<string>();
	for (const ct of completedTasks) touched.add(ct.phase);
	return touched.size > 0 ? touched : null;
}

function formatPhaseProgress(phase: TodoPhase): string {
	const done = phase.tasks.filter(t => t.status === "completed" || t.status === "abandoned").length;
	return theme.fg("dim", `  ${done}/${phase.tasks.length}`);
}

function formatPhaseSummary(
	phase: TodoPhase,
	oneBasedIndex: number,
): string {
	const header = theme.fg("muted", formatPhaseDisplayName(phase.name, oneBasedIndex));
	return header + formatPhaseProgress(phase);
}

// ── Component ─────────────────────────────────────────────────────────────────

export class TodoBar implements Component {
	private phases: TodoPhase[] = [];
	private completedTasks: CompletionTransition[] | undefined;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private onInvalidate: (() => void) | null = null;

	// Completion animation state: track which tasks just completed.
	private completionFrames = new Map<string, number>(); // "phase\x00content" -> frame
	private timer: ReturnType<typeof setInterval> | null = null;
	private readonly COMPLETION_FRAME_TICKS = STRIKE_TOTAL_FRAMES;
	private readonly COMPLETION_INTERVAL_MS = 40;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	setPhases(phases: TodoPhase[], completedTasks?: CompletionTransition[]): void {
		this.phases = phases;
		this.completedTasks = completedTasks;

		// Mark newly completed tasks with animation frames.
		const prevKeys = new Set<string>(this.completionFrames.keys());
		const currentKeys = new Set<string>();
		for (const ct of completedTasks ?? []) {
			const key = `${ct.phase}\x00${ct.content}`;
			currentKeys.add(key);
			if (!prevKeys.has(key)) {
				this.completionFrames.set(key, 0);
			}
		}
		for (const key of prevKeys) {
			if (!currentKeys.has(key)) {
				this.completionFrames.delete(key);
			}
		}

		this.startAnimation();
		this.cachedLines = null;
		this.onInvalidate?.();
	}

	invalidate(): void {
		this.cachedLines = null;
		this.onInvalidate?.();
	}

	private startAnimation(): void {
		if (this.timer) return;
		this.timer = setInterval(() => {
			let stillRunning = false;
			for (const [key, frame] of this.completionFrames) {
				const next = frame + 1;
				if (next >= this.COMPLETION_FRAME_TICKS) {
					this.completionFrames.delete(key);
				} else {
					this.completionFrames.set(key, next);
					stillRunning = true;
				}
			}
			this.cachedLines = null;
			this.onInvalidate?.();
			if (!stillRunning) this.stopAnimation();
		}, this.COMPLETION_INTERVAL_MS);
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
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}
		const lines = renderRaw(width, this.phases, this.completedTasks, this.completionFrames);
		this.cachedWidth = width;
		this.cachedLines = lines;
		return lines;
	}
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderRaw(
	width: number,
	phases: TodoPhase[],
	completedTasks: CompletionTransition[] | undefined,
	completionFrames: Map<string, number>,
): string[] {
	const visibleTasks = phases.flatMap(p =>
		p.tasks.filter(t => t.status !== "abandoned"),
	);
	if (visibleTasks.length === 0) return [];

	const done = phases.reduce(
		(sum, p) => sum + p.tasks.filter(t => t.status === "completed").length,
		0,
	);
	const total = phases.reduce((sum, p) => sum + p.tasks.filter(t => t.status !== "abandoned").length, 0);
	const active = total - done;

	const lines: string[] = [];

	// Header
	if (active > 0) {
		const header = `${theme.fg("muted", "Tasks ")}${done}/${total}${RESET}${theme.fg("accent", ` · ${active} active`)}`;
		lines.push(pad(clampLine(header, width), width));
	} else {
		const header = `${theme.fg("muted", "Tasks ")}${done}/${total}${RESET}`;
		lines.push(pad(clampLine(header, width), width));
	}

	// Determine touched phases (from completion transitions).
	const touched = computeTouchedPhases(completedTasks);

	// Find the phase with in_progress tasks — always render fully.
	const activePhaseNames = new Set<string>();
	for (const phase of phases) {
		if (phase.tasks.some(t => t.status === "in_progress")) {
			activePhaseNames.add(phase.name);
		}
	}

	let shownRows = 0;
	const isTouched = (phaseName: string): boolean =>
		touched !== null
			? touched.has(phaseName) || activePhaseNames.has(phaseName)
			: true; // null means render everything fully

	for (let idx = 0; idx < phases.length && shownRows < MAX_ROWS; idx++) {
		const phase = phases[idx];
		const oneBasedIndex = idx + 1;

		if (isTouched(phase.name)) {
			// Full phase rendering
			lines.push(pad(clampLine(formatPhaseDisplayName(phase.name, oneBasedIndex), width), width));
			for (const t of phase.tasks) {
				if (shownRows >= MAX_ROWS) break;
				const line = buildTaskLine(t, completionFrames);
				lines.push(pad(clampLine(line, width), width));
				shownRows++;
			}
			lines.push(pad(clampLine(formatPhaseProgress(phase), width), width));
			shownRows++;
		} else {
			// Collapsed summary
			if (shownRows >= MAX_ROWS) break;
			lines.push(pad(clampLine(formatPhaseSummary(phase, oneBasedIndex), width), width));
			shownRows++;
		}
	}

	const hiddenCount =
		total -
		phases.reduce(
			(sum, p) =>
				sum +
				(isTouched(p.name) ? p.tasks.filter(t => t.status !== "abandoned").length : 0),
			0,
		);

	if (hiddenCount > 0) {
		lines.push(pad(clampLine(`   ${DIM}… ${hiddenCount} more${RESET}`, width), width));
	}

	return lines;
}

function buildTaskLine(
	t: TodoTask,
	completionFrames: Map<string, number>,
): string {
	const mark = STATUS[t.status].sym;
	const markColored = theme.fg(STATUS[t.status].color, mark);

	let text = markColored + " ";

	// Apply strikethrough animation for just-completed tasks.
	if (t.status === "completed") {
		const key = `completed\x00${t.content}`;
		const frame = completionFrames.get(key);
		if (frame !== undefined) {
			const revealCount = strikeRevealCount(t.content.length, frame);
			text += partialStrikethrough(t.content, revealCount);
		} else {
			text += STRIKE + t.content + STRIKE_END;
		}
	} else if (t.status === "abandoned") {
		text += STRIKE + t.content + STRIKE_END;
	} else {
		text += t.content;
	}

	// Show blocker inline.
	if (t.blocker) {
		text += ` ${DIM}[blocked: ${t.blocker}]${RESET}`;
	}

	return theme.fg(STATUS[t.status].color, text);
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
