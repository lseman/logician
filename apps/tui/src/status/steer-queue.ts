// ── Steering queue component ────────────────────────────────────────────────────
// Pinned display of pending steering / follow-up / next-turn messages shown
// directly above the input bar. Renders nothing when all three queues are
// empty, so it only takes vertical space while there's something queued.
// Mirrors Pi's updatePendingMessagesDisplay() pattern.

import {
	type Component,
	clampLineToWidth,
	visibleWidth,
} from "../terminal/core.ts";
import { type ThemeColor, theme } from "../terminal/theme.ts";

type QueueKind = "steering" | "followUp" | "nextTurn";

const KIND_META: Record<
	QueueKind,
	{ mark: string; label: string; color: ThemeColor; textColor: ThemeColor }
> = {
	steering: { mark: "▸", label: "QUEUE", color: "accent", textColor: "text" },
	followUp: { mark: "↳", label: "LATER", color: "dim", textColor: "muted" },
	nextTurn: { mark: "↷", label: "NEXT", color: "muted", textColor: "muted" },
};

// New rows play a brief arrival pulse instead of appearing at full brightness
// instantly — same easing feel as TodoBar's status transitions, scoped to
// "just landed" rather than a status change (queue entries don't have states).
const ARRIVAL_FRAMES = ["◦", "◔", "◑", "◕"];
const ARRIVAL_TICKS = ARRIVAL_FRAMES.length;
const ARRIVAL_INTERVAL_MS = 70;

const MAX_ROWS = 6;

interface Row {
	kind: QueueKind;
	key: string;
	msg: string;
}

export class SteerQueue implements Component {
	private steering: readonly string[] = [];
	private followUp: readonly string[] = [];
	private nextTurn: readonly string[] = [];
	private onInvalidate: (() => void) | null = null;
	private cachedLines: string[] | null = null;
	private cachedKey = "";

	// Arrival animation state, keyed by "kind:message" (stable across re-renders
	// as long as the message text and its queue don't change).
	private seen = new Set<string>();
	private arrivalFrame = new Map<string, number>();
	private timer: ReturnType<typeof setInterval> | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	setItems(
		steering: readonly string[],
		followUp: readonly string[] = [],
		nextTurn: readonly string[] = [],
	): void {
		this.steering = steering;
		this.followUp = followUp;
		this.nextTurn = nextTurn;

		const liveKeys = new Set<string>();
		let anyArrival = false;
		for (const row of this.rows()) {
			liveKeys.add(row.key);
			if (!this.seen.has(row.key)) {
				this.arrivalFrame.set(row.key, 0);
				anyArrival = true;
			}
		}
		this.seen = liveKeys;
		for (const key of this.arrivalFrame.keys()) {
			if (!liveKeys.has(key)) this.arrivalFrame.delete(key);
		}
		if (anyArrival) this.startAnimation();

		this.cachedLines = null;
		this.onInvalidate?.();
	}

	invalidate(): void {
		this.cachedLines = null;
		this.onInvalidate?.();
	}

	private rows(): Row[] {
		const rows: Row[] = [];
		for (const msg of this.steering) {
			rows.push({ kind: "steering", key: `steering:${msg}`, msg });
		}
		for (const msg of this.followUp) {
			rows.push({ kind: "followUp", key: `followUp:${msg}`, msg });
		}
		for (const msg of this.nextTurn) {
			rows.push({ kind: "nextTurn", key: `nextTurn:${msg}`, msg });
		}
		return rows;
	}

	private startAnimation(): void {
		if (this.timer) return;
		this.timer = setInterval(() => {
			let stillRunning = false;
			for (const [key, frame] of this.arrivalFrame) {
				const next = frame + 1;
				if (next >= ARRIVAL_TICKS) {
					this.arrivalFrame.delete(key);
				} else {
					this.arrivalFrame.set(key, next);
					stillRunning = true;
				}
			}
			this.cachedLines = null;
			this.onInvalidate?.();
			if (!stillRunning) this.stopAnimation();
		}, ARRIVAL_INTERVAL_MS);
	}

	private stopAnimation(): void {
		if (this.timer) {
			clearInterval(this.timer);
			this.timer = null;
		}
	}

	render(width: number): string[] {
		const rows = this.rows();
		const key = `${width}:${rows.map(r => r.key).join("")}`;
		if (
			this.cachedLines !== null &&
			this.cachedKey === key &&
			this.arrivalFrame.size === 0
		) {
			return this.cachedLines;
		}
		this.cachedKey = key;
		this.cachedLines = renderRows(rows, width, this.arrivalFrame);
		return this.cachedLines;
	}
}

// ── Render (pure function) ──────────────────────────────────────────────────

function renderRows(
	rows: Row[],
	width: number,
	arrivalFrame: Map<string, number>,
): string[] {
	if (rows.length === 0) return [];

	const lines: string[] = [];
	lines.push(""); // blank spacer before the block

	// Header: "STEERING  2 queued · 1 follow-up · 1 next turn"
	const steeringCount = rows.filter(r => r.kind === "steering").length;
	const followUpCount = rows.filter(r => r.kind === "followUp").length;
	const nextTurnCount = rows.filter(r => r.kind === "nextTurn").length;
	const parts: string[] = [];
	if (steeringCount > 0) parts.push(`${steeringCount} queued`);
	if (followUpCount > 0) parts.push(`${followUpCount} follow-up`);
	if (nextTurnCount > 0) parts.push(`${nextTurnCount} next turn`);
	const header = `${theme.fg("muted", "STEERING")}  ${theme.fg("dim", parts.join(" · "))}`;
	lines.push(pad(clampLineToWidth(header, width), width));

	const shown = rows.slice(0, MAX_ROWS);
	shown.forEach((row, i) => {
		const meta = KIND_META[row.kind];
		const frame = arrivalFrame.get(row.key);
		const mark = arrivalMark(meta, frame);
		const n = theme.fg("muted", `${i + 1}.`);
		const label = theme.fg("dim", meta.label);
		const text = `${mark} ${n} ${label}  ${theme.fg(meta.textColor, oneLine(row.msg))}`;
		lines.push(pad(clampLineToWidth(text, width), width));
	});

	const hidden = rows.length - shown.length;
	if (hidden > 0) {
		lines.push(
			pad(
				clampLineToWidth(`   ${theme.fg("dim", `… ${hidden} more`)}`, width),
				width,
			),
		);
	}

	lines.push(
		pad(
			clampLineToWidth(
				`   ${theme.dim("Enter queue · Ctrl+Enter steer now · /queue manage")}`,
				width,
			),
			width,
		),
	);

	return lines;
}

function arrivalMark(
	meta: (typeof KIND_META)[QueueKind],
	frame: number | undefined,
): string {
	const glyph = frame === undefined ? meta.mark : ARRIVAL_FRAMES[frame];
	return ` ${theme.fg(meta.color, glyph)}`;
}

/** Collapse newlines/whitespace runs so each queued message is one row. */
function oneLine(msg: string): string {
	const flat = msg.replace(/\s+/g, " ").trim();
	return flat.length > 120 ? `${flat.slice(0, 120)}…` : flat;
}

function pad(line: string, width: number): string {
	const w = visibleWidth(line);
	return w < width ? line + " ".repeat(width - w) : line;
}
