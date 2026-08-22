// ── Steering queue component ────────────────────────────────────────────────────
// Pinned display of pending steering / follow-up / next-turn messages shown
// directly above the input bar. Renders nothing when all three queues are
// empty, so it only takes vertical space while there's something queued.
//
// Clickable: clicking any queued steering message immediately steers with it.
// Ctrl+Q opens the queue manager for full management.

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

const MAX_ROWS = 6;

interface Row {
	kind: QueueKind;
	key: string;
	msg: string;
}

export type SteerQueueAction =
	| { type: "steerNow"; message: string }
	| { type: "openManager" };

export interface SteerQueueCallbacks {
	onAction?: (action: SteerQueueAction) => void;
}

export class SteerQueue implements Component {
	private steering: readonly string[] = [];
	private followUp: readonly string[] = [];
	private nextTurn: readonly string[] = [];
	private onInvalidate: (() => void) | null = null;
	private cachedLines: string[] | null = null;
	private cachedKey = "";

	// Number of body rows for hit-testing. Body rows are rendered at
	// content-relative y = 2 .. 1 + bodyRowCount.
	private bodyRowCount = 0;

	private callbacks: SteerQueueCallbacks = {};

	setCallbacks(cb: SteerQueueCallbacks): void {
		this.callbacks = cb;
	}

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

	// ── Mouse hit-testing ─────────────────────────────────────────────────

	/**
	 * Handle a mouse click. `row` is content-relative (0-indexed line within
	 * this component's rendered output). Returns true if the click was handled.
	 *
	 * Layout: blank(0) | header(1) | body(2..2+N-1) | footer(2+N)
	 * Body rows are 2 through 1+bodyRowCount (inclusive).
	 */
	handleMouse(_column: number, row: number): boolean {
		const bodyStart = 2;
		const bodyEnd = bodyStart + this.bodyRowCount - 1;
		if (row >= bodyStart && row <= bodyEnd && this.bodyRowCount > 0) {
			const allRows = this.rows();
			const clickedRow = allRows[row - bodyStart];
			if (clickedRow && clickedRow.kind === "steering") {
				this.callbacks.onAction?.({ type: "steerNow", message: clickedRow.msg });
				return true;
			}
		}
		// Footer row — open queue manager
		const totalLines = 1 + 1 + this.bodyRowCount + 1; // blank + header + body + footer
		if (row === totalLines - 1) {
			this.callbacks.onAction?.({ type: "openManager" });
			return true;
		}
		return false;
	}

	render(width: number): string[] {
		const rows = this.rows();
		const key = `${width}:${rows.map(r => r.key).join("\x01")}`;
		if (this.cachedLines !== null && this.cachedKey === key) {
			return this.cachedLines;
		}
		this.cachedKey = key;
		const rendered = renderRows(rows, width);
		this.bodyRowCount = Math.min(rows.length, MAX_ROWS);
		return rendered;
	}
}

// ── Render (pure function) ──────────────────────────────────────────────────

function renderRows(rows: Row[], width: number): string[] {
	if (rows.length === 0) return [];

	const lines: string[] = [];
	lines.push(""); // blank spacer

	// Header: "◈ Queue · 2 queued · 1 follow-up"
	const steeringCount = rows.filter(r => r.kind === "steering").length;
	const followUpCount = rows.filter(r => r.kind === "followUp").length;
	const nextTurnCount = rows.filter(r => r.kind === "nextTurn").length;
	const parts: string[] = [];
	if (steeringCount > 0) parts.push(`${steeringCount} queued`);
	if (followUpCount > 0) parts.push(`${followUpCount} follow-up`);
	if (nextTurnCount > 0) parts.push(`${nextTurnCount} next turn`);

	const headerIcon = theme.fg("accent", "◈");
	const headerLabel = theme.bold(theme.fg("muted", "QUEUE"));
	const headerCount = parts.length ? theme.fg("dim", ` · ${parts.join(" · ")}`) : "";
	const header = `${headerIcon} ${headerLabel}${headerCount}`;
	lines.push(pad(clampLineToWidth(header, width), width));

	const shown = rows.slice(0, MAX_ROWS);
	shown.forEach((row, i) => {
		const meta = KIND_META[row.kind];
		const isSteering = row.kind === "steering";

		// Steering rows get a clickable ▶ indicator.
		const clickable = isSteering
			? theme.fg("accent", "▶")
			: theme.fg("dim", "·");

		const msgText = oneLine(row.msg);
		const msgStyled = theme.fg(meta.textColor, msgText);

		// First steering row gets "steer now" affordance.
		const isFirstSteering = isSteering && i === 0;
		const affordance = isFirstSteering
			? ` ${theme.fg("accent", "steer now")}`
			: "";

		const text = `${clickable} ${meta.mark} ${msgStyled}${affordance}`;
		lines.push(pad(clampLineToWidth(text, width), width));
	});

	const hidden = rows.length - shown.length;
	if (hidden > 0) {
		lines.push(
			pad(
				clampLineToWidth(
					`   ${theme.fg("dim", `… ${hidden} more — click to steer`)}`,
					width,
				),
				width,
			),
		);
	}

	// Footer: action hints
	lines.push(
		pad(
			clampLineToWidth(
				`${theme.fg("dim", "click to steer")}  ${theme.fg("dim", "Ctrl+Enter")} ${theme.fg("muted", "steer now")}  ${theme.fg("dim", "Ctrl+Q")} ${theme.fg("muted", "manage")}`,
				width,
			),
			width,
		),
	);

	return lines;
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
