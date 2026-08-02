// ── Steering queue component ────────────────────────────────────────────────────
// Pinned display of pending steering messages shown directly above the input bar.
// Renders nothing when the queue is empty, so it only takes vertical space while
// there are pending steering messages.
// Mirrors Pi's updatePendingMessagesDisplay() pattern.

import {
	clampLineToWidth,
	type InkTextComponent,
	type InkTextRow,
	visibleWidth,
} from "../terminal/core.ts";
import { DIM, RESET, semanticMarkupToInkRow, theme } from "../rendering/transcript/semantic-markup.ts";

export class SteerQueue implements InkTextComponent {
	private steering: string[] = [];
	private followUp: string[] = [];
	private onInvalidate: (() => void) | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	setItems(steering: string[], followUp: string[] = []): void {
		this.steering = steering;
		this.followUp = followUp;
		this.onInvalidate?.();
	}

	invalidate(): void {
		this.onInvalidate?.();
	}

	getInkTextRows(width: number): InkTextRow[] {
		const total = this.steering.length + this.followUp.length;
		if (total === 0) return [];

		const lines: string[] = [];
		lines.push(""); // blank spacer before the block

		// Header: "STEERING  2 queued · 1 follow-up"
		const parts: string[] = [];
		if (this.steering.length > 0) {
			parts.push(`${this.steering.length} queued`);
		}
		if (this.followUp.length > 0) {
			parts.push(`${this.followUp.length} follow-up`);
		}
		const header =
			`${getHeader()} STEERING${RESET}  ` +
			`${getCount()}${parts.join(" · ")}${RESET}`;
		lines.push(pad(clampLineToWidth(header, width), width));

		// Numbered rows, capped. Delivery labels make queue semantics visible
		// without relying on color alone.
		const rows: {
			mark: string;
			label: string;
			style: (s: string) => string;
			msg: string;
		}[] = [];
		this.steering.forEach((msg) =>
			rows.push({ mark: getSteerMark(), label: "QUEUE", style: steerStyle, msg }),
		);
		this.followUp.forEach((msg) =>
			rows.push({
				mark: getFollowMark(),
				label: "LATER",
				style: followStyle,
				msg,
			}),
		);

		const shown = rows.slice(0, MAX_ROWS);
		shown.forEach((row, i) => {
			const n = `${getNum()}${i + 1}.${RESET}`;
			const label = `${getLabel()}${row.label}${RESET}`;
			const text = `${row.mark} ${n} ${label}  ${row.style(oneLine(row.msg))}`;
			lines.push(pad(clampLineToWidth(text, width), width));
		});

		const hidden = rows.length - shown.length;
		if (hidden > 0) {
			lines.push(
				pad(
					clampLineToWidth(`   ${getCount()}… ${hidden} more${RESET}`, width),
					width,
				),
			);
		}

		lines.push(
			pad(
				clampLineToWidth(
					`   ${DIM}Enter queue · Ctrl+Enter steer now · /queue manage${RESET}`,
					width,
				),
				width,
			),
		);

		return lines.map(semanticMarkupToInkRow);
	}
}

// ── Styling ──────────────────────────────────────────────────────────────────

const MAX_ROWS = 6;
const getHeader = (): string => theme.fgRaw("muted");
const getCount = (): string => theme.fgRaw("dim");
const getNum = (): string => theme.fgRaw("muted");
const getLabel = (): string => theme.fgRaw("dim");
const getSteerMark = (): string => ` ${theme.fg("accent", "▸")}`;
const getFollowMark = (): string => " " + theme.fgRaw("dim") + "↳" + RESET;
const steerStyle = (s: string): string => theme.fgRaw("text") + s + RESET;
const followStyle = (s: string): string => theme.fgRaw("muted") + s + RESET;

/** Collapse newlines/whitespace runs so each queued message is one row. */
function oneLine(msg: string): string {
	const flat = msg.replace(/\s+/g, " ").trim();
	return flat.length > 120 ? `${flat.slice(0, 120)}…` : flat;
}

function pad(line: string, width: number): string {
	const w = visibleWidth(line);
	return w < width ? line + " ".repeat(width - w) : line;
}
