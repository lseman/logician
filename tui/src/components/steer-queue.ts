// ── Steering queue component ────────────────────────────────────────────────────
// Pinned display of pending steering messages shown directly above the input bar.
// Renders nothing when the queue is empty, so it only takes vertical space while
// there are pending steering messages.
// Mirrors Pi's updatePendingMessagesDisplay() pattern.

import { type Component, visibleWidth } from "../tui-core.ts";

export class SteerQueue implements Component {
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

	render(width: number): string[] {
		const total = this.steering.length + this.followUp.length;
		if (total === 0) return [];

		const lines: string[] = [];
		lines.push(""); // blank spacer before the block

		// Header: "Queued  2 steering · 1 follow-up"
		const parts: string[] = [];
		if (this.steering.length > 0) {
			parts.push(`${this.steering.length} steering`);
		}
		if (this.followUp.length > 0) {
			parts.push(`${this.followUp.length} follow-up`);
		}
		const header =
			`${HEADER} Queued\x1b[0m  ` + `${COUNT}${parts.join(" · ")}\x1b[0m`;
		lines.push(pad(clampLine(header, width), width));

		// Numbered rows, capped. Steering first (cyan), then follow-up (dim).
		const rows: {
			mark: string;
			style: (s: string) => string;
			msg: string;
		}[] = [];
		this.steering.forEach((msg) =>
			rows.push({ mark: STEER_MARK, style: steerStyle, msg }),
		);
		this.followUp.forEach((msg) =>
			rows.push({ mark: FOLLOW_MARK, style: followStyle, msg }),
		);

		const shown = rows.slice(0, MAX_ROWS);
		shown.forEach((row, i) => {
			const n = `${NUM}${i + 1}.\x1b[0m`;
			const text = `${row.mark} ${n} ${row.style(oneLine(row.msg))}`;
			lines.push(pad(clampLine(text, width), width));
		});

		const hidden = rows.length - shown.length;
		if (hidden > 0) {
			lines.push(
				pad(clampLine(`   ${COUNT}… ${hidden} more\x1b[0m`, width), width),
			);
		}

		lines.push(
			pad(
				clampLine("   \x1b[2mctrl+u to edit queued messages\x1b[0m", width),
				width,
			),
		);

		return lines;
	}
}

// ── Styling ──────────────────────────────────────────────────────────────────

const MAX_ROWS = 6;
const HEADER = "\x1b[38;5;245m";
const COUNT = "\x1b[38;5;240m";
const NUM = "\x1b[38;5;245m";
const STEER_MARK = " \x1b[36m▸\x1b[0m"; // cyan triangle ▸
const FOLLOW_MARK = " \x1b[38;5;240m↳\x1b[0m"; // dim hook arrow ↳
const steerStyle = (s: string): string => `\x1b[38;5;252m${s}\x1b[0m`;
const followStyle = (s: string): string => `\x1b[38;5;245m${s}\x1b[0m`;

/** Collapse newlines/whitespace runs so each queued message is one row. */
function oneLine(msg: string): string {
	const flat = msg.replace(/\s+/g, " ").trim();
	return flat.length > 120 ? `${flat.slice(0, 120)}…` : flat;
}

function clampLine(text: string, maxW: number): string {
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
