// ── Steering queue component ────────────────────────────────────────────────────
// Pinned display of pending steering messages shown directly above the input bar.
// Renders nothing when the queue is empty, so it only takes vertical space while
// there are pending steering messages.
// Mirrors Pi's updatePendingMessagesDisplay() pattern.

import { type Component, visibleWidth, RESET } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

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
			`${getHeader()} Queued\x1b[0m  ` +
			`${getCount()}${parts.join(" · ")}\x1b[0m`;
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
			rows.push({ mark: getFollowMark(), style: followStyle, msg }),
		);

		const shown = rows.slice(0, MAX_ROWS);
		shown.forEach((row, i) => {
			const n = `${getNum()}${i + 1}.\x1b[0m`;
			const text = `${row.mark} ${n} ${row.style(oneLine(row.msg))}`;
			lines.push(pad(clampLine(text, width), width));
		});

		const hidden = rows.length - shown.length;
		if (hidden > 0) {
			lines.push(
				pad(clampLine(`   ${getCount()}… ${hidden} more\x1b[0m`, width), width),
			);
		}

		lines.push(pad(clampLine("   \x1b[2m/steer-now process · /queue-drop N remove · /queue-clear clear\x1b[0m", width), width));

		return lines;
	}
}

// ── Styling ──────────────────────────────────────────────────────────────────

const MAX_ROWS = 6;
const getHeader = (): string => theme.fg("muted", "");
const getCount = (): string => theme.fg("dim", "");
const getNum = (): string => theme.fg("muted", "");
const STEER_MARK = " \x1b[36m▸\x1b[0m"; // cyan triangle ▸
const getFollowMark = (): string => " " + theme.fg("dim", "") + "↳" + RESET;
const steerStyle = (s: string): string => theme.fg("text", "") + s + RESET;
const followStyle = (s: string): string => theme.fg("muted", "") + s + RESET;

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
