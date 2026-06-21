// ── ReasonerSelectorOverlay ──────────────────────────────────────────────────────
// Overlay for selecting an active reasoning mode.
// Mirrors PluginManagerOverlay pattern: list, select, confirm, close.
// Reasoner selection applies to the next turn (never mutates an in-flight run).

import { type Component, clampLineToWidth, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const getHeader = (): string => theme.fg("header", "");
const getSelected = (): string => theme.fg("selected", "");
const getMuted = (): string => theme.fg("muted", "");
const getActive = (): string => theme.fg("active", "");

export interface ReasonerInfo {
	id: string;
	name: string;
	description: string;
	active: boolean;
}

export type ReasonerSelectorAction =
	| { type: "select"; reasoner: ReasonerInfo }
	| { type: "close" };

export class ReasonerSelectorOverlay implements Component {
	public visible = false;
	private reasoners: ReasonerInfo[] = [];
	private selectedIndex = 0;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setReasoners(reasoners: ReasonerInfo[]): void {
		this.reasoners = reasoners;
		if (this.selectedIndex >= this.reasoners.length) {
			this.selectedIndex = Math.max(0, this.reasoners.length - 1);
		}
		this.invalidate();
	}

	setMessage(message: string): void {
		this.message = message;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): ReasonerSelectorAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			const reasoner = this.reasoners[this.selectedIndex];
			return reasoner ? { type: "select", reasoner } : { type: "close" };
		}
		if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
			this.moveSelection(-1);
			return null;
		}
		if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
			this.moveSelection(1);
			return null;
		}
		if (data === "\x1b[5~") {
			this.moveSelection(-8);
			return null;
		}
		if (data === "\x1b[6~") {
			this.moveSelection(8);
			return null;
		}
		return null;
	}

	invalidate(): void {
		this.cachedLines = null;
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}
		this.cachedWidth = width;

		if (!this.visible) return [];

		const overlayWidth = Math.max(48, Math.min(width, 110));
		const innerWidth = Math.max(1, overlayWidth - 4);
		const lines: string[] = [];

		lines.push(`${getHeader()}┌${"─".repeat(overlayWidth - 2)}┐${RESET}`);
		lines.push(
			boxLine(
				`${BOLD}Reasoning Mode${RESET}${DIM} (${this.reasoners.length})${RESET}`,
				"↑↓ select · enter confirm · esc close",
				innerWidth,
			),
		);
		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);

		if (!this.reasoners.length) {
			lines.push(
				boxLine(
					`${getMuted()}No reasoning modes available.${RESET}`,
					"",
					innerWidth,
				),
			);
		} else {
			const maxRows = 10;
			const start = Math.max(
				0,
				Math.min(
					this.selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, this.reasoners.length - maxRows),
				),
			);
			const end = Math.min(this.reasoners.length, start + maxRows);
			if (start > 0) {
				lines.push(
					boxLine(`${getMuted()}↑ ${start} more${RESET}`, "", innerWidth),
				);
			}
			for (let i = start; i < end; i++) {
				const r = this.reasoners[i];
				const selected = i === this.selectedIndex;
				const cursor = selected ? "▸" : " ";
				const activeMark = r.active ? `${getActive()}● active${RESET}` : "";
				const name = selected
					? `${getSelected()}${BOLD}${r.name}${RESET}`
					: r.name;
				const desc = `${DIM}${r.description}${RESET}`;
				const meta = activeMark ? `${desc}  ${activeMark}` : desc;
				lines.push(boxLine(`${cursor} ${name}`, meta, innerWidth));
			}
			if (end < this.reasoners.length) {
				lines.push(
					boxLine(
						`${getMuted()}↓ ${this.reasoners.length - end} more${RESET}`,
						"",
						innerWidth,
					),
				);
			}
		}

		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);
		lines.push(
			boxLine(
				this.message
					? `${DIM}${this.message}${RESET}`
					: `${getMuted()}Select a reasoning mode for the next turn.${RESET}`,
				"",
				innerWidth,
			),
		);
		lines.push(`${getHeader()}└${"─".repeat(overlayWidth - 2)}┘${RESET}`);

		this.cachedLines = lines.map((line) => clampLineToWidth(line, width));
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.reasoners.length;
		if (!n) return;
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}
}

function boxLine(left: string, right: string, width: number): string {
	const leftWidth = visibleWidth(left);
	const rightWidth = visibleWidth(right);
	const gap = Math.max(1, width - leftWidth - rightWidth);
	const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
	const pad = Math.max(0, width - visibleWidth(content));
	return `${getHeader()}│${RESET} ${content}${" ".repeat(pad)} ${getHeader()}│${RESET}`;
}
