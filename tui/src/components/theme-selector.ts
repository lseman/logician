// ── ThemeSelectorOverlay ───────────────────────────────────────────────────────
// Overlay for selecting a color theme.
// Pattern: list, select, confirm, close.
// Mirrors ReasonerSelectorOverlay / PluginManagerOverlay.

import { type Component, clampLineToWidth, visibleWidth } from "../tui-core.ts";
import { theme } from "../theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";

// Lazy getters to avoid accessing theme before initTheme() is called
const HEADER = (): string => theme.fg("header", "");
const SELECTED = (): string => theme.fg("selected", "");
const MUTED = (): string => theme.fg("muted", "");

export interface ThemeInfo {
	name: string;
	description: string;
}

export type ThemeSelectorAction =
	| { type: "select"; theme: ThemeInfo }
	| { type: "close" };

export class ThemeSelectorOverlay implements Component {
	public visible = false;
	private themes: ThemeInfo[] = [];
	private selectedIndex = 0;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setThemes(themes: ThemeInfo[]): void {
		this.themes = themes;
		if (this.selectedIndex >= this.themes.length) {
			this.selectedIndex = Math.max(0, this.themes.length - 1);
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

	handleInput(data: string): ThemeSelectorAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			const themeInfo = this.themes[this.selectedIndex];
			return themeInfo
				? { type: "select", theme: themeInfo }
				: { type: "close" };
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

		lines.push(`${HEADER()}┌${"─".repeat(overlayWidth - 2)}┐${RESET}`);
		lines.push(
			boxLine(
				`${BOLD}Theme${RESET}${DIM} (${this.themes.length})${RESET}`,
				"↑↓ select · enter confirm · esc close",
				innerWidth,
			),
		);
		lines.push(`${HEADER()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);

		if (!this.themes.length) {
			lines.push(
				boxLine(`${MUTED()}No themes available.${RESET}`, "", innerWidth),
			);
		} else {
			const maxRows = 10;
			const start = Math.max(
				0,
				Math.min(
					this.selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, this.themes.length - maxRows),
				),
			);
			const end = Math.min(this.themes.length, start + maxRows);
			if (start > 0) {
				lines.push(
					boxLine(`${MUTED()}↑ ${start} more${RESET}`, "", innerWidth),
				);
			}
			for (let i = start; i < end; i++) {
				const t = this.themes[i];
				const selected = i === this.selectedIndex;
				const cursor = selected ? "▸" : " ";
				const name = selected
					? `${SELECTED()}${BOLD}${t.name}${RESET}`
					: t.name;
				const desc = `${DIM}${t.description}${RESET}`;
				lines.push(boxLine(`${cursor} ${name}`, desc, innerWidth));
			}
			if (end < this.themes.length) {
				lines.push(
					boxLine(
						`${MUTED()}↓ ${this.themes.length - end} more${RESET}`,
						"",
						innerWidth,
					),
				);
			}
		}

		lines.push(`${HEADER()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);
		lines.push(
			boxLine(
				this.message
					? `${DIM}${this.message}${RESET}`
					: `${MUTED()}Select a color theme.${RESET}`,
				"",
				innerWidth,
			),
		);
		lines.push(`${HEADER()}└${"─".repeat(overlayWidth - 2)}┘${RESET}`);

		this.cachedLines = lines.map((line) => clampLineToWidth(line, width));
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.themes.length;
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
	return `${HEADER()}│${RESET} ${content}${" ".repeat(pad)} ${HEADER()}│${RESET}`;
}
