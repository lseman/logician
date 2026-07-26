// ── PermissionPopup — tool permission overlay popup ──────────────────────────
// Rounded-corner overlay for tool permission requests with selectable
// allow-once / always-allow / deny options.

import {
	type Component,
	visibleWidth,
} from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import {
	renderQuestion,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	type ChoiceOption,
	renderChoiceOption,
} from "./popup-utils.ts";

export interface PermissionChoice {
	/** Value sent back to the agent: "allow" | "always" | "deny" */
	value: "allow" | "always" | "deny";
	/** Display label */
	label: string;
	/** Short description */
	description?: string;
}

export type PermissionPopupAction =
	| { type: "select"; choice: PermissionChoice }
	| { type: "close" };

export class PermissionPopup implements Component {
	private toolName = "";
	private toolArgs = "";
	private choices: PermissionChoice[] = [];
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setToolInfo(toolName: string, toolArgs: string): void {
		this.toolName = toolName;
		this.toolArgs = toolArgs;
		this.selectedIndex = 0;
		this.invalidate();
	}

	setChoices(choices: PermissionChoice[]): void {
		this.choices = choices;
		this.selectedIndex = 0;
		this.invalidate();
	}

	getSelected(): PermissionChoice | null {
		if (this.choices.length === 0) return null;
		return this.choices[this.selectedIndex];
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	// ── Navigation ─────────────────────────────────────────────────────────

	moveSelection(delta: number): void {
		if (this.choices.length === 0) return;
		this.selectedIndex =
			(this.selectedIndex + delta + this.choices.length) % this.choices.length;
		this.invalidate();
	}

	// ── Input handling (called from the TUI input listener) ─────────────────

	handleInput(data: string): PermissionPopupAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03") {
			return { type: "close" };
		}

		if (data === "\r" || data === "\n") {
			const selected = this.getSelected();
			if (selected) return { type: "select", choice: selected };
			return null;
		}

		if (data === "\x1b[A" || data === "\x1bOA") {
			this.moveSelection(-1);
			return null;
		}

		if (data === "\x1b[B" || data === "\x1bOB") {
			this.moveSelection(1);
			return null;
		}

		// Number keys 1-9 — select that option directly
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c >= 0x31 && c <= 0x39) {
				const idx = c - 0x31;
				if (idx < this.choices.length) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
				return null;
			}
		}

		// Single-letter shortcuts: a=allow, v=always, n=deny
		if (data.length === 1) {
			const lower = data.toLowerCase();
			if (lower === "a" && this.choices.some(c => c.value === "allow")) {
				const idx = this.choices.findIndex(c => c.value === "allow");
				if (idx !== -1) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
			}
			if (lower === "v" && this.choices.some(c => c.value === "always")) {
				const idx = this.choices.findIndex(c => c.value === "always");
				if (idx !== -1) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
			}
			if (lower === "n" && this.choices.some(c => c.value === "deny")) {
				const idx = this.choices.findIndex(c => c.value === "deny");
				if (idx !== -1) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
			}
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

		const popupWidth = Math.max(48, Math.min(width, 120));
		const innerWidth = Math.max(1, popupWidth - 4);
		const lines: string[] = [];

		const headerFg = theme.fg("header", "");
		const muted = theme.fg("muted", "");
		const dim = "\x1b[2m";
		const reset = "\x1b[0m";
		const bold = "\x1b[1m";

		// ── Top border ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${muted}`);

		// ── Title row ──
		const hintsText = " ↑↓ select · enter confirm · esc close";
		const titleLine = `${bold}Permission${muted} (${this.choices.length})${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Tool name ──
		lines.push(renderQuestion(this.toolName, innerWidth));

		// ── Tool args preview ──
		if (this.toolArgs) {
			const argsPreview = this.toolArgs.length > innerWidth
				? this.toolArgs.slice(0, innerWidth - 3) + "…"
				: this.toolArgs;
			const argsLine = `${dim}${argsPreview}${reset}`;
			const argsVisible = visibleWidth(argsLine);
			const argsPad = Math.max(0, innerWidth - argsVisible);
			lines.push(` ${argsLine}${" ".repeat(argsPad + 1)}`);
		}

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Choices ──
		if (this.choices.length > 0) {
			const maxRows = 6;
			const start = Math.max(
				0,
				Math.min(
					this.selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, this.choices.length - maxRows),
				),
			);
			const end = Math.min(this.choices.length, start + maxRows);
			if (start > 0) {
				lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
			}
			for (let i = start; i < end; i++) {
				const ch = this.choices[i];
				const option: ChoiceOption = {
					label: ch.label,
					value: ch.value,
					selected: i === this.selectedIndex,
					description: ch.description,
				};
				lines.push(renderChoiceOption(option, innerWidth, i));
			}
			if (end < this.choices.length) {
				lines.push(renderStatusLine(`↓ ${this.choices.length - end} more`, innerWidth));
			}
		} else {
			lines.push("");
		}

		// ── Bottom hint ──
		lines.push(renderSeparator(popupWidth));
		const bottomText = this.choices.length > 0
			? "Select an option to decide."
			: "Deny by default.";
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom border ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${muted}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}
