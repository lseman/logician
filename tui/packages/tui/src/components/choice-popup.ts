// ── ChoicePopup — beautiful agent Q&A popup ────────────────────────────────
// Rounded-corner overlay popup for agent questions with numbered selectable options.
// Uses the shared popup-utils design system.

import {
	type Component,
	clampLineToWidth,
	visibleWidth,
} from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import {
	BOX,
	renderQuestion,
	renderChoiceOption,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	type ChoiceOption,
} from "./popup-utils.ts";

export interface ChoiceItem {
	/** The value sent back to the agent when selected. */
	value: string;
	/** Display label for the user. */
	label: string;
	/** Optional short description shown on the right. */
	description?: string;
}

export interface ChoicePopupOptions {
	/** The question id (unique identifier for the agent to track). */
	questionId?: string;
	/** The question being asked. */
	question: string;
	/** List of selectable options. */
	choices: ChoiceItem[];
}

export type ChoicePopupAction =
	| { type: "select"; item: ChoiceItem }
	| { type: "close" };

export class ChoicePopup implements Component {
	private question = "";
	private questionId = "";
	private choices: ChoiceItem[] = [];
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setQuestion(q: string): void {
		this.question = q;
		this.invalidate();
	}

	setQuestionId(id: string): void {
		this.questionId = id;
	}

	getQuestionId(): string {
		return this.questionId;
	}

	setChoices(choices: ChoiceItem[]): void {
		this.choices = choices;
		this.selectedIndex = 0;
		this.invalidate();
	}

	getQuestion(): string {
		return this.question;
	}

	getChoices(): ChoiceItem[] {
		return this.choices;
	}

	getSelectedIndex(): number {
		return this.selectedIndex;
	}

	getSelected(): ChoiceItem | null {
		if (this.choices.length === 0) return null;
		return this.choices[this.selectedIndex];
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	// ── Navigation ─────────────────────────────────────────────────────────

	moveSelection(delta: number): void {
		if (this.choices.length === 0) return;
		this.selectedIndex =
			(this.selectedIndex + delta + this.choices.length) % this.choices.length;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	// ── Input handling (called from the TUI input listener) ─────────────────

	handleInput(data: string): ChoicePopupAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03") {
			return { type: "close" };
		}

		if (data === "\r" || data === "\n") {
			if (this.choices.length === 0) return null;
			return {
				type: "select",
				item: this.choices[this.selectedIndex],
			};
		}

		if (data === "\x1b[A" || data === "\x1bOA") {
			this.moveSelection(-1);
			return null;
		}

		if (data === "\x1b[B" || data === "\x1bOB") {
			this.moveSelection(1);
			return null;
		}

		if (data === "\t") {
			if (this.choices.length === 0) return null;
			return { type: "select", item: this.choices[0] };
		}

		// Number keys 1-9 — select that option directly
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c >= 0x31 && c <= 0x39) {
				const idx = c - 0x31;
				if (idx < this.choices.length) {
					this.selectedIndex = idx;
					return { type: "select", item: this.choices[idx] };
				}
				return null;
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

		const popupWidth = Math.max(48, Math.min(width, 110));
		const innerWidth = popupWidth - 4;
		const lines: string[] = [];

		// ── Top rounded corner ──
		const headerFg = theme.fg("header", "");
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		// ── Title row ──
		const titleText = "Question";
		const subtitleText = ` (${this.choices.length})`;
		const hintsText = " ↑↓ select · enter confirm · esc close";
		const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

		// ── Separator ──
		lines.push(renderSeparator(popupWidth, 1));

		// ── Question text ──
		if (this.question) {
			lines.push(renderQuestion(this.question, innerWidth));
		} else {
			lines.push("");
		}

		// ── Separator ──
		lines.push(renderSeparator(popupWidth, 1));

		// ── Choices ──
		if (this.choices.length > 0) {
			const maxRows = 10;
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
		lines.push(renderSeparator(popupWidth, 1));
		const bottomText = this.question
			? "Select an option to answer."
			: "";
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom rounded corner ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}
