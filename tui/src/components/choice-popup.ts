// ── ChoicePopup — agent Q&A dropdown ────────────────────────────────────────
// Overlay popup that lets the agent ask questions with selectable options.
// The user navigates with ↑/↓ and confirms with Enter (or Tab to accept).
// The selected answer is sent back via onSubmit so the bridge can resolve
// the agent's tool call / pending question.

import { type Component, visibleWidth } from "../tui-core.ts";

export interface ChoiceItem {
	/** The value sent back to the agent when selected. */
	value: string;
	/** Display label for the user. */
	label: string;
}

export interface ChoicePopupOptions {
	/** The question id (unique identifier for the agent to track). */
	questionId?: string;
	/** The question being asked. */
	question: string;
	/** List of selectable options. */
	choices: ChoiceItem[];
}

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const QUESTION_COLOR = "\x1b[38;5;159m"; // aqua
const SELECTED_COLOR = "\x1b[38;5;111m"; // green
const KEY_COLOR = "\x1b[38;5;245m"; // dim gray

export class ChoicePopup implements Component {
	private question = "";
	private questionId = "";
	private choices: ChoiceItem[] = [];
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private onSubmit?: (selected: ChoiceItem) => void;
	private onDismiss?: () => void;

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

	select(): void {
		if (this.choices.length === 0) return;
		const selected = this.choices[this.selectedIndex];
		this._lastSelected = selected;
		this.onSubmit?.(selected);
		this.hide();
	}

	selectFirst(): void {
		if (this.choices.length === 0) return;
		const selected = this.choices[0];
		this._lastSelected = selected;
		this.onSubmit?.(selected);
		this.hide();
	}

	hide(userDismissed = false): void {
		this.visible = false;
		if (userDismissed && this.onDismiss) {
			this.onDismiss();
		}
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	// ── Input handling (called from the TUI input listener) ─────────────────

	handleInput(data: string): void {
		if (data === "\r" || data === "\n") {
			this.select();
			return;
		}

		if (data === "\x1b" || data === "\x03") {
			// Escape — dismiss without selecting; send null to let agent know.
			this._lastSelected = null;
			this.hide(true);
			return;
		}

		if (data === "\t") {
			// Tab — accept the first option
			this.selectFirst();
			return;
		}

		// Up arrow
		if (data === "\x1b[A" || data === "\x1bOA") {
			this.moveSelection(-1);
			return;
		}

		// Down arrow
		if (data === "\x1b[B" || data === "\x1bOB") {
			this.moveSelection(1);
			return;
		}

		// Number keys 1-9 — select that option directly
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c >= 0x31 && c <= 0x39) {
				const idx = c - 0x31;
				if (idx < this.choices.length) {
					this.selectedIndex = idx;
					this.select();
				}
				return;
			}
		}
	}

	/** The last selected item (or null if dismissed). */
	_lastSelected: ChoiceItem | null = null;

	/** The value of the last selection, or null if dismissed. */
	getSelectedValue(): string | null {
		return this._lastSelected?.value ?? null;
	}

	setOnSubmit(cb: (selected: ChoiceItem) => void): void {
		this.onSubmit = cb;
	}

	setOnDismiss(cb: () => void): void {
		this.onDismiss = cb;
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

		if (this.choices.length === 0) return [];

		const contentWidth = Math.min(80, Math.max(40, width - 4));
		const lines: string[] = [];

		// Question line
		const qLine = ` ${QUESTION_COLOR}${BOLD}${this.question.slice(0, contentWidth)}${RESET}`;
		lines.push(qLine);

		// Choices
		for (let i = 0; i < this.choices.length; i++) {
			const ch = this.choices[i];
			const isSelected = i === this.selectedIndex;
			const numLabel = `${DIM}${i + 1}.${RESET}`;

			// Key hint: show the key in brackets (e.g. [1], [2])
			const keyHint = isSelected
				? ` ${KEY_COLOR}[${i + 1}]${RESET}`
				: ` ${KEY_COLOR}[${i + 1}]${RESET}`;

			let label = ch.label;
			// Truncate long labels to fit
			const prefixWidth = visibleWidth(`${keyHint} `) + numLabel.length + 1;
			const maxLabel = contentWidth - prefixWidth - 2;
			if (maxLabel > 0 && visibleWidth(label) > maxLabel) {
				// Strip ANSI codes, truncate, re-add reset
				const plain = label.replace(/\x1b\[[0-9;]*m/g, "");
				if (plain.length > maxLabel) {
					label = plain.slice(0, maxLabel - 1) + "…";
				}
			}

			const line = isSelected
				? ` ${SELECTED_COLOR}${keyHint} ${numLabel} ${BOLD}${label}${RESET}`
				: `   ${keyHint}   ${numLabel} ${label}`;
			lines.push(line);
		}

		// Hint bar
	 const hint = `${DIM}↑↓ navigate · 1-9 select · ⏎ confirm · Tab accept · Esc dismiss${RESET}`;
		lines.push(`  ${hint}`);

		this.cachedLines = lines;
		return lines;
	}
}
