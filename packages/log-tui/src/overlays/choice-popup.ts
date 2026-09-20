// ── ChoicePopup — beautiful agent Q&A popup ────────────────────────────────
// Rounded-corner overlay popup for agent questions with numbered selectable options.
// Uses the shared popup-utils design system.

import { wrapText } from "../rendering/transcript/layout.ts";
import {
	BOLD,
	type Component,
	clampLineToWidth,
	DIM,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";

export interface ChoiceItem {
	/** The value sent back to the agent when selected. */
	value: string;
	/** Display label for the user. */
	label: string;
	/** Optional short description shown on the right. */
	description?: string;
	/** Selecting this choice opens free-text entry instead of submitting `value` directly. */
	isFreeText?: boolean;
}

export interface ChoiceQuestion {
	id: string;
	header?: string;
	question: string;
	choices: ChoiceItem[];
	/** Allow selecting more than one choice. Answer is returned as a string array. */
	multi?: boolean;
	/** Value of the choice to pre-select and mark as recommended. No auto-timeout. */
	recommended?: string;
}

export type ChoicePopupAction =
	| { type: "submit"; answers: Record<string, string | string[]> }
	| { type: "close" };

/** Render an answer (single value, multi-select array, or free text) for the submit summary. */
function summarizeAnswer(
	answer: string | string[] | undefined,
	choices: ChoiceItem[],
): string {
	if (answer === undefined) return "Not answered";
	const labelFor = (value: string) =>
		choices.find(c => c.value === value)?.label ?? value;
	if (Array.isArray(answer)) {
		return answer.length ? answer.map(labelFor).join(", ") : "Not answered";
	}
	return labelFor(answer);
}

export class ChoicePopup implements Component {
	private questionId = "";
	private questions: ChoiceQuestion[] = [];
	private currentTab = 0;
	private selectedIndices: number[] = [];
	/** Per-question toggled indices, for `multi` questions only. */
	private multiSelections: Set<number>[] = [];
	private answers = new Map<string, string | string[]>();
	/** Index of the question currently in free-text entry mode, or null. */
	private typingTab: number | null = null;
	private typingBuffer = "";
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setQuestion(q: string): void {
		const choices = this.questions[0]?.choices ?? [];
		this.setQuestions([{ id: "answer", question: q, choices }]);
	}

	setQuestionId(id: string): void {
		this.questionId = id;
	}

	getQuestionId(): string {
		return this.questionId;
	}

	setChoices(choices: ChoiceItem[]): void {
		const question = this.questions[0]?.question ?? "";
		this.setQuestions([{ id: "answer", question, choices }]);
	}

	setQuestions(questions: ChoiceQuestion[]): void {
		this.questions = questions;
		this.currentTab = 0;
		this.selectedIndices = questions.map(q => {
			if (q.recommended === undefined) return 0;
			const idx = q.choices.findIndex(c => c.value === q.recommended);
			return idx >= 0 ? idx : 0;
		});
		this.multiSelections = questions.map(() => new Set<number>());
		this.answers.clear();
		this.typingTab = null;
		this.typingBuffer = "";
		this.invalidate();
	}

	getQuestion(): string {
		return this.questions[0]?.question ?? "";
	}

	getChoices(): ChoiceItem[] {
		return this.questions[0]?.choices ?? [];
	}

	getSelectedIndex(): number {
		return this.selectedIndices[this.currentTab] ?? 0;
	}

	getSelected(): ChoiceItem | null {
		const question = this.questions[this.currentTab] ?? this.questions[0];
		if (!question?.choices.length) return null;
		return question.choices[this.selectedIndices[this.currentTab] ?? 0] ?? null;
	}

	getAnswers(): Record<string, string | string[]> {
		return Object.fromEntries(this.answers);
	}

	getResponseValue(): string {
		const answers = this.getAnswers();
		const single = this.questions[0];
		if (this.questions.length === 1 && single && !single.multi) {
			const value = answers[single.id];
			return typeof value === "string" ? value : "";
		}
		return JSON.stringify(answers);
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	// ── Navigation ─────────────────────────────────────────────────────────

	moveSelection(delta: number): void {
		const choices = this.questions[this.currentTab]?.choices ?? [];
		if (!choices.length) return;
		const current = this.selectedIndices[this.currentTab] ?? 0;
		this.selectedIndices[this.currentTab] =
			(current + delta + choices.length) % choices.length;
		this.invalidate();
	}

	private moveTab(delta: number): void {
		const count = this.questions.length + (this.questions.length > 1 ? 1 : 0);
		if (!count) return;
		this.currentTab = (this.currentTab + delta + count) % count;
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

		if (this.typingTab !== null) {
			return this.handleTypingInput(data);
		}

		if (data === "\x1b" || data === "\x03") {
			return { type: "close" };
		}

		if (data === "\r" || data === "\n") {
			if (this.currentTab === this.questions.length) {
				return this.answers.size === this.questions.length
					? { type: "submit", answers: this.getAnswers() }
					: null;
			}
			const question = this.questions[this.currentTab];
			if (!question) return null;

			if (question.multi) {
				const selected = this.multiSelections[this.currentTab];
				if (!selected?.size) return null;
				const values = [...selected]
					.sort((a, b) => a - b)
					.map(i => question.choices[i]?.value)
					.filter((v): v is string => v !== undefined);
				this.answers.set(question.id, values);
				if (this.questions.length === 1) {
					return { type: "submit", answers: this.getAnswers() };
				}
				this.moveTab(1);
				return null;
			}

			const item = question.choices[this.selectedIndices[this.currentTab] ?? 0];
			if (!item) return null;
			if (item.isFreeText) {
				this.typingTab = this.currentTab;
				this.typingBuffer = "";
				this.invalidate();
				return null;
			}
			this.answers.set(question.id, item.value);
			if (this.questions.length === 1) {
				return { type: "submit", answers: this.getAnswers() };
			}
			this.moveTab(1);
			return null;
		}

		if (
			data === "\x1b[A" ||
			data === "\x1bOA" ||
			data === "k" ||
			data === "\x10"
		) {
			this.moveSelection(-1);
			return null;
		}

		if (
			data === "\x1b[B" ||
			data === "\x1bOB" ||
			data === "j" ||
			data === "\x0e"
		) {
			this.moveSelection(1);
			return null;
		}

		if (data === "\t") {
			this.moveTab(1);
			return null;
		}
		if (data === "\x1b[Z") {
			this.moveTab(-1);
			return null;
		}

		// Space — toggle the highlighted choice for `multi` questions.
		if (data === " ") {
			const question = this.questions[this.currentTab];
			if (question?.multi) {
				const index = this.selectedIndices[this.currentTab] ?? 0;
				const selected = this.multiSelections[this.currentTab];
				if (selected) {
					if (selected.has(index)) selected.delete(index);
					else selected.add(index);
					this.invalidate();
				}
			}
			return null;
		}

		// Number keys 1-9 — select that option directly
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c >= 0x31 && c <= 0x39) {
				const idx = c - 0x31;
				const question = this.questions[this.currentTab];
				if (question && idx < question.choices.length) {
					this.selectedIndices[this.currentTab] = idx;
					this.invalidate();
				}
				return null;
			}
		}

		return null;
	}

	/** Route keystrokes into the free-text buffer while `typingTab` is set. */
	private handleTypingInput(data: string): ChoicePopupAction | null {
		if (data === "\x1b") {
			// Cancel free-text entry, return to the choice list.
			this.typingTab = null;
			this.typingBuffer = "";
			this.invalidate();
			return null;
		}
		if (data === "\x03") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			const tab = this.typingTab;
			const question = tab === null ? undefined : this.questions[tab];
			if (tab === null || !question) return null;
			const value = this.typingBuffer.trim();
			this.typingTab = null;
			this.typingBuffer = "";
			if (!value) {
				this.invalidate();
				return null;
			}
			this.answers.set(question.id, value);
			if (this.questions.length === 1) {
				return { type: "submit", answers: this.getAnswers() };
			}
			this.currentTab = tab;
			this.moveTab(1);
			return null;
		}
		if (data === "\x7f" || data === "\x08") {
			this.typingBuffer = this.typingBuffer.slice(0, -1);
			this.invalidate();
			return null;
		}
		// Reject other control/escape sequences (arrow keys, etc.) — only
		// accept plain printable input as free-text content.
		if (data.length > 0 && data.charCodeAt(0) >= 0x20) {
			this.typingBuffer += data;
			this.invalidate();
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

		const popupWidth = Math.max(1, width);
		const innerWidth = Math.max(1, popupWidth - 4);
		const lines: string[] = [];
		const accent = theme.fgRaw("accent");
		const selectedColor = theme.fgRaw("selected");
		const text = theme.fgRaw("text");
		const muted = theme.fgRaw("muted");
		const active = theme.fgRaw("active");
		const border = theme.fgRaw("borderMuted");
		const line = (content = ""): string => {
			const clipped = clampLineToWidth(content, innerWidth);
			return `${border}│${RESET} ${clipped}${" ".repeat(
				Math.max(0, innerWidth - visibleWidth(clipped)),
			)} ${border}│${RESET}`;
		};

		lines.push(`${border}╭${"─".repeat(popupWidth - 2)}╮${RESET}`);
		lines.push(
			line(
				`${accent}${BOLD}ASK${RESET}${muted}  ${this.questions.length > 1 ? `${this.answers.size}/${this.questions.length} answered` : "choose one"}${RESET}`,
			),
		);
		if (this.questions.length > 1) {
			const tabs = this.questions.map((question, index) => {
				const activeTab = index === this.currentTab;
				const answered = this.answers.has(question.id);
				const color = activeTab ? selectedColor : muted;
				const mark = answered ? "✓" : "□";
				return `${color}${activeTab ? BOLD : ""}${mark} ${question.header || `Question ${index + 1}`}${RESET}`;
			});
			const submitColor =
				this.currentTab === this.questions.length ? selectedColor : muted;
			tabs.push(
				`${submitColor}${this.currentTab === this.questions.length ? BOLD : ""}✓ Submit${RESET}`,
			);
			lines.push(line(`←  ${tabs.join(` ${muted}·${RESET} `)}  →`));
		}
		lines.push(line());

		const activeQuestion = this.questions[this.currentTab];
		if (!activeQuestion) {
			const complete = this.answers.size === this.questions.length;
			lines.push(
				line(
					`${text}${BOLD}${complete ? "Ready to submit your answers?" : "Answer every question before submitting."}${RESET}`,
				),
			);
			lines.push(line());
			for (const question of this.questions) {
				const answer = this.answers.get(question.id);
				const summary = summarizeAnswer(answer, question.choices);
				lines.push(
					line(
						`${answer !== undefined ? active : muted}${answer !== undefined ? "✓" : "□"} ${question.header || question.question}: ${summary}${RESET}`,
					),
				);
			}
			lines.push(line());
			lines.push(
				line(
					`${muted}${BOLD}tab${RESET}${muted} questions   ${BOLD}enter${RESET}${muted} submit   ${BOLD}esc${RESET}${muted} dismiss${RESET}`,
				),
			);
			lines.push(`${border}╰${"─".repeat(popupWidth - 2)}╯${RESET}`);
			this.cachedLines = lines.map(value => clampLineToWidth(value, width));
			return this.cachedLines;
		}

		const questionLines = wrapText(
			activeQuestion.question || "What should we do?",
			Math.max(1, innerWidth - 2),
		);
		for (const questionLine of questionLines) {
			lines.push(line(`${text}${BOLD}${questionLine}${RESET}`));
		}
		lines.push(line());

		// ── Free-text entry ──
		if (this.typingTab === this.currentTab) {
			lines.push(
				line(
					`${selectedColor}❯ ${text}${this.typingBuffer}${selectedColor}▏${RESET}`,
				),
			);
			lines.push(line());
			lines.push(
				line(
					`${muted}${BOLD}enter${RESET}${muted} submit   ${BOLD}esc${RESET}${muted} back to choices${RESET}`,
				),
			);
			lines.push(`${border}╰${"─".repeat(popupWidth - 2)}╯${RESET}`);
			this.cachedLines = lines.map(value => clampLineToWidth(value, width));
			return this.cachedLines;
		}

		// ── Choices ──
		const choices = activeQuestion.choices;
		const selectedIndex = this.selectedIndices[this.currentTab] ?? 0;
		const toggled = this.multiSelections[this.currentTab];
		if (choices.length > 0) {
			const maxRows = 10;
			const start = Math.max(
				0,
				Math.min(
					selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, choices.length - maxRows),
				),
			);
			const end = Math.min(choices.length, start + maxRows);
			if (start > 0) lines.push(line(`${muted}  ↑ ${start} more${RESET}`));
			for (let i = start; i < end; i++) {
				const ch = choices[i];
				if (!ch) continue;
				const cursor = i === selectedIndex;
				const isRecommended = ch.value === activeQuestion.recommended;
				const labelColor = cursor ? `${selectedColor}${BOLD}` : text;
				let marker: string;
				if (activeQuestion.multi) {
					const on = toggled?.has(i) ?? false;
					marker = on
						? `${selectedColor}${BOLD}[x]${RESET}`
						: `${cursor ? selectedColor : muted}[ ]${RESET}`;
				} else {
					marker = cursor ? `${selectedColor}●${RESET}` : `${muted}○${RESET}`;
				}
				const recommendedTag = isRecommended
					? ` ${muted}${DIM}(recommended)${RESET}`
					: "";
				lines.push(
					line(
						`${marker} ${labelColor}${ch.label}${RESET}${recommendedTag}${muted}  ${i + 1}${RESET}`,
					),
				);
				if (ch.description) {
					for (const descriptionLine of wrapText(
						ch.description,
						Math.max(1, innerWidth - 4),
					)) {
						lines.push(
							line(
								`${cursor ? active : muted}   ${DIM}${descriptionLine}${RESET}`,
							),
						);
					}
				}
			}
			if (end < choices.length) {
				lines.push(line(`${muted}  ↓ ${choices.length - end} more${RESET}`));
			}
		} else {
			lines.push(line(`${muted}No options available${RESET}`));
		}

		lines.push(line());
		const multiHint = activeQuestion.multi
			? `${BOLD}space${RESET}${muted} toggle   `
			: "";
		lines.push(
			line(
				`${muted}${BOLD}↑↓${RESET}${muted} move   ${multiHint}${BOLD}enter${RESET}${muted} answer   ${this.questions.length > 1 ? `${BOLD}tab${RESET}${muted} questions   ` : ""}${BOLD}esc${RESET}${muted} dismiss${RESET}`,
			),
		);
		lines.push(`${border}╰${"─".repeat(popupWidth - 2)}╯${RESET}`);

		this.cachedLines = lines.map(value => clampLineToWidth(value, width));
		return this.cachedLines;
	}
}
