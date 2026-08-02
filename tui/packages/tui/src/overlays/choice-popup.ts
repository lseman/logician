// ── ChoicePopup — beautiful agent Q&A popup ────────────────────────────────
// Rounded-corner overlay popup for agent questions with numbered selectable options.
// Uses the shared popup-utils design system.

import type { InkListOverlayModel } from "./ink-overlay-model.ts";

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

export interface ChoiceQuestion {
	id: string;
	header?: string;
	question: string;
	choices: ChoiceItem[];
}

export type ChoicePopupAction =
	| { type: "submit"; answers: Record<string, string> }
	| { type: "close" };

export class ChoicePopup {
	private questionId = "";
	private questions: ChoiceQuestion[] = [];
	private currentTab = 0;
	private selectedIndices: number[] = [];
	private answers = new Map<string, string>();
	public visible = false;

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
		this.selectedIndices = questions.map(() => 0);
		this.answers.clear();
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
		return question.choices[this.selectedIndices[this.currentTab] ?? 0];
	}

	getAnswers(): Record<string, string> {
		return Object.fromEntries(this.answers);
	}

	getResponseValue(): string {
		const answers = this.getAnswers();
		return this.questions.length === 1
			? (answers[this.questions[0]?.id ?? "answer"] ?? "")
			: JSON.stringify(answers);
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
			const item =
				question?.choices[this.selectedIndices[this.currentTab] ?? 0];
			if (!question || !item) return null;
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

	invalidate(): void {
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		const question = this.questions[this.currentTab];
		if (!question) {
			return {
				kind: "list",
				title: "ASK",
				subtitle: ` · ${this.answers.size}/${this.questions.length} answered`,
				hints: "tab questions · enter submit · esc dismiss",
				items: this.questions.map((item) => {
					const answer = this.answers.get(item.id);
					const choice = item.choices.find((candidate) => candidate.value === answer);
					return {
						label: item.header || item.question,
						metadata: choice?.label ?? "Not answered",
						current: Boolean(answer),
					};
				}),
				emptyText: "No questions available.",
				footer: this.answers.size === this.questions.length
					? "Ready to submit."
					: "Answer every question before submitting.",
				selectedIndex: 0,
			};
		}
		const selectedIndex = this.selectedIndices[this.currentTab] ?? 0;
		return {
			kind: "list",
			title: "ASK",
			subtitle: this.questions.length > 1
				? ` · ${this.answers.size}/${this.questions.length} answered`
				: " · choose one",
			hints: "↑↓ choose · enter confirm · esc dismiss",
			headerLines: [question.header || `Question ${this.currentTab + 1}`, question.question],
			items: question.choices.map((choice, index) => ({
				label: choice.label,
				metadata: choice.description,
				selected: index === selectedIndex,
				current: this.answers.get(question.id) === choice.value,
			})),
			emptyText: "No choices available.",
			footer: this.questions.length > 1 ? "Tab moves between questions and Submit." : "Select one option.",
			selectedIndex,
		};
	}

}
