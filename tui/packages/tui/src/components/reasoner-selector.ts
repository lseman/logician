// ── ReasonerSelectorOverlay ──────────────────────────────────────────────────────
// Overlay for selecting an active reasoning mode.
// Mirrors PluginManagerOverlay pattern: list, select, confirm, close.
// Reasoner selection applies to the next turn (never mutates an in-flight run).

import { type Component, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import { SelectorController } from "./selector-controller.ts";
import {
	renderListItem,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
	type ListItem,
} from "./popup-utils.ts";

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
	private selection = new SelectorController();
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setReasoners(reasoners: ReasonerInfo[]): void {
		this.reasoners = reasoners;
		this.selection.set(this.selection.index, this.reasoners.length);
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
			const reasoner = this.reasoners[this.selection.index];
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

		const popupWidth = Math.max(1, width);
		const innerWidth = Math.max(1, popupWidth - POPUP_FRAME_OVERHEAD);
		const lines: string[] = [];

		const headerFg = theme.fg("header", "");

		// ── Top rule ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		// ── Title row ──
		const titleText = "Reasoning Mode";
		const subtitleText = ` (${this.reasoners.length})`;
		const hintsText = " ↑↓ select · enter confirm · esc close";
		const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Reasoner list ──
		if (!this.reasoners.length) {
			lines.push(
				renderStatusLine(
					"No reasoning modes available.",
					innerWidth,
					theme.fg("warning", ""),
				),
			);
		} else {
			const maxRows = 10;
			const { start, end } = this.selection.window(this.reasoners.length, maxRows);
			if (start > 0) {
				lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
			}
			for (let i = start; i < end; i++) {
				const r = this.reasoners[i];
				const isSelected = i === this.selection.index;

				const item: ListItem = {
					label: r.name,
					metadata: r.active ? `${r.description}  active` : r.description,
					selected: isSelected,
					statusDot: r.active ? "active" : undefined,
				};

				lines.push(renderListItem(item, innerWidth));
			}
			if (end < this.reasoners.length) {
				lines.push(renderStatusLine(`↓ ${this.reasoners.length - end} more`, innerWidth));
			}
		}

		// ── Bottom bar ──
		lines.push(renderSeparator(popupWidth));
		const bottomText = this.message
			? this.message
			: "Select a reasoning mode for the next turn.";
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom rule ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.reasoners.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
