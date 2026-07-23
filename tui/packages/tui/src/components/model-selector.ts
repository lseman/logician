// ── ModelSelectorOverlay — beautiful model selection popup ─────────────────
// Rounded-corner overlay for selecting an active model from the configured list.
// Uses the shared popup-utils design system.

import {
	type Component,
	clampLineToWidth,
	visibleWidth,
} from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import { SelectorController } from "./selector-controller.ts";
import {
	BOX,
	renderListItem,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	type ListItem,
} from "./popup-utils.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";

export interface ModelInfo {
	id: string;
	name: string;
	active: boolean;
	url?: string;
}

export type ModelSelectorAction =
	| { type: "select"; model: ModelInfo }
	| { type: "close" };

export class ModelSelectorOverlay implements Component {
	public visible = false;
	private models: ModelInfo[] = [];
	private selection = new SelectorController();
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setModels(models: ModelInfo[]): void {
		this.models = models;
		const activeIndex = this.models.findIndex((model) => model.active);
		this.selection.set(activeIndex >= 0
			? activeIndex
			: this.selection.index, this.models.length);
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

	handleInput(data: string): ModelSelectorAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			const model = this.models[this.selection.index];
			return model ? { type: "select", model } : { type: "close" };
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
		const innerWidth = Math.max(1, popupWidth - 4);
		const lines: string[] = [];

		// ── Top rounded corner ──
		const headerFg = theme.fg("header", "");
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		// ── Title row ──
		const titleText = "Model";
		const subtitleText = ` (${this.models.length})`;
		const hintsText = " ↑↓ select · enter confirm · esc close";
		const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

		// ── Separator ──
		lines.push(renderSeparator(popupWidth, 1));

		// ── Model list ──
		if (!this.models.length) {
			lines.push(
				renderStatusLine(
					"No models configured. Add \"models\" array to settings.json.",
					innerWidth,
					theme.fg("warning", ""),
				),
			);
		} else {
			const maxRows = 10;
			const { start, end } = this.selection.window(this.models.length, maxRows);
			if (start > 0) {
				lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
			}
			for (let i = start; i < end; i++) {
				const m = this.models[i];
				const isSelected = i === this.selection.index;

				// Build the item
				const item: ListItem = {
					label: m.name,
					metadata: m.url ?? m.id,
					selected: isSelected,
					statusDot: m.active ? "active" : undefined,
				};

				lines.push(renderListItem(item, innerWidth));
			}
			if (end < this.models.length) {
				lines.push(renderStatusLine(`↓ ${this.models.length - end} more`, innerWidth));
			}
		}

		// ── Bottom bar ──
		lines.push(renderSeparator(popupWidth, 1));
		const bottomText = this.message
			? this.message
			: "Select a model for this session.";
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom rounded corner ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.models.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
