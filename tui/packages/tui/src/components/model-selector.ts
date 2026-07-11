// ── ModelSelectorOverlay — beautiful model selection popup ─────────────────
// Rounded-corner overlay for selecting an active model from the configured list.
// Uses the shared popup-utils design system.

import {
	type Component,
	clampLineToWidth,
	visibleWidth,
} from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
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
	private selectedIndex = 0;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setModels(models: ModelInfo[]): void {
		this.models = models;
		if (this.selectedIndex >= this.models.length) {
			this.selectedIndex = Math.max(0, this.models.length - 1);
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

	handleInput(data: string): ModelSelectorAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}
		if (data === "\r" || data === "\n") {
			const model = this.models[this.selectedIndex];
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

		const popupWidth = Math.max(48, Math.min(width, 120));
		const innerWidth = popupWidth - 4;
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
			const start = Math.max(
				0,
				Math.min(
					this.selectedIndex - Math.floor(maxRows / 2),
					Math.max(0, this.models.length - maxRows),
				),
			);
			const end = Math.min(this.models.length, start + maxRows);
			if (start > 0) {
				lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
			}
			for (let i = start; i < end; i++) {
				const m = this.models[i];
				const isSelected = i === this.selectedIndex;

				// Build the item
				const item: ListItem = {
					label: m.name,
					metadata: m.url ?? m.id,
					selected: isSelected,
					statusDot: m.active ? "active" : undefined,
				};

				// If active, add "● active" badge to the left
				if (m.active) {
					const activeDot = theme.fg("active", "");
					item.label = `${item.label}  ${activeDot}● active${RESET}`;
				}

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
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}
}
