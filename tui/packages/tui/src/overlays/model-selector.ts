// ── ModelSelectorOverlay — beautiful model selection popup ─────────────────
// Rounded-corner overlay for selecting an active model from the configured list.
// Uses the shared popup-utils design system.

import type { Component } from "../terminal/core.ts";
import {
	clampPopupLines,
	type ListItem,
	POPUP_FRAME_OVERHEAD,
	parsePopupListNav,
	renderListItem,
	renderListPopupBody,
	renderListPopupFrame,
} from "./popup-utils.ts";
import { SelectorController } from "./selector-controller.ts";

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
		this.selection.set(
			activeIndex >= 0 ? activeIndex : this.selection.index,
			this.models.length,
		);
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

		const nav = parsePopupListNav(data);
		if (nav?.type === "close") return { type: "close" };
		if (nav?.type === "confirm") {
			const model = this.models[this.selection.index];
			return model ? { type: "select", model } : { type: "close" };
		}
		if (nav?.type === "move") {
			this.moveSelection(nav.delta);
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

		const bodyLines = renderListPopupBody(
			this.models,
			this.selection,
			innerWidth,
			10,
			(m, i) => {
				const item: ListItem = {
					label: m.name,
					metadata: m.url ?? m.id,
					selected: i === this.selection.index,
					current: m.active,
				};
				return renderListItem(item, innerWidth);
			},
			"No models configured. Add \"models\" array to settings.json.",
		);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: "Model",
			subtitle: ` (${this.models.length})`,
			hints: " ↑↓ select · enter confirm · esc close",
			bodyLines,
			bottomText: this.message || "Select a model for this session.",
		});

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
