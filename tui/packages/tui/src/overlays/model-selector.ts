// ── ModelSelectorOverlay — beautiful model selection popup ─────────────────
// Rounded-corner overlay for selecting an active model from the configured list.
// Uses the shared popup-utils design system.

import { parsePopupListNav } from "./popup-utils.ts";
import { SelectorController } from "./selector-controller.ts";
import type { InkListOverlayModel } from "./ink-overlay-model.ts";

export interface ModelInfo {
	id: string;
	name: string;
	active: boolean;
	url?: string;
}

export type ModelSelectorAction =
	| { type: "select"; model: ModelInfo }
	| { type: "close" };

export class ModelSelectorOverlay {
	public visible = false;
	private models: ModelInfo[] = [];
	private selection = new SelectorController();
	private message = "";

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
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		return {
			kind: "list",
			title: "Model",
			subtitle: ` (${this.models.length})`,
			hints: "↑↓ select · enter confirm · esc close",
			items: this.models.map((model, index) => ({
				label: model.name,
				metadata: model.url ?? model.id,
				selected: index === this.selection.index,
				current: model.active,
			})),
			emptyText: "No models configured. Add models to settings.json.",
			footer: this.message || "Select a model for this session.",
			selectedIndex: this.selection.index,
		};
	}

	private moveSelection(delta: number): void {
		const n = this.models.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
