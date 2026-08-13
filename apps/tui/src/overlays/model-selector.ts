// ── ModelSelectorOverlay — beautiful model selection popup ─────────────────
// Rounded-corner overlay for selecting an active model from the configured list.

import { type ListItem, ListSelectorOverlay } from "./popup-utils.ts";

export interface ModelInfo {
	id: string;
	name: string;
	active: boolean;
	url?: string;
}

export type ModelSelectorAction =
	| { type: "select"; model: ModelInfo }
	| { type: "close" };

export class ModelSelectorOverlay extends ListSelectorOverlay<ModelInfo> {
	constructor() {
		super({
			title: "Model",
			emptyText: 'No models configured. Add "models" array to settings.json.',
			defaultMessage: "Select a model for this session.",
			toItem: (m, i, selectedIndex): ListItem => ({
				label: m.name,
				metadata: m.url ?? m.id,
				selected: i === selectedIndex,
				current: m.active,
			}),
		});
	}

	setModels(models: ModelInfo[]): void {
		const activeIndex = models.findIndex(model => model.active);
		this.setItems(
			models,
			activeIndex >= 0 ? activeIndex : this.selection.index,
		);
	}

	handleInput(data: string): ModelSelectorAction | null {
		const action = this.handleListInput(data);
		if (!action) return null;
		return action.type === "select"
			? { type: "select", model: action.item }
			: action;
	}
}
