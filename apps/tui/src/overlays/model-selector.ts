// ── ModelSelectorOverlay — beautiful model selection popup ─────────────────
// Uses the shared ListSelectorOverlay factory (popup-utils) so subclasses only
// need to declare the item shape and ListItem mapping.

import { createListSelector, type ListSelectorOverlay } from "./popup-utils.ts";

export interface ModelInfo {
	id: string;
	name: string;
	active: boolean;
	url?: string;
}

export type ModelSelectorAction =
	| { type: "select"; item: ModelInfo }
	| { type: "close" };

export const ModelSelectorOverlay = createListSelector<ModelInfo>({
	title: "Model",
	emptyText: 'No models configured. Add "models" array to settings.json.',
	defaultMessage: "Select a model for this session.",
	toItem: function (this: ListSelectorOverlay<ModelInfo>, m, i, selectedIndex) {
		return {
			label: m.name,
			metadata: m.url ?? m.id,
			selected: i === selectedIndex,
			current: m.active,
		};
	},
});
