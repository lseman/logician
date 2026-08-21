// ── ThinkingLevelSelectorOverlay ─────────────────────────────────────────────
// Overlay for selecting a thinking (reasoning) level.

import { type ListItem, ListSelectorOverlay } from "./popup-utils.ts";

export interface ThinkingLevelInfo {
	id: string;
	label: string;
	description: string;
	active: boolean;
}

export type ThinkingLevelSelectorAction =
	| { type: "select"; level: ThinkingLevelInfo }
	| { type: "close" };

export class ThinkingLevelSelectorOverlay extends ListSelectorOverlay<ThinkingLevelInfo> {
	constructor() {
		super({
			title: "Thinking Level",
			emptyText: "No thinking levels available.",
			defaultMessage: "Select a thinking level for the next turn.",
			toItem: (l, i, selectedIndex): ListItem => ({
				label: l.label,
				metadata: l.description,
				selected: i === selectedIndex,
				statusDot: l.active ? "active" : undefined,
			}),
		});
	}

	setLevels(levels: ThinkingLevelInfo[]): void {
		const activeIndex = levels.findIndex(l => l.active);
		this.setItems(
			levels,
			activeIndex >= 0 ? activeIndex : this.selection.index,
		);
	}

	handleInput(data: string): ThinkingLevelSelectorAction | null {
		const action = this.handleListInput(data);
		if (!action) return null;
		return action.type === "select"
			? { type: "select", level: action.item }
			: action;
	}
}
