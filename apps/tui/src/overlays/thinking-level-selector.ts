// ── ThinkingLevelSelectorOverlay ─────────────────────────────────────────────
// Overlay for selecting a thinking (reasoning) level.

import { createListSelector, type ListSelectorOverlay } from "./popup-utils.ts";

export interface ThinkingLevelInfo {
	id: string;
	label: string;
	description: string;
	active: boolean;
}

export type ThinkingLevelSelectorAction =
	| { type: "select"; item: ThinkingLevelInfo }
	| { type: "close" };

export const ThinkingLevelSelectorOverlay =
	createListSelector<ThinkingLevelInfo>({
		title: "Thinking Level",
		emptyText: "No thinking levels available.",
		defaultMessage: "Select a thinking level for the next turn.",
		toItem: function (
			this: ListSelectorOverlay<ThinkingLevelInfo>,
			l,
			i,
			selectedIndex,
		) {
			return {
				label: l.label,
				metadata: l.description,
				selected: i === selectedIndex,
				statusDot: l.active ? "active" : undefined,
			};
		},
	});
