// ── ThemeSelectorOverlay ───────────────────────────────────────────────────────
// Overlay for selecting a color theme.

import { createListSelector, type ListSelectorOverlay } from "./popup-utils.ts";

export interface ThemeInfo {
	name: string;
	description: string;
	active: boolean;
}

export type ThemeSelectorAction =
	| { type: "select"; item: ThemeInfo }
	| { type: "close" };

export const ThemeSelectorOverlay = createListSelector<ThemeInfo>({
	title: "Theme",
	emptyText: "No themes available.",
	defaultMessage: "Select a color theme.",
	toItem: function (this: ListSelectorOverlay<ThemeInfo>, t, i, selectedIndex) {
		return {
			label: t.name,
			metadata: t.description,
			selected: i === selectedIndex,
			current: t.active,
		};
	},
});
