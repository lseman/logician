// ── ThemeSelectorOverlay ───────────────────────────────────────────────────────
// Overlay for selecting a color theme.

import { type ListItem, ListSelectorOverlay } from "./popup-utils.ts";

export interface ThemeInfo {
	name: string;
	description: string;
	active: boolean;
}

export type ThemeSelectorAction =
	| { type: "select"; theme: ThemeInfo }
	| { type: "close" };

export class ThemeSelectorOverlay extends ListSelectorOverlay<ThemeInfo> {
	constructor() {
		super({
			title: "Theme",
			emptyText: "No themes available.",
			defaultMessage: "Select a color theme.",
			toItem: (t, i, selectedIndex): ListItem => ({
				label: t.name,
				metadata: t.description,
				selected: i === selectedIndex,
				current: t.active,
			}),
		});
	}

	setThemes(themes: ThemeInfo[]): void {
		const activeIndex = themes.findIndex(theme => theme.active);
		this.setItems(
			themes,
			activeIndex >= 0 ? activeIndex : this.selection.index,
		);
	}

	handleInput(data: string): ThemeSelectorAction | null {
		const action = this.handleListInput(data);
		if (!action) return null;
		return action.type === "select"
			? { type: "select", theme: action.item }
			: action;
	}
}
