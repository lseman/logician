// ── ThemeSelectorOverlay ───────────────────────────────────────────────────────
// Overlay for selecting a color theme.
// Pattern: list, select, confirm, close.
// Mirrors ReasonerSelectorOverlay / PluginManagerOverlay.

import { type Component } from "../layers/core/tui-core.ts";
import { SelectorController } from "./selector-controller.ts";
import {
	renderListItem,
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
	parsePopupListNav,
	renderListPopupFrame,
	renderListPopupBody,
	type ListItem,
} from "./popup-utils.ts";

export interface ThemeInfo {
	name: string;
	description: string;
}

export type ThemeSelectorAction =
	| { type: "select"; theme: ThemeInfo }
	| { type: "close" };

export class ThemeSelectorOverlay implements Component {
	public visible = false;
	private themes: ThemeInfo[] = [];
	private selection = new SelectorController();
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setThemes(themes: ThemeInfo[]): void {
		this.themes = themes;
		this.selection.set(this.selection.index, this.themes.length);
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

	handleInput(data: string): ThemeSelectorAction | null {
		if (!this.visible) return null;

		const nav = parsePopupListNav(data);
		if (nav?.type === "close") return { type: "close" };
		if (nav?.type === "confirm") {
			const themeInfo = this.themes[this.selection.index];
			return themeInfo
				? { type: "select", theme: themeInfo }
				: { type: "close" };
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
			this.themes,
			this.selection,
			innerWidth,
			10,
			(t, i) => {
				const item: ListItem = {
					label: t.name,
					metadata: t.description,
					selected: i === this.selection.index,
				};
				return renderListItem(item, innerWidth);
			},
			"No themes available.",
		);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: "Theme",
			subtitle: ` (${this.themes.length})`,
			hints: " ↑↓ select · enter confirm · esc close",
			bodyLines,
			bottomText: this.message || "Select a color theme.",
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.themes.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
