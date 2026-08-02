// ── ThemeSelectorOverlay ───────────────────────────────────────────────────────
// Overlay for selecting a color theme.
// Pattern: list, select, confirm, close.
// Mirrors ReasonerSelectorOverlay / PluginManagerOverlay.

import { SelectorController } from "./selector-controller.ts";
import type { InkListOverlayModel } from "./ink-overlay-model.ts";
import { parsePopupListNav } from "./popup-utils.ts";

export interface ThemeInfo {
	name: string;
	description: string;
}

export type ThemeSelectorAction =
	| { type: "select"; theme: ThemeInfo }
	| { type: "close" };

export class ThemeSelectorOverlay {
	public visible = false;
	private themes: ThemeInfo[] = [];
	private selection = new SelectorController();
	private message = "";

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
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		return {
			kind: "list",
			title: "Theme",
			subtitle: ` (${this.themes.length})`,
			hints: "↑↓ select · enter confirm · esc close",
			items: this.themes.map((themeInfo, index) => ({
				label: themeInfo.name,
				metadata: themeInfo.description,
				selected: index === this.selection.index,
			})),
			emptyText: "No themes available.",
			footer: this.message || "Select a color theme.",
			selectedIndex: this.selection.index,
		};
	}

	private moveSelection(delta: number): void {
		const n = this.themes.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
