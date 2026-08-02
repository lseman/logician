// ── SettingsOverlay — beautiful settings browser ────────────────────────────
// Rounded-corner overlay for browsing and modifying runtime settings.
// Two views: main menu (list of settings with current values) and detail view
// (show available options for a selected setting, with enter to apply).
// Uses the shared popup-utils design system.

import type { InkListOverlayModel } from "./ink-overlay-model.ts";

// ── Data types ──────────────────────────────────────────────────────────────

export interface SettingOption {
	/** Label shown to the user. */
	label: string;
	/** Value sent back on selection (the value to apply). */
	value: string;
	/** Whether this option is currently active. */
	current?: boolean;
	/** For boolean toggles: show [on]/[off] indicator. true=on. */
	toggleOn?: boolean;
}

export interface SettingDef {
	/** Display name shown in the menu. */
	name: string;
	/** Current value. */
	currentValue: string;
	/** Short description of what this setting controls. */
	description: string;
	/** Available options. */
	options: SettingOption[];
}

export type SettingsSelectorAction =
	| { type: "change"; settingName: string; value: string }
	| { type: "open"; settingName: string }
	| { type: "close" };

export class SettingsSelectorOverlay {
	public visible = false;
	private settings: SettingDef[] = [];
	/** Index into `settings` array (main menu view). */
	private selectedIndex = 0;
	/** When in detail view, the selected setting's option index. */
	private selectedOptionIndex = 0;
	/** `true` when showing the detail/option-selection view. */
	private inDetailView = false;
	private message = "";

	setSettings(settings: SettingDef[]): void {
		this.settings = settings;
		if (this.selectedIndex >= this.settings.length) {
			this.selectedIndex = Math.max(0, this.settings.length - 1);
		}
		this.invalidate();
	}

	setMessage(message: string): void {
		this.message = message;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.inDetailView = false;
		this.selectedIndex = 0;
		this.selectedOptionIndex = 0;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): SettingsSelectorAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
			return { type: "close" };
		}

		if (this.inDetailView) {
			return this.handleDetailInput(data);
		}

		return this.handleMenuInput(data);
	}

	private handleMenuInput(data: string): SettingsSelectorAction | null {
		if (data === "\r" || data === "\n") {
			// Enter opens detail view for the selected setting
			const s = this.settings[this.selectedIndex];
			if (s?.name.toLowerCase() === "model") {
				return { type: "open", settingName: s.name };
			}
			this.inDetailView = true;
			this.selectedOptionIndex = s
				? s.options.findIndex((o) => o.current) >= 0
					? s.options.findIndex((o) => o.current)
					: 0
				: 0;
			this.invalidate();
			return null;
		}
		if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
			this.moveSelection(-1);
			return null;
		}
		if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
			this.moveSelection(1);
			return null;
		}
		if (data === "\x1b[5~") {
			this.moveSelection(-8);
			return null;
		}
		if (data === "\x1b[6~") {
			this.moveSelection(8);
			return null;
		}
		return null;
	}

	private handleDetailInput(data: string): SettingsSelectorAction | null {
		const s = this.settings[this.selectedIndex];
		if (!s) return { type: "close" };

		// Tab or backspace goes back to menu
		if (data === "\t" || data === "\x08") {
			this.inDetailView = false;
			this.invalidate();
			return null;
		}

		if (data === "\r" || data === "\n") {
			// Apply the selected option
			const opt = s.options[this.selectedOptionIndex];
			if (opt) {
				return { type: "change", settingName: s.name, value: opt.value };
			}
			return { type: "close" };
		}

		if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
			this.moveOptionSelection(1);
			return null;
		}
		if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
			this.moveOptionSelection(-1);
			return null;
		}
		if (data === "\x1b[5~") {
			this.moveOptionSelection(-8);
			return null;
		}
		if (data === "\x1b[6~") {
			this.moveOptionSelection(8);
			return null;
		}
		return null;
	}

	invalidate(): void {
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		const setting = this.settings[this.selectedIndex];
		if (this.inDetailView) {
			return {
				kind: "list",
				title: setting?.name ?? "Settings",
				subtitle: ` (${setting?.options.length ?? 0} options)`,
				hints: "↑↓ navigate · enter apply · tab back · esc close",
				items: (setting?.options ?? []).map((option, index) => ({
					label: option.label,
					metadata: typeof option.toggleOn === "boolean"
						? option.toggleOn ? "on" : "off"
						: undefined,
					selected: index === this.selectedOptionIndex,
					current: option.current,
				})),
				emptyText: "No options available.",
				footer: this.message || "Select an option to apply.",
				selectedIndex: this.selectedOptionIndex,
			};
		}
		return {
			kind: "list",
			title: "Runtime Settings",
			subtitle: ` (${this.settings.length})`,
			hints: "↑↓ navigate · enter configure · esc close",
			items: this.settings.map((item, index) => ({
				label: item.name,
				metadata: `(${item.currentValue})`,
				selected: index === this.selectedIndex,
			})),
			emptyText: "No settings available.",
			footer: this.message || "Select a setting to configure.",
			selectedIndex: this.selectedIndex,
			maxRows: 12,
		};
	}


	private moveSelection(delta: number): void {
		const n = this.settings.length;
		if (!n) return;
		this.selectedIndex = (this.selectedIndex + delta + n) % n;
		this.invalidate();
	}

	private moveOptionSelection(delta: number): void {
		const s = this.settings[this.selectedIndex];
		if (!s) return;
		const n = s.options.length;
		if (!n) return;
		this.selectedOptionIndex = (this.selectedOptionIndex + delta + n) % n;
		this.invalidate();
	}
}
