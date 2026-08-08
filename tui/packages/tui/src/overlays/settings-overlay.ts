// ── SettingsOverlay — beautiful settings browser ────────────────────────────
// Rounded-corner overlay for browsing and modifying runtime settings.
// Two views: main menu (list of settings with current values) and detail view
// (show available options for a selected setting, with enter to apply).
// Uses the shared popup-utils design system.

import { type Component, RESET } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import {
	clampPopupLines,
	type ListItem,
	renderListItem,
	renderListPopupFrame,
	renderSeparator,
	renderStatusLine,
} from "./popup-utils.ts";

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

export class SettingsSelectorOverlay implements Component {
	public visible = false;
	private settings: SettingDef[] = [];
	/** Index into `settings` array (main menu view). */
	private _selectedIndex = 0;
	/** When in detail view, the selected setting's option index. */
	private _selectedOptionIndex = 0;
	/** `true` when showing the detail/option-selection view. */
	private _inDetailView = false;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	/** @internal Exposed for tests. */
	get selectedIndex(): number { return this._selectedIndex; }
	get selectedOptionIndex(): number { return this._selectedOptionIndex; }
	get inDetailView(): boolean { return this._inDetailView; }

	setSettings(settings: SettingDef[]): void {
		this.settings = settings;
		if (this._selectedIndex >= this.settings.length) {
			this._selectedIndex = Math.max(0, this.settings.length - 1);
		}
		this.invalidate();
	}

	setMessage(message: string): void {
		this.message = message;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this._inDetailView = false;
		this._selectedIndex = 0;
		this._selectedOptionIndex = 0;
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

		if (this._inDetailView) {
			return this.handleDetailInput(data);
		}

		return this.handleMenuInput(data);
	}

	private handleMenuInput(data: string): SettingsSelectorAction | null {
		if (data === "\r" || data === "\n") {
			// Enter opens detail view for the selected setting
			const s = this.settings[this._selectedIndex];
			if (s?.name.toLowerCase() === "model") {
				return { type: "open", settingName: s.name };
			}
			this._inDetailView = true;
			this._selectedOptionIndex = s
				? s.options.findIndex(o => o.current) >= 0
					? s.options.findIndex(o => o.current)
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
		const s = this.settings[this._selectedIndex];
		if (!s) return { type: "close" };

		// Tab or backspace goes back to menu
		if (data === "\t" || data === "\x08") {
			this._inDetailView = false;
			this.invalidate();
			return null;
		}

		if (data === "\r" || data === "\n") {
			// Apply the selected option
			const opt = s.options[this._selectedOptionIndex];
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
		this.cachedLines = null;
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}
		this.cachedWidth = width;

		if (!this.visible) return [];

		const popupWidth = Math.max(1, width);
		const innerWidth = Math.max(1, popupWidth - 4);
		const bodyLines: string[] = [];

		// ── Content ──
		if (!this._inDetailView) {
			this.renderMainMenu(bodyLines, innerWidth, popupWidth);
		} else {
			this.renderDetailView(bodyLines, innerWidth, popupWidth);
		}

		const setting = this.settings[this._selectedIndex];
		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: this._inDetailView
				? (setting?.name ?? "Settings")
				: "Runtime Settings",
			subtitle: this._inDetailView
				? ` (${setting?.options.length ?? 0} options)`
				: ` (${this.settings.length})`,
			hints: this._inDetailView
				? "↑↓ navigate · enter apply · tab back · esc close"
				: "↑↓ navigate · enter configure · esc close",
			bodyLines,
			bottomText:
				this.message ||
				(this._inDetailView
					? "Select an option to apply."
					: "Select a setting to configure."),
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private renderMainMenu(
		lines: string[],
		innerWidth: number,
		_popupWidth: number,
	): void {
		if (!this.settings.length) {
			lines.push(renderStatusLine("No settings available.", innerWidth));
			return;
		}

		const maxRows = 12;
		const start = Math.max(
			0,
			Math.min(
				this._selectedIndex - Math.floor(maxRows / 2),
				Math.max(0, this.settings.length - maxRows),
			),
		);
		const end = Math.min(this.settings.length, start + maxRows);
		if (start > 0) {
			lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
		}
		for (let i = start; i < end; i++) {
			const s = this.settings[i];
			const isSelected = i === this._selectedIndex;

			// Build the item with a gear icon for settings
			const item: ListItem = {
				label: s.name,
				metadata: `(${s.currentValue})`,
				selected: isSelected,
			};

			lines.push(renderListItem(item, innerWidth));
		}
		if (end < this.settings.length) {
			lines.push(
				renderStatusLine(`↓ ${this.settings.length - end} more`, innerWidth),
			);
		}
	}

	private renderDetailView(
		lines: string[],
		innerWidth: number,
		popupWidth: number,
	): void {
		const s = this.settings[this._selectedIndex];
		if (!s) {
			lines.push(renderStatusLine("No setting selected.", innerWidth));
			return;
		}

		// ── Current value indicator ──
		const currentMark = s.options.find(o => o.current);
		if (currentMark) {
			const currentColor =
				typeof currentMark.toggleOn === "boolean"
					? currentMark.toggleOn
						? theme.fg("success", "")
						: theme.fg("error", "")
					: theme.fg("active", "");
			const indicator = `${currentColor}Current: ${currentMark.label} ✓${RESET}`;
			lines.push(renderStatusLine(indicator, innerWidth, ""));
		}

		// ── Separator before options ──
		lines.push(renderSeparator(popupWidth));

		const maxRows = 10;
		const start = Math.max(
			0,
			Math.min(
				this._selectedOptionIndex - Math.floor(maxRows / 2),
				Math.max(0, s.options.length - maxRows),
			),
		);
		const end = Math.min(s.options.length, start + maxRows);
		if (start > 0) {
			lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
		}
		for (let i = start; i < end; i++) {
			const opt = s.options[i];
			const isSelected = i === this._selectedOptionIndex;

			// Build the item
			const item: ListItem = {
				label: opt.label,
				selected: isSelected,
				current: opt.current,
			};

			// Toggle mark
			if (typeof opt.toggleOn === "boolean") {
				const mark = opt.toggleOn
					? `${theme.fg("success", "")}[on]${RESET}`
					: `${theme.fg("error", "")}[off]${RESET}`;
				item.metadata = mark;
			}

			lines.push(renderListItem(item, innerWidth));
		}
		if (end < s.options.length) {
			lines.push(
				renderStatusLine(`↓ ${s.options.length - end} more`, innerWidth),
			);
		}
	}

	private moveSelection(delta: number): void {
		const n = this.settings.length;
		if (!n) return;
		this._selectedIndex = (this._selectedIndex + delta + n) % n;
		this.invalidate();
	}

	private moveOptionSelection(delta: number): void {
		const s = this.settings[this._selectedIndex];
		if (!s) return;
		const n = s.options.length;
		if (!n) return;
		this._selectedOptionIndex = (this._selectedOptionIndex + delta + n) % n;
		this.invalidate();
	}
}
