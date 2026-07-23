// ── SettingsOverlay — beautiful settings browser ────────────────────────────
// Rounded-corner overlay for browsing and modifying runtime settings.
// Two views: main menu (list of settings with current values) and detail view
// (show available options for a selected setting, with enter to apply).
// Uses the shared popup-utils design system.

import { type Component, clampLineToWidth, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import {
	BOX,
	renderListItem,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	type ListItem,
} from "./popup-utils.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";

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
	private selectedIndex = 0;
	/** When in detail view, the selected setting's option index. */
	private selectedOptionIndex = 0;
	/** `true` when showing the detail/option-selection view. */
	private inDetailView = false;
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

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
		const lines: string[] = [];

		const headerFg = theme.fg("header", "");

		// ── Top rounded corner ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		// ── Title row ──
		if (!this.inDetailView) {
			const titleText = "Runtime Settings";
			const subtitleText = ` (${this.settings.length})`;
			const hintsText = " ↑↓ select · enter detail · esc close";
			const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
			const titleVisible = visibleWidth(titleLine);
			const titlePad = Math.max(0, innerWidth - titleVisible);
			lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
		} else {
			const s = this.settings[this.selectedIndex];
			const subtitle = s ? ` (${s.options.length} options)` : "";
			const titleText = s?.name ?? "Settings";
			const hintsText = " ↑↓ select · enter apply · tab back";
			const titleLine = `${titleText}${theme.fg("muted", "")}${subtitle}${hintsText}`;
			const titleVisible = visibleWidth(titleLine);
			const titlePad = Math.max(0, innerWidth - titleVisible);
			lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
		}

		// ── Separator ──
		lines.push(renderSeparator(popupWidth, 1));

		// ── Content ──
		if (!this.inDetailView) {
			this.renderMainMenu(lines, innerWidth, popupWidth);
		} else {
			this.renderDetailView(lines, innerWidth, popupWidth);
		}

		// ── Bottom bar ──
		lines.push(renderSeparator(popupWidth, 1));
		const bottomText = this.message
			? this.message
			: this.inDetailView
				? "Select an option to apply."
				: "Press Enter to configure a setting.";
		lines.push(renderStatusLine(bottomText, innerWidth));

		// ── Bottom rounded corner ──
		lines.push(`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private renderMainMenu(lines: string[], innerWidth: number, popupWidth: number): void {
		if (!this.settings.length) {
			lines.push(renderStatusLine("No settings available.", innerWidth));
			return;
		}

		const maxRows = 12;
		const start = Math.max(
			0,
			Math.min(
				this.selectedIndex - Math.floor(maxRows / 2),
				Math.max(0, this.settings.length - maxRows),
			),
		);
		const end = Math.min(this.settings.length, start + maxRows);
		if (start > 0) {
			lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
		}
		for (let i = start; i < end; i++) {
			const s = this.settings[i];
			const isSelected = i === this.selectedIndex;

			// Build the item with a gear icon for settings
			const item: ListItem = {
				label: s.name,
				metadata: `(${s.currentValue})`,
				selected: isSelected,
				bullet: isSelected ? "▸" : " ",
			};

			lines.push(renderListItem(item, innerWidth));
		}
		if (end < this.settings.length) {
			lines.push(renderStatusLine(`↓ ${this.settings.length - end} more`, innerWidth));
		}
	}

	private renderDetailView(lines: string[], innerWidth: number, popupWidth: number): void {
		const s = this.settings[this.selectedIndex];
		if (!s) {
			lines.push(renderStatusLine("No setting selected.", innerWidth));
			return;
		}

		// ── Current value indicator ──
		const currentMark = s.options.find((o) => o.current);
		if (currentMark) {
			const currentColor =
				typeof currentMark.toggleOn === "boolean"
					? currentMark.toggleOn
						? theme.fg("success", "")
						: theme.fg("error", "")
					: theme.fg("active", "");
			const indicator = `${currentColor}● Current: ${currentMark.label}${RESET}`;
			lines.push(renderStatusLine(indicator, innerWidth, ""));
		}

		// ── Separator before options ──
		lines.push(renderSeparator(popupWidth, 1));

		const maxRows = 10;
		const start = Math.max(
			0,
			Math.min(
				this.selectedOptionIndex - Math.floor(maxRows / 2),
				Math.max(0, s.options.length - maxRows),
			),
		);
		const end = Math.min(s.options.length, start + maxRows);
		if (start > 0) {
			lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
		}
		for (let i = start; i < end; i++) {
			const opt = s.options[i];
			const isSelected = i === this.selectedOptionIndex;

			// Build the item
			const item: ListItem = {
				label: opt.label,
				selected: isSelected,
				bullet: isSelected ? "▸" : " ",
			};

			// Current dot
			if (opt.current) {
				item.statusDot = "active";
			}

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
			lines.push(renderStatusLine(`↓ ${s.options.length - end} more`, innerWidth));
		}
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
