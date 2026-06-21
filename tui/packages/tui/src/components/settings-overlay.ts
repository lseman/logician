// ── SettingsOverlay ──────────────────────────────────────────────────────────
// Overlay for browsing and modifying runtime settings.
// Two views: main menu (list of settings with current values) and detail view
// (show available options for a selected setting, with enter to apply).
// Mirrors ReasonerSelectorOverlay / ThemeSelectorOverlay pattern.

import { type Component, clampLineToWidth, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const getHeader = (): string => theme.fg("header", "");
const getSelected = (): string => theme.fg("selected", "");
const getMuted = (): string => theme.fg("muted", "");
const getActive = (): string => theme.fg("active", "");
const getGreen = (): string => theme.fg("success", "");
const getRed = (): string => theme.fg("error", "");

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
			this.inDetailView = true;
			const s = this.settings[this.selectedIndex];
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

		const overlayWidth = Math.max(56, Math.min(width, 120));
		const innerWidth = Math.max(1, overlayWidth - 4);
		const lines: string[] = [];

		lines.push(`${getHeader()}┌${"─".repeat(overlayWidth - 2)}┐${RESET}`);

		if (!this.inDetailView) {
			const title = `${BOLD}Runtime Settings${RESET}${DIM} (${this.settings.length})${RESET}`;
			lines.push(
				boxLine(title, "↑↓ select · enter detail · esc close", innerWidth),
			);
		} else {
			const s = this.settings[this.selectedIndex];
			const subtitle = s ? `${s.options.length} options` : "";
			const title = `${BOLD}${s?.name ?? "Settings"}${RESET}${DIM}${subtitle ? ` (${subtitle})` : ""}${RESET}`;
			lines.push(
				boxLine(title, "↑↓ select · enter apply · tab back", innerWidth),
			);
		}

		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);

		if (!this.inDetailView) {
			this.renderMainMenu(lines, innerWidth);
		} else {
			this.renderDetailView(lines, innerWidth);
		}

		lines.push(`${getHeader()}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);
		lines.push(
			boxLine(
				this.message
					? `${DIM}${this.message}${RESET}`
					: this.inDetailView
						? `${getMuted()}Select an option to apply.${RESET}`
						: `${getMuted()}Press Enter to configure a setting.${RESET}`,
				"",
				innerWidth,
			),
		);
		lines.push(`${getHeader()}└${"─".repeat(overlayWidth - 2)}┘${RESET}`);

		this.cachedLines = lines.map((line) => clampLineToWidth(line, width));
		return this.cachedLines;
	}

	private renderMainMenu(lines: string[], innerWidth: number): void {
		if (!this.settings.length) {
			lines.push(
				boxLine(`${getMuted()}No settings available.${RESET}`, "", innerWidth),
			);
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
			lines.push(
				boxLine(`${getMuted()}↑ ${start} more${RESET}`, "", innerWidth),
			);
		}
		for (let i = start; i < end; i++) {
			const s = this.settings[i];
			const selected = i === this.selectedIndex;
			const cursor = selected ? `${getSelected()}▸${RESET}` : " ";
			const currentMark = `${getMuted()}(${s.currentValue})${RESET}`;
			const name = selected
				? `${getSelected()}${BOLD}${s.name}${RESET}`
				: s.name;
			lines.push(boxLine(`${cursor} ${name}`, currentMark, innerWidth));
		}
		if (end < this.settings.length) {
			lines.push(
				boxLine(
					`${getMuted()}↓ ${this.settings.length - end} more${RESET}`,
					"",
					innerWidth,
				),
			);
		}
	}

	private renderDetailView(lines: string[], innerWidth: number): void {
		const s = this.settings[this.selectedIndex];
		if (!s) {
			lines.push(
				boxLine(`${getMuted()}No setting selected.${RESET}`, "", innerWidth),
			);
			return;
		}

		const currentMark = s.options.find((o) => o.current);
		if (currentMark) {
			const markColor =
				typeof currentMark.toggleOn === "boolean"
					? currentMark.toggleOn
						? getGreen()
						: getRed()
					: getActive();
			lines.push(
				boxLine(
					`${markColor}Current: ${currentMark.label}${RESET}`,
					"",
					innerWidth,
				),
			);
		}

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
			lines.push(
				boxLine(`${getMuted()}↑ ${start} more${RESET}`, "", innerWidth),
			);
		}
		for (let i = start; i < end; i++) {
			const opt = s.options[i];
			const selected = i === this.selectedOptionIndex;
			const cursor = selected ? `${getSelected()}▸${RESET}` : " ";
			const currentDot = opt.current ? `${getActive()}●${RESET}` : "";
			const toggleMark =
				typeof opt.toggleOn === "boolean"
					? opt.toggleOn
						? `${getGreen()}[on]${RESET}`
						: `${getRed()}[off]${RESET}`
					: "";
			const rightSide = [currentDot, toggleMark].filter(Boolean).join(" ");
			const value = selected
				? `${getSelected()}${BOLD}${opt.label}${RESET}`
				: opt.label;
			lines.push(boxLine(`${cursor} ${value}`, rightSide, innerWidth));
		}
		if (end < s.options.length) {
			lines.push(
				boxLine(
					`${getMuted()}↓ ${s.options.length - end} more${RESET}`,
					"",
					innerWidth,
				),
			);
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

function boxLine(left: string, right: string, width: number): string {
	const leftWidth = visibleWidth(left);
	const rightWidth = visibleWidth(right);
	const gap = Math.max(1, width - leftWidth - rightWidth);
	const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
	const pad = Math.max(0, width - visibleWidth(content));
	return `${getHeader()}│${RESET} ${content}${" ".repeat(pad)} ${getHeader()}│${RESET}`;
}
