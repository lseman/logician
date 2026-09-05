// ── SettingsSelectorOverlay — full-screen tabbed settings browser ────────────
// OMP-style fullscreen overlay with tab bar, search banner, split list,
// dividers, and footer hint. Compatible with the existing SettingDef interface.

import { clampLineToWidth } from "../terminal/core.ts";
import {
	BOLD,
	type Component,
	RESET,
	visibleWidth,
} from "../terminal/primitives.ts";
import { theme } from "../terminal/theme.ts";

// ── Data types ──────────────────────────────────────────────────────────────

export interface SettingOption {
	label: string;
	value: string;
	current?: boolean;
	toggleOn?: boolean;
}

export interface SettingDef {
	name: string;
	/** Display name (may differ from internal name). */
	label?: string;
	currentValue: string;
	/** Human-readable description shown in the settings list. */
	description?: string;
	/** Warning text shown when the setting has a problematic value. */
	warning?: string;
	/** Tab grouping for the settings overlay UI. */
	tab?: string;
	/** Section grouping within a tab. */
	section?: string;
	/** Display type for rendering: "text", "select", "toggle", "number", "credential". */
	displayType?: "text" | "select" | "toggle" | "number" | "credential";
	/** Available options for select/toggle display types. */
	options?: SettingOption[];
	/** When true, redact value in UI (for passwords/API keys). */
	redact?: boolean;
}

export type SettingsSelectorAction =
	| { type: "change"; settingName: string; value: string }
	| { type: "open"; settingName: string }
	| { type: "close" };

// ── Theme helpers ────────────────────────────────────────────────────────────

const getHeader = (): string => theme.fgRaw("header");
const getMuted = (): string => theme.fgRaw("muted");
const getSuccess = (): string => theme.fgRaw("success");
const getWarning = (): string => theme.fgRaw("warning");

// ── Layout helpers ───────────────────────────────────────────────────────────

const BOX = {
	topLeft: "┌",
	topRight: "┐",
	bottomLeft: "└",
	bottomRight: "┘",
	horizontal: "─",
	teeRight: "├",
	teeLeft: "┤",
};

// ── Tab management ───────────────────────────────────────────────────────────

interface SettingTab {
	name: string;
}

function deriveTabs(settings: SettingDef[]): SettingTab[] {
	const tabs: SettingTab[] = [];
	const tabNames: string[] = [];
	for (const s of settings) {
		const tabName = s.tab ?? "General";
		if (!tabNames.includes(tabName)) {
			tabNames.push(tabName);
			tabs.push({ name: tabName });
		}
	}
	return tabs;
}

function filterSettingsForTab(
	settings: SettingDef[],
	tabName: string,
): SettingDef[] {
	const items = settings.filter(s => (s.tab ?? "General") === tabName);
	const sections = [...new Set(items.map(s => s.section ?? tabName))];
	return sections.flatMap(section => items.filter(s => (s.section ?? tabName) === section));
}

// ── Fuzzy filter ─────────────────────────────────────────────────────────────

function fuzzyMatch(query: string, text: string): boolean {
	const q = query.toLowerCase();
	const t = text.toLowerCase();
	let qi = 0;
	for (let ti = 0; ti < t.length && qi < q.length; ti++) {
		if (t[ti] === q[qi]) qi++;
	}
	return qi === q.length;
}

function filterSettings(settings: SettingDef[], query: string): SettingDef[] {
	if (!query.trim()) return settings;
	const q = query.toLowerCase();
	return settings.filter((s: SettingDef) => {
		if (fuzzyMatch(q, s.name)) return true;
		if (fuzzyMatch(q, s.currentValue)) return true;
		if (s.description && fuzzyMatch(q, s.description)) return true;
		if (s.tab && fuzzyMatch(q, s.tab)) return true;
		if (s.section && fuzzyMatch(q, s.section)) return true;
		return false;
	});
}

// ── Border helpers ───────────────────────────────────────────────────────────

function topBorder(width: number, title: string): string {
	const inner = Math.max(0, width - 2);
	const label = clampLineToWidth(title, Math.max(0, inner - 1));
	return `${getHeader()}${BOX.topLeft}${BOX.horizontal}${BOLD}${label}${RESET}${getHeader()}${BOX.horizontal.repeat(Math.max(0, inner - visibleWidth(label) - 1))}${BOX.topRight}${RESET}`;
}

function bottomBorder(width: number): string {
	const inner = Math.max(0, width - 2);
	const hc = getHeader();
	return `${hc}${BOX.bottomLeft}${BOX.horizontal.repeat(inner)}${BOX.bottomRight}`;
}

function divider(width: number): string {
	const inner = Math.max(0, width - 2);
	const hc = getMuted();
	return `${hc}${BOX.teeRight}${BOX.horizontal.repeat(inner)}${BOX.teeLeft}`;
}

function row(content: string, width: number): string {
	const sep = getMuted();
	const inset = Math.max(0, width - 4);
	const clamped = clampLineToWidth(content, inset);
	return `${sep}│${RESET} ${clamped}${" ".repeat(Math.max(0, inset - visibleWidth(clamped)))} ${sep}│${RESET}`;
}

// ── Section management ───────────────────────────────────────────────────────

interface SettingSection {
	name: string;
	firstItemIndex: number;
	lastItemIndex: number;
}

function deriveSections(settings: SettingDef[]): SettingSection[] {
	const sections: SettingSection[] = [];
	let current: SettingSection | null = null;
	for (let i = 0; i < settings.length; i++) {
		const s = settings[i];
		const sectionName = s.section ?? s.tab ?? "General";
		if (!current || current.name !== sectionName) {
			if (current) sections.push(current);
			current = { name: sectionName, firstItemIndex: i, lastItemIndex: i };
		} else {
			current.lastItemIndex = i;
		}
	}
	if (current) sections.push(current);
	return sections;
}

function findActiveSection(
	sections: SettingSection[],
	selectedIndex: number,
): number {
	for (let i = sections.length - 1; i >= 0; i--) {
		if (sections[i].firstItemIndex <= selectedIndex) return i;
	}
	return 0;
}

// ── Main overlay class ───────────────────────────────────────────────────────

export class SettingsSelectorOverlay implements Component {
	public visible = false;
	private _message = "";
	private _settings: SettingDef[] = [];
	private _filtered: SettingDef[] = [];
	private _tabs: SettingTab[] = [];
	private _currentTabId = 0;
	private _selectedIndex = 0;
	private _selectedOptionIndex = 0;
	private _inDetailView = false;
	private _searchQuery = "";
	private _availableHeight: number | undefined;

	setMaxHeight(height: number): void {
		this._availableHeight = Math.max(1, Math.floor(height));
	}

	get selectedIndex(): number { return this._selectedIndex; }
	get selectedOptionIndex(): number { return this._selectedOptionIndex; }
	get inDetailView(): boolean { return this._inDetailView; }
	get currentTabId(): number { return this._currentTabId; }

	invalidate(): void {
		// Rendering is inexpensive and reflects terminal resize and theme changes.
	}

	setSettings(settings: SettingDef[]): void {
		this._settings = settings;
		this._tabs = deriveTabs(settings);
		this._currentTabId = 0;
		this._filtered = filterSettingsForTab(settings, this._tabs[0]?.name ?? "");
		if (this._selectedIndex >= this._filtered.length) {
			this._selectedIndex = Math.max(0, this._filtered.length - 1);
		}
		this.invalidate();
	}

	setMessage(message: string): void {
		this._message = message;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this._inDetailView = false;
		this._selectedIndex = 0;
		this._selectedOptionIndex = 0;
		this._searchQuery = "";
		this._currentTabId = 0;
		this._filtered = filterSettingsForTab(this._settings, this._tabs[0]?.name ?? "");
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
		if (!this.visible || !data) return null;
		if (data === "\x03") return { type: "close" };
		if (data === "\x1b") {
			if (this._inDetailView) {
				this._inDetailView = false;
				return null;
			}
			if (this._searchQuery) {
				this._searchQuery = "";
				this._filtered = filterSettingsForTab(this._settings, this._tabs[this._currentTabId]?.name ?? "");
				this._selectedIndex = 0;
				return null;
			}
			return { type: "close" };
		}
		if (this._inDetailView) return this.handleDetailInput(data);
		if (data === "\x7f" || data === "\x08") {
			this._searchQuery = this._searchQuery.slice(0, -1);
		} else if ([...data].length === 1 && data >= " " && (data !== " " || this._searchQuery)) {
			this._searchQuery += data;
		} else {
			return this.handleMenuInput(data);
		}
		this._filtered = this._searchQuery
			? filterSettings(this._settings, this._searchQuery)
			: filterSettingsForTab(this._settings, this._tabs[this._currentTabId]?.name ?? "");
		this._selectedIndex = 0;
		return null;
	}

	private handleMenuInput(data: string): SettingsSelectorAction | null {
		if (data === "\r" || data === "\n" || data === " ") {
			const s = this._filtered[this._selectedIndex];
			if (!s) return null;
			if (s.name.toLowerCase() === "model") {
				return { type: "open", settingName: s.name };
			}
			this._inDetailView = true;
			this._selectedOptionIndex = s.options ? Math.max(0, s.options.findIndex(option => option.current)) : 0;
			this.invalidate();
			return null;
		}

		if (data === "\t" || data === "\x1b[Z") {
			const sections = deriveSections(this._filtered);
			if (sections.length) {
				const delta = data === "\t" ? 1 : -1;
				const next = (findActiveSection(sections, this._selectedIndex) + delta + sections.length) % sections.length;
				this._selectedIndex = sections[next].firstItemIndex;
			}
			return null;
		}

		if (data === "\x1b[A" || data === "\x1bOA") {
			this.moveSelection(-1);
			return null;
		}
		if (data === "\x1b[B" || data === "\x1bOB") {
			this.moveSelection(1);
			return null;
		}
		if (data === "\x1b[D" || data === "\x1bOD") {
			this._switchTab(this._currentTabId - 1);
			return null;
		}
		if (data === "\x1b[C" || data === "\x1bOC") {
			this._switchTab(this._currentTabId + 1);
			return null;
		}
		if (data === "\x1b[5~") {
			this.jumpSection(-1);
			return null;
		}
		if (data === "\x1b[6~") {
			this.jumpSection(1);
			return null;
		}
		return null;
	}

	private handleDetailInput(data: string): SettingsSelectorAction | null {
		const s = this._filtered[this._selectedIndex];
		if (!s) return { type: "close" };

		if (data === "\t" || data === "\x08") {
			this._inDetailView = false;
			this.invalidate();
			return null;
		}

		if (data === "\r" || data === "\n" || data === " ") {
			const opt = s.options?.[this._selectedOptionIndex];
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

	private _switchTab(nextId: number): void {
		if (this._tabs.length <= 1) return;
		if (nextId < 0) nextId = this._tabs.length - 1;
		if (nextId >= this._tabs.length) nextId = 0;
		this._currentTabId = nextId;
		const tabName = this._tabs[nextId].name;
		this._filtered = filterSettingsForTab(this._settings, tabName);
		this._searchQuery = "";
		this._selectedIndex = 0;
		this.invalidate();
	}

	private jumpSection(delta: -1 | 1): void {
		const len = this._filtered.length;
		if (!len) return;
		const jump = 8;
		this._selectedIndex = Math.max(0, Math.min(this._selectedIndex + delta * jump, len - 1));
		this.invalidate();
	}

	private moveSelection(delta: number): void {
		const n = this._filtered.length;
		if (!n) return;
		this._selectedIndex = (this._selectedIndex + delta + n) % n;
		this.invalidate();
	}

	private moveOptionSelection(delta: number): void {
		const s = this._filtered[this._selectedIndex];
		if (!s) return;
		const n = s.options?.length ?? 0;
		if (!n) return;
		this._selectedOptionIndex = ((this._selectedOptionIndex + delta) % n + n) % n;
		this.invalidate();
	}

	public render(width: number): string[] {
		if (!this.visible || width < 1) return [];
		const height = this._availableHeight ?? (process.stdout.rows || 40);
		const inner = Math.max(0, width - 4);
		const selected = this._filtered[this._selectedIndex];
		const lines = [topBorder(width, " Settings "), row(this.renderTabs(inner), width), divider(width)];
		if (this._searchQuery) {
			lines.push(row(`${getHeader()}Search${RESET}  ${this._searchQuery}  ${getMuted()}${this._filtered.length} matches${RESET}`, width));
		}
		const contentRows = Math.max(1, height - lines.length - 6);
		const content = this._inDetailView
			? this.renderOptions(inner, contentRows)
			: this.renderSettings(inner, contentRows);
		for (const line of content) lines.push(row(line, width));
		lines.push(row("", width));
		lines.push(row(`${getMuted()}${selected?.description ?? "Choose a setting to configure Logician."}${RESET}`, width));
		lines.push(row(selected?.warning
			? `${getWarning()}⚠ ${selected.warning}${RESET}`
			: `${getMuted()}${this._message}${RESET}`, width));
		lines.push(divider(width));
		const hint = this._inDetailView
			? "↑↓ select · Enter/Space apply · Tab/Esc back"
			: inner < 75
				? "↑↓ select · ←→ tabs · Tab section · Enter edit · Esc back"
				: "↑↓ select · Enter/Space change · Tab section · ←→ tabs · Type to search · Esc close";
		lines.push(row(`${getMuted()}${hint}${RESET}`, width), bottomBorder(width));
		return lines.slice(0, height).map(line => clampLineToWidth(line, width));
	}

	private renderTabs(width: number): string {
		const symbols: Record<string, string> = { Model: "◇", Behavior: "≡", Tools: "⚒", Guards: "◆", Appearance: "◐" };
		const labels = this._tabs.map(tab => ` ${symbols[tab.name] ?? "·"} ${tab.name} `);
		let start = this._currentTabId;
		let end = start + 1;
		let used = visibleWidth(labels[start] ?? "") + 4;
		while (start > 0 && used + visibleWidth(labels[start - 1]) + 1 <= width) {
			used += visibleWidth(labels[--start]) + 1;
		}
		while (end < labels.length && used + visibleWidth(labels[end]) + 1 <= width) {
			used += visibleWidth(labels[end++]) + 1;
		}
		return `${start > 0 ? "‹ " : ""}${labels.slice(start, end).map((label, offset) =>
			start + offset === this._currentTabId
				? `${getHeader()}${BOLD}\x1b[7m${label}${RESET}`
				: `${getMuted()}${label}${RESET}`,
		).join(" ")}${end < labels.length ? " ›" : ""}`;
	}

	private renderSettings(width: number, height: number): string[] {
		if (!this._filtered.length) {
			return Array.from({ length: height }, (_, i) => i === 0
				? `${getMuted()}${this._settings.length ? "No matching settings · Backspace to edit · Esc clear" : "No settings available"}${RESET}` : "");
		}
		const sections = deriveSections(this._filtered);
		const active = findActiveSection(sections, this._selectedIndex);
		const sidebar = width >= 70 && !this._searchQuery
			? Math.min(22, Math.max(14, ...sections.map(section => visibleWidth(section.name) + 3))) : 0;
		const paneWidth = Math.max(0, width - (sidebar ? sidebar + 3 : 0));
		const labelWidth = Math.min(32, Math.floor(paneWidth * 0.55), Math.max(...this._filtered.map(s => visibleWidth(s.name))));
		const rows: string[] = [];
		let selectedRow = 0;
		for (let i = 0; i < this._filtered.length; i++) {
			const setting = this._filtered[i];
			const section = sections.find(section => section.firstItemIndex === i);
			if (section) {
				if (i > 0) rows.push("");
				rows.push(`${getMuted()}  \x1b[4m${section.name}${RESET}`);
			}
			const isSelected = i === this._selectedIndex;
			if (isSelected) selectedRow = rows.length;
			const label = clampLineToWidth(setting.name, labelWidth);
			const pad = " ".repeat(Math.max(0, labelWidth - visibleWidth(label)));
			const cursor = isSelected ? `${getHeader()}❯${RESET}` : " ";
			const color = isSelected ? getWarning() + BOLD : theme.fgRaw("text");
			const valueColor = isSelected ? getWarning() + BOLD : getMuted();
			rows.push(`${cursor} ${color}${label}${pad}${RESET}  ${valueColor}${setting.currentValue}${RESET}${setting.warning ? ` ${getWarning()}⚠${RESET}` : ""}`);
		}
		const start = Math.max(0, Math.min(selectedRow - Math.floor(height / 2), rows.length - height));
		const sidebarStart = Math.max(0, active - height + 1);
		return Array.from({ length: height }, (_, index) => {
			const content = clampLineToWidth(rows[start + index] ?? "", paneWidth);
			if (!sidebar) return content;
			const section = sections[sidebarStart + index];
			const name = clampLineToWidth(section?.name ?? "", sidebar);
			const color = sidebarStart + index === active ? getHeader() + BOLD : getMuted();
			return `${color}${name}${RESET}${" ".repeat(Math.max(0, sidebar - visibleWidth(name)))} ${getMuted()}│${RESET} ${content}`;
		});
	}

	private renderOptions(width: number, height: number): string[] {
		const setting = this._filtered[this._selectedIndex];
		if (!setting) return Array.from({ length: height }, () => "");
		const lines = [`${getHeader()}${BOLD}${setting.section ?? setting.tab ?? "General"}${RESET} ${getMuted()}/ ${setting.name}${RESET}`, ""];
		const count = Math.max(1, height - lines.length);
		const options = setting.options ?? [];
		const start = Math.max(0, Math.min(this._selectedOptionIndex - Math.floor(count / 2), options.length - count));
		for (let i = start; i < Math.min(options.length, start + count); i++) {
			const option = options[i];
			const selected = i === this._selectedOptionIndex;
			const color = selected ? getWarning() + BOLD : theme.fgRaw("text");
			const mark = typeof option.toggleOn === "boolean"
				? option.toggleOn ? `${getSuccess()} [on]` : `${getMuted()} [off]` : "";
			lines.push(clampLineToWidth(`${selected ? getHeader() + "❯" : " "}${RESET} ${color}${option.label}${RESET}${mark}${option.current ? `${getSuccess()} ✓` : ""}${RESET}`, width));
		}
		while (lines.length < height) lines.push("");
		return lines.slice(0, height);
	}
}
