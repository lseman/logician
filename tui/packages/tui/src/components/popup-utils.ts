// ── Popup Utilities — shared beautiful popup rendering helpers ───────────────
// Provides a consistent, modern popup design system:
//   - Rounded corner boxes with colored header bars
//   - Selected-item background highlight
//   - Category badges with color coding
//   - Status indicator dots
//   - Action bar with keyboard hints

import { theme } from "../layers/theme/theme.ts";
import { visibleWidth, clampLineToWidth, RESET, BOLD, DIM } from "../layers/core/tui-core.ts";

// ── ANSI codes ──────────────────────────────────────────────────────────────

const _BG_HIGHLIGHT = "\x1b[48;2;";
const _FG_OVERRIDE = "\x1b[38;2;";

// ── Box-drawing characters ──────────────────────────────────────────────────

export const BOX = {
	topLeft: "╭",
	topRight: "╮",
	bottomLeft: "╰",
	bottomRight: "╯",
	horiz: "─",
	vert: "│",
	separator: "├",
	sepHoriz: "─",
	sepRight: "┤",
	sepLeft: "┬",
};

// ── Theme helpers ───────────────────────────────────────────────────────────

const getSelectedBg = (): string => theme.fgAsBg("selected");
const BLACK_FG = "\x1b[38;5;16m";

const getHeaderFg = (): string => theme.fg("header", "");
const _getSelectedFg = (): string => theme.fg("selected", "");
const getMuted = (): string => theme.fg("muted", "");
const getActive = (): string => theme.fg("active", "");
const getSuccess = (): string => theme.fg("success", "");
const getError = (): string => theme.fg("error", "");
const getWarning = (): string => theme.fg("warning", "");

// ── Shared layout constants ─────────────────────────────────────────────────

/** Columns consumed by the popup border + 1-col padding on each side. */
export const POPUP_FRAME_OVERHEAD = 4;

// ── Popup design config ─────────────────────────────────────────────────────

export interface PopupConfig {
	/** Title shown in the header bar */
	title: string;
	/** Subtitle / count shown in the header bar */
	subtitle?: string;
	/** Keyboard hints shown in the header bar */
	hints?: string;
	/** Popup width in terminal columns (auto-calculated if not set) */
	width?: number;
	/** Padding inside the popup (default: 1) */
	padding?: number;
	/** Whether to show a bottom action bar */
	showBottomBar?: boolean;
	/** Bottom bar text */
	bottomText?: string;
}

// ── Render a popup frame ────────────────────────────────────────────────────

export function renderPopupFrame(
	config: PopupConfig,
	width: number,
): {
	topLine: string;
	bottomLine: string;
	innerWidth: number;
} {
	const popupWidth =
		config.width ?? Math.max(48, Math.min(width, 120));
	const pad = config.padding ?? 1;
	const innerWidth = Math.max(1, popupWidth - 2 - pad * 2);

	// Header bar with colored background
	const headerFg = getHeaderFg();
	const titleText = config.title;
	const subtitleText = config.subtitle ? ` ${DIM}${config.subtitle}${RESET}` : "";
	const hintsText = config.hints ? `  ${DIM}${config.hints}${RESET}` : "";
	const headerContent = `${BOLD}${titleText}${RESET}${subtitleText}${hintsText}`;
	const headerVisible = visibleWidth(headerContent);
	const _headerPad = Math.max(0, innerWidth - headerVisible);

	const topLine = `${headerFg}${BOX.horiz.repeat(popupWidth)}${RESET}`;

	const _headerLine = `${headerFg}${" ".repeat(pad)}${headerContent}${" ".repeat(
		innerWidth - headerVisible + pad,
	)}${" ".repeat(pad)}${headerFg}${BOX.vert}${RESET}`;

	// Bottom bar
	const bottomFg = getMuted();
	const bottomContent = config.showBottomBar
		? config.bottomText ?? ""
		: "";
	const bottomVisible = visibleWidth(bottomContent);
	const bottomPad = Math.max(0, innerWidth - bottomVisible);

	const bottomLine = `${bottomFg}${" ".repeat(pad)}${bottomContent}${" ".repeat(
		bottomPad,
	)}${" ".repeat(pad)}${RESET}`;

	return { topLine, bottomLine, innerWidth };
}

// ── Render a separator line ─────────────────────────────────────────────────

export function renderSeparator(popupWidth: number): string {
	const sep = getMuted();
	return `${sep}${BOX.sepHoriz.repeat(popupWidth)}${RESET}`;
}

// ── Render a left/right justified line, padded and clamped to width ─────────

export function boxLine(left: string, right: string, width: number): string {
	const leftWidth = visibleWidth(left);
	const rightWidth = visibleWidth(right);
	const gap = Math.max(1, width - leftWidth - rightWidth);
	const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
	const pad = Math.max(0, width - visibleWidth(content));
	return ` ${content}${" ".repeat(pad)} `;
}

// ── Render a single list item ───────────────────────────────────────────────

export interface ListItem {
	/** Left text (label) */
	label: string;
	/** Right text (metadata) */
	metadata?: string;
	/** Whether this item is selected */
	selected?: boolean;
	/** Icon or bullet before the label */
	bullet?: string;
	/** Badge text with color */
	badge?: { text: string; color: string };
	/** Status dot color (green, red, yellow, blue, gray) */
	statusDot?: "green" | "red" | "yellow" | "blue" | "gray" | "active";
	/** Whether to dim the label */
	dim?: boolean;
}

export function renderListItem(
	item: ListItem,
	innerWidth: number,
): string {
	const pad = 1;
	const isSelected = !!item.selected;
	// On a selected row every segment must re-assert the bg after any RESET,
	// otherwise the background would cut out at the first inner reset.
	const bg = isSelected ? getSelectedBg() : "";
	const segReset = isSelected ? `${RESET}${bg}` : RESET;

	// Build left side
	let left = "";

	// Bullet
	const bullet = item.bullet ?? (isSelected ? "▸" : " ");
	left += isSelected
		? `${bg}${BLACK_FG}${BOLD}${bullet}${segReset}`
		: `${getMuted()}${bullet}${RESET}`;

	// Badge (if present, render before label)
	if (item.badge) {
		const badgeText = `[${item.badge.text}]`;
		left += isSelected
			? ` ${bg}${BLACK_FG}${badgeText}${segReset}`
			: ` ${item.badge.color}${badgeText}${RESET}`;
	}

	// Status dot
	if (item.statusDot) {
		const dotColors: Record<string, string> = {
			green: getSuccess(),
			red: getError(),
			yellow: getWarning(),
			blue: getHeaderFg(),
			gray: getMuted(),
			active: getActive(),
		};
		const dot = isSelected ? BLACK_FG : (dotColors[item.statusDot] ?? getMuted());
		left += isSelected ? ` ${bg}${dot}●${segReset}` : ` ${dot}●${RESET}`;
	}

	left += " ";

	// Label
	if (isSelected) {
		left += `${bg}${BLACK_FG}${BOLD}${item.label}${segReset}`;
	} else {
		const labelColor = item.dim ? DIM : "";
		left += `${labelColor}${item.label}${RESET}`;
	}

	// Right side (metadata)
	let right = item.metadata ?? "";
	if (right) {
		right = isSelected
			? `${bg}${BLACK_FG}${right}${segReset}`
			: `${DIM}${right}${RESET}`;
	}

	// Combine with spacing
	const leftVisible = visibleWidth(left);
	const rightVisible = visibleWidth(right);
	const gap = Math.max(1, innerWidth - leftVisible - rightVisible);
	const gapFill = isSelected ? `${bg}${" ".repeat(gap)}` : " ".repeat(gap);
	const content = right ? `${left}${gapFill}${right}` : left;
	const padRight = Math.max(0, innerWidth - visibleWidth(content));

	if (isSelected) {
		return `${bg}${" ".repeat(pad)}${content}${bg}${" ".repeat(
			padRight + pad,
		)}${RESET}`;
	}

	return `${" ".repeat(pad)}${content}${" ".repeat(padRight + pad)}`;
}

// ── Render a question display ───────────────────────────────────────────────

export function renderQuestion(
	question: string,
	innerWidth: number,
): string {
	const pad = 1;
	const icon = `${getHeaderFg()}❯${RESET}`;
	const text = `${BOLD}${question}${RESET}`;
	const line = `${icon} ${text}`;
	const lineVisible = visibleWidth(line);
	const padRight = Math.max(0, innerWidth - lineVisible);
	return `${" ".repeat(pad)}${line}${" ".repeat(padRight + pad)}`;
}

// ── Render a choice option ──────────────────────────────────────────────────

export interface ChoiceOption {
	label: string;
	value: string;
	selected?: boolean;
	description?: string;
}

export function renderChoiceOption(
	option: ChoiceOption,
	innerWidth: number,
	index: number,
): string {
	const pad = 1;
	const isSelected = !!option.selected;
	// On a selected row every segment must re-assert the bg after any RESET,
	// otherwise the background would cut out at the first inner reset.
	const bg = isSelected ? getSelectedBg() : "";
	const segReset = isSelected ? `${RESET}${bg}` : RESET;

	const numLabel = `${index + 1}`;

	let left = "";
	if (isSelected) {
		left += `${bg}${BLACK_FG}${BOLD}▸${segReset} `;
	} else {
		left += `${DIM}${numLabel}${RESET}  `;
	}

	// Label
	if (isSelected) {
		left += `${bg}${BLACK_FG}${BOLD}${option.label}${segReset}`;
	} else {
		left += `${option.label}${RESET}`;
	}

	// Description
	let right = "";
	if (option.description) {
		right = isSelected
			? `${bg}${BLACK_FG}${option.description}${segReset}`
			: `${DIM}${option.description}${RESET}`;
	}

	// Combine
	const leftVisible = visibleWidth(left);
	const rightVisible = visibleWidth(right);
	const gap = Math.max(1, innerWidth - leftVisible - rightVisible);
	const gapFill = isSelected ? `${bg}${" ".repeat(gap)}` : " ".repeat(gap);
	const content = right ? `${left}${gapFill}${right}` : left;
	const padRight = Math.max(0, innerWidth - visibleWidth(content));

	if (isSelected) {
		return `${bg}${" ".repeat(pad)}${content}${bg}${" ".repeat(
			padRight + pad,
		)}${RESET}`;
	}

	return `${" ".repeat(pad)}${content}${" ".repeat(padRight + pad)}`;
}

// ── Shared list-popup navigation + frame ────────────────────────────────────
// Common to the simple "list, select, confirm, close" overlays (theme/model/
// reasoner selectors, mcp/plugin managers): identical cache-invalidation,
// arrow/vim/page-key handling, and top-rule/title/separator/bottom-rule frame.

/** Result of parsing a keypress against the standard list-popup key bindings. */
export type PopupListNavResult =
	| { type: "move"; delta: number }
	| { type: "confirm" }
	| { type: "close" }
	| null;

/**
 * Parses a keypress against the shared list-popup bindings: ↑/k, ↓/j,
 * PageUp/PageDown (±8), Enter/confirm, Esc/Ctrl-C/q/close. Returns null if
 * the key isn't one of these — callers should fall through to their own
 * handling (e.g. space-to-toggle, r-to-refresh) in that case.
 */
export function parsePopupListNav(data: string): PopupListNavResult {
	if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
		return { type: "close" };
	}
	if (data === "\r" || data === "\n") {
		return { type: "confirm" };
	}
	if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
		return { type: "move", delta: -1 };
	}
	if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
		return { type: "move", delta: 1 };
	}
	if (data === "\x1b[5~") {
		return { type: "move", delta: -8 };
	}
	if (data === "\x1b[6~") {
		return { type: "move", delta: 8 };
	}
	return null;
}

export interface ListPopupFrameOptions {
	popupWidth: number;
	innerWidth: number;
	/** Title text, e.g. "Theme" */
	title: string;
	/** Subtitle appended after title in muted color, e.g. " (12)" */
	subtitle?: string;
	/** Keyboard hints appended after subtitle, e.g. " ↑↓ select · enter confirm · esc close" */
	hints: string;
	/** Extra line(s) rendered right after the title row (e.g. a config path), before the separator. */
	extraHeaderLines?: string[];
	/** Body lines: the rendered list (or an empty-state status line). */
	bodyLines: string[];
	/** Bottom status bar text. */
	bottomText: string;
}

/** Renders the shared top-rule/title/separator/body/bottom-bar/bottom-rule frame used by list popups. */
export function renderListPopupFrame(opts: ListPopupFrameOptions): string[] {
	const headerFg = getHeaderFg();
	const lines: string[] = [];

	lines.push(`${headerFg}${"─".repeat(opts.popupWidth)}${getMuted()}`);

	const subtitleText = opts.subtitle ?? "";
	const titleLine = `${opts.title}${getMuted()}${subtitleText}${opts.hints}`;
	const titleVisible = visibleWidth(titleLine);
	const titlePad = Math.max(0, opts.innerWidth - titleVisible);
	lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

	if (opts.extraHeaderLines) {
		for (const line of opts.extraHeaderLines) {
			lines.push(renderStatusLine(line, opts.innerWidth));
		}
	}

	lines.push(renderSeparator(opts.popupWidth));
	lines.push(...opts.bodyLines);
	lines.push(renderSeparator(opts.popupWidth));
	lines.push(renderStatusLine(opts.bottomText, opts.innerWidth));
	lines.push(`${headerFg}${"─".repeat(opts.popupWidth)}${getMuted()}`);

	return lines;
}

/** Renders a windowed list body (with "N more" indicators) using the shared SelectorController window. */
export function renderListPopupBody<T>(
	items: T[],
	selection: { window(count: number, maxRows: number): { start: number; end: number } },
	innerWidth: number,
	maxRows: number,
	renderItem: (item: T, index: number) => string,
	emptyText: string,
): string[] {
	const lines: string[] = [];
	if (!items.length) {
		lines.push(renderStatusLine(emptyText, innerWidth, getWarning()));
		return lines;
	}
	const { start, end } = selection.window(items.length, maxRows);
	if (start > 0) {
		lines.push(renderStatusLine(`↑ ${start} more`, innerWidth));
	}
	for (let i = start; i < end; i++) {
		lines.push(renderItem(items[i], i));
	}
	if (end < items.length) {
		lines.push(renderStatusLine(`↓ ${items.length - end} more`, innerWidth));
	}
	return lines;
}

// ── Clamp all lines to terminal width ───────────────────────────────────────

export function clampPopupLines(lines: string[], width: number): string[] {
	return lines.map((line) => clampLineToWidth(line, width));
}

// ── Render a section divider ────────────────────────────────────────────────

export function renderSectionDivider(
	title: string,
	innerWidth: number,
): string {
	const pad = 1;
	const color = getHeaderFg();
	const divider = `${color}── ${title} ──${RESET}`;
	const divVisible = visibleWidth(divider);
	const padRight = Math.max(0, innerWidth - divVisible);
	return `${" ".repeat(pad)}${divider}${" ".repeat(padRight + pad)}`;
}

// ── Render a status line (small info text) ──────────────────────────────────

export function renderStatusLine(
	text: string,
	innerWidth: number,
	color: string = getMuted(),
): string {
	const pad = 1;
	const line = `${color}${text}${RESET}`;
	const lineVisible = visibleWidth(line);
	const padRight = Math.max(0, innerWidth - lineVisible);
	return `${" ".repeat(pad)}${line}${" ".repeat(padRight + pad)}`;
}
