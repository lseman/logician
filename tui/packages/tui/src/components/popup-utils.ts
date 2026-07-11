// ── Popup Utilities — shared beautiful popup rendering helpers ───────────────
// Provides a consistent, modern popup design system:
//   - Rounded corner boxes with colored header bars
//   - Selected-item background highlight
//   - Category badges with color coding
//   - Status indicator dots
//   - Action bar with keyboard hints

import { theme } from "../layers/theme/theme.ts";
import { visibleWidth, clampLineToWidth } from "../layers/core/tui-core.ts";

// ── ANSI codes ──────────────────────────────────────────────────────────────

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const BG_HIGHLIGHT = "\x1b[48;2;";
const FG_OVERRIDE = "\x1b[38;2;";

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
const getSelectedFg = (): string => theme.fg("selected", "");
const getMuted = (): string => theme.fg("muted", "");
const getActive = (): string => theme.fg("active", "");
const getSuccess = (): string => theme.fg("success", "");
const getError = (): string => theme.fg("error", "");
const getWarning = (): string => theme.fg("warning", "");

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
	const headerPad = Math.max(0, innerWidth - headerVisible);

	const topLine = `${headerFg}${BOX.horiz.repeat(popupWidth)}${RESET}`;

	const headerLine = `${headerFg}${" ".repeat(pad)}${headerContent}${" ".repeat(
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

export function renderSeparator(
	popupWidth: number,
	_pad: number = 1,
): string {
	const sep = getMuted();
	return `${sep}${BOX.sepHoriz.repeat(popupWidth)}${RESET}`;
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

// ── Clamp all lines to terminal width ───────────────────────────────────────

export function clampPopupLines(lines: string[], width: number): string[] {
	return lines.map((line) => clampLineToWidth(line, width));
}

// ── Category badge colors ───────────────────────────────────────────────────

export const CATEGORY_BADGE_COLORS: Record<string, string> = {
	help: "\x1b[36m",
	session: "\x1b[33m",
	agent: "\x1b[35m",
	context: "\x1b[34m",
	rag: "\x1b[32m",
	skills: "\x1b[95m",
	reasoning: "\x1b[37m",
	display: "\x1b[93m",
	permissions: "\x1b[31m",
	shortcuts: "\x1b[36m",
	loop: "\x1b[94m",
	misc: "\x1b[90m",
};

// ── Render a category badge ─────────────────────────────────────────────────

export function renderCategoryBadge(cat: string): string {
	const color = CATEGORY_BADGE_COLORS[cat] ?? "\x1b[90m";
	return `${color}[${cat}]${RESET}`;
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
