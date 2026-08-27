// ── Popup Utilities — shared beautiful popup rendering helpers ───────────────
// Provides a consistent, modern popup design system:
//   - Rounded corner boxes with colored header bars
//   - Selected-item background highlight
//   - Category badges with color coding
//   - Status indicator dots
//   - Action bar with keyboard hints

import {
	BOLD,
	type Component,
	clampLineToWidth,
	DIM,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import { SelectorController } from "./selector-controller.ts";

// ── ANSI codes ──────────────────────────────────────────────────────────────

// ── Box-drawing characters ──────────────────────────────────────────────────

export const BOX = {
	topLeft: "┌",
	topRight: "┐",
	bottomLeft: "└",
	bottomRight: "┘",
	horiz: "─",
	vert: "│",
	separator: "├",
	sepHoriz: "─",
	sepRight: "┤",
	sepLeft: "┬",
};

// ── Theme helpers ───────────────────────────────────────────────────────────

const getHeaderFg = (): string => theme.fgRaw("header");
const getSelectedFg = (): string => theme.fgRaw("selected");
const getMuted = (): string => theme.fgRaw("muted");
const getActive = (): string => theme.fgRaw("active");
const getSuccess = (): string => theme.fgRaw("success");
const getError = (): string => theme.fgRaw("error");
const getWarning = (): string => theme.fgRaw("warning");

// ── Shared layout constants ─────────────────────────────────────────────────

/** Columns consumed by the popup border + 1-col padding on each side. */
export const POPUP_FRAME_OVERHEAD = 4;

// ── Render a separator line ─────────────────────────────────────────────────

export function renderSeparator(popupWidth: number): string {
	const sep = getMuted();
	return `${sep}${BOX.sepHoriz.repeat(popupWidth)}${RESET}`;
}

// ── Render a left/right justified line, padded and clamped to width ─────────

function boxLine(left: string, right: string, width: number): string {
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
	/** Whether this item is the currently applied value. */
	current?: boolean;
	/** Icon or bullet before the label */
	bullet?: string;
	/** Badge text with color */
	badge?: { text: string; color: string };
	/** Status dot color (green, red, yellow, blue, gray) */
	statusDot?: "green" | "red" | "yellow" | "blue" | "gray" | "active";
	/** Whether to dim the label */
	dim?: boolean;
}

export function renderListItem(item: ListItem, innerWidth: number): string {
	const pad = 1;
	const isSelected = !!item.selected;
	let left = "";
	const bullet = item.bullet ?? (isSelected ? "❯" : " ");
	left += isSelected
		? `${getSelectedFg()}${BOLD}${bullet}${RESET}`
		: `${getMuted()}${bullet}${RESET}`;

	if (item.badge) {
		const badgeText = `[${item.badge.text}]`;
		left += isSelected
			? ` ${getSelectedFg()}${badgeText}${RESET}`
			: ` ${item.badge.color}${badgeText}${RESET}`;
	}

	if (item.statusDot) {
		const dotColors: Record<string, string> = {
			green: getSuccess(),
			red: getError(),
			yellow: getWarning(),
			blue: getHeaderFg(),
			gray: getMuted(),
			active: getActive(),
		};
		const dot = dotColors[item.statusDot] ?? getMuted();
		left += ` ${dot}●${RESET}`;
	}

	left += " ";
	if (isSelected) {
		left += `${getSelectedFg()}${BOLD}${item.label}${RESET}`;
	} else {
		const labelColor = item.dim ? DIM : "";
		left += `${labelColor}${item.label}${RESET}`;
	}
	if (item.current) {
		left += ` ${getActive()}${BOLD}✓${RESET}`;
	}

	let right = item.metadata ?? "";
	if (right) {
		right = isSelected
			? `${getActive()}${right}${RESET}`
			: `${DIM}${right}${RESET}`;
	}

	const leftVisible = visibleWidth(left);
	const rightVisible = visibleWidth(right);
	const gap = Math.max(1, innerWidth - leftVisible - rightVisible);
	const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
	const padRight = Math.max(0, innerWidth - visibleWidth(content));

	return `${" ".repeat(pad)}${content}${" ".repeat(padRight + pad)}`;
}

// ── Render a question display ───────────────────────────────────────────────

export function renderQuestion(question: string, innerWidth: number): string {
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
	const numLabel = `${index + 1}`;

	let left = "";
	if (isSelected) {
		left += `${getSelectedFg()}${BOLD}❯${RESET} `;
	} else {
		left += `${DIM}${numLabel}${RESET}  `;
	}

	if (isSelected) {
		left += `${getSelectedFg()}${BOLD}${option.label}${RESET}`;
	} else {
		left += `${option.label}${RESET}`;
	}

	let right = "";
	if (option.description) {
		right = isSelected
			? `${getActive()}${option.description}${RESET}`
			: `${DIM}${option.description}${RESET}`;
	}

	const leftVisible = visibleWidth(left);
	const rightVisible = visibleWidth(right);
	const gap = Math.max(1, innerWidth - leftVisible - rightVisible);
	const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
	const padRight = Math.max(0, innerWidth - visibleWidth(content));

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
	const border = theme.fg("borderMuted", "");
	const lines: string[] = [];
	const framed = (content = ""): string => {
		const clipped = clampLineToWidth(content, Math.max(1, opts.popupWidth - 2));
		return `${border}│${RESET}${clipped}${" ".repeat(
			Math.max(0, opts.popupWidth - 2 - visibleWidth(clipped)),
		)}${border}│${RESET}`;
	};

	lines.push(
		`${border}┌${"─".repeat(Math.max(0, opts.popupWidth - 2))}┐${RESET}`,
	);

	const subtitleText = opts.subtitle ?? "";
	const titleLine = ` ${headerFg}${BOLD}${opts.title}${RESET}${getMuted()}${subtitleText}${RESET}`;
	const titleVisible = visibleWidth(titleLine);
	const titlePad = Math.max(0, opts.innerWidth - titleVisible);
	lines.push(framed(`${titleLine}${" ".repeat(titlePad + 1)}`));
	lines.push(framed());

	if (opts.extraHeaderLines) {
		for (const line of opts.extraHeaderLines) {
			lines.push(framed(renderStatusLine(line, opts.innerWidth)));
		}
	}

	for (const bodyLine of opts.bodyLines) lines.push(framed(bodyLine));
	lines.push(framed());
	if (opts.bottomText) {
		lines.push(framed(renderStatusLine(opts.bottomText, opts.innerWidth)));
	}
	lines.push(framed(renderStatusLine(opts.hints.trim(), opts.innerWidth)));
	lines.push(
		`${border}└${"─".repeat(Math.max(0, opts.popupWidth - 2))}┘${RESET}`,
	);

	return lines;
}

/** Renders a windowed list body (with "N more" indicators) using the shared SelectorController window. */
export function renderListPopupBody<T>(
	items: T[],
	selection: {
		window(count: number, maxRows: number): { start: number; end: number };
	},
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
		lines.push(renderStatusLine(`↑ ${start} more above`, innerWidth));
	}
	for (let i = start; i < end; i++) {
		lines.push(renderItem(items[i], i));
	}
	if (end < items.length) {
		lines.push(
			renderStatusLine(`↓ ${items.length - end} more below`, innerWidth),
		);
	}
	return lines;
}

// ── Clamp all lines to terminal width ───────────────────────────────────────

export function clampPopupLines(lines: string[], width: number): string[] {
	return lines.map(line => clampLineToWidth(line, width));
}

// ── Render a section divider ────────────────────────────────────────────────

function renderSectionDivider(title: string, innerWidth: number): string {
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

// ── Generic list-select overlay ─────────────────────────────────────────────
// Base for the "list, select, confirm, close" popups (theme/model/reasoner
// selectors): identical show/hide/cache-invalidation/handleInput/render
// shape, differing only in item type, its ListItem mapping, and title/hints/
// empty-state text. Subclasses stay tiny wrappers that supply that config.

export type SelectAction<T> = { type: "select"; item: T } | { type: "close" };

export interface ListSelectorConfig<T> {
	title: string;
	hints?: string;
	emptyText: string;
	/** Shown as the bottom-bar text whenever no transient message (e.g. "Switching to X...") is set. */
	defaultMessage: string;
	maxRows?: number;
	toItem: (
		this: ListSelectorOverlay<T>,
		item: T,
		index: number,
		selectedIndex: number,
	) => ListItem;
}

/** Shared helper: find the index of the first item whose active flag is true. */
function findActiveIndex<T>(items: T[], active: (item: T) => boolean): number {
	const idx = items.findIndex(active);
	return idx >= 0 ? idx : 0;
}

export class ListSelectorOverlay<T> implements Component {
	public visible = false;
	protected items: T[] = [];
	protected selection = new SelectorController();
	protected message = "";
	/** Set to mark a specific item as "current" (shown with a ✓). Read by `toItem` via `this`. */
	activeId: string | undefined;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	constructor(private readonly config: ListSelectorConfig<T>) {}

	setItems(items: T[], preferredIndex = this.selection.index): void {
		this.items = items;
		this.selection.set(preferredIndex, this.items.length);
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

	/** Named distinctly from `handleInput` so subclasses can expose their own
	 * narrowed action type (e.g. `{ type: "select"; reasoner: ReasonerInfo }`)
	 * without hitting TS's method-override return-type variance rules. */
	handleListInput(data: string): SelectAction<T> | null {
		if (!this.visible) return null;

		const nav = parsePopupListNav(data);
		if (nav?.type === "close") return { type: "close" };
		if (nav?.type === "confirm") {
			const item = this.items[this.selection.index];
			return item ? { type: "select", item } : { type: "close" };
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
			this.items,
			this.selection,
			innerWidth,
			this.config.maxRows ?? 10,
			(item, i) =>
				renderListItem(
					this.config.toItem.call(this, item, i, this.selection.index),
					innerWidth,
				),
			this.config.emptyText,
		);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: this.config.title,
			subtitle: ` (${this.items.length})`,
			hints: this.config.hints ?? " ↑↓ select · enter confirm · esc close",
			bodyLines,
			bottomText: this.message || this.config.defaultMessage,
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.items.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}

// ── List-selector factory ────────────────────────────────────────────────────
// Creates a typed list-selector overlay with a narrowed `handleInput` return
// type. Each selector passes its config and an action-key; the factory produces
// a class whose handleInput wraps the base SelectAction into a typed variant.

type ListSelectorAction<T> = { type: "select"; item: T } | { type: "close" };

/** Constructor signature for list-selector overlays created by `createListSelector`. */
export interface ListSelectorCtor<T> {
	new (): ListSelectorOverlay<T> & {
		handleInput(data: string): ListSelectorAction<T> | null;
	};
	prototype: ListSelectorOverlay<T> & {
		handleInput(data: string): ListSelectorAction<T> | null;
	};
}

/**
 * Creates a list-selector overlay constructor with a narrowed `handleInput` return
 * type. The returned class extends `ListSelectorOverlay<T>` and overrides
 * `handleInput` to return `ListSelectorAction<T> | null` instead of the
 * base `SelectAction<T> | null`.
 */
export function createListSelector<T>(
	config: ListSelectorConfig<T>,
): ListSelectorCtor<T> {
	return class ListSelector extends ListSelectorOverlay<T> {
		constructor() {
			super(config);
		}
		handleInput(data: string): ListSelectorAction<T> | null {
			const action = this.handleListInput(data);
			if (!action) return null;
			return action.type === "select"
				? { type: "select", item: action.item }
				: { type: "close" };
		}
	} as unknown as ListSelectorCtor<T>;
}
