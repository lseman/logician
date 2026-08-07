// ── TUI primitives ───────────────────────────────────────────────────────────
// Component/Container model, cursor marker, and ANSI-safe string utilities.
// Deliberately has no dependency on the layout engine (rendering/layout.ts)
// or on TUI itself — those depend on this module, not the other way around,
// so this file must stay a leaf to avoid a Container-before-initialization
// circular-import crash (stack.ts/scroll-view.ts extend Container).

import stringWidth from "string-width";

// ── Interfaces ────────────────────────────────────────────────────────────────

export interface Component {
	render(width: number): string[];
	invalidate?(): void;
}

export interface Focusable {
	focused: boolean;
}

export interface Scrollable extends Component {
	scrollOffset: number;
	totalHeight: number;
	scroll(delta: number): void;
	scrollToBottom(): void;
	isAtBottom: boolean;
	handleMouse?(column: number, row: number): boolean;
}

export function isFocusable(c: Component | null): c is Component & Focusable {
	return c !== null && "focused" in c;
}

// ── Cursor marker ────────────────────────────────────────────────────────────

export const CURSOR_MARKER = "\x1b_pi:c\x07";

// ── Shared ANSI codes ─────────────────────────────────────────────────────────
// Every component previously redeclared these identically — single source now.

export const RESET = "\x1b[0m";
export const BOLD = "\x1b[1m";
export const DIM = "\x1b[2m";

// ── Width utilities ──────────────────────────────────────────────────────────

// Ported from pi's tui/src/utils.ts visibleWidth: an ASCII fast path (skip
// regex/grapheme work entirely — most rendered text is plain ASCII with ANSI
// color codes wrapped around it, not the codes themselves) plus a persistent
// FIFO-bounded cache for the slow path, so repeated calls with the same
// styled string across frames don't re-strip-and-measure every time. Unlike
// pi we keep string-width (already a dependency) as the wide-character/CJK
// width authority for the slow path rather than porting pi's own Unicode
// tables — this only changes which library computes the non-ASCII case, not
// the fast-path/caching structure that made pi's version cheap.
function isPrintableAscii(text: string): boolean {
	for (let i = 0; i < text.length; i++) {
		const code = text.charCodeAt(i);
		if (code < 0x20 || code > 0x7e) return false;
	}
	return true;
}

const WIDTH_CACHE_SIZE = 512;
const widthCache = new Map<string, number>();

export function visibleWidth(text: string): number {
	if (text.length === 0) return 0;
	if (isPrintableAscii(text)) return text.length;

	const cached = widthCache.get(text);
	if (cached !== undefined) return cached;

	const width = stringWidth(
		text
			.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
			.replace(/\x1b[\]_][\s\S]*?(?:\x07|\x1b\\)/g, ""),
	);

	if (widthCache.size >= WIDTH_CACHE_SIZE) {
		const firstKey = widthCache.keys().next().value;
		if (firstKey !== undefined) widthCache.delete(firstKey);
	}
	widthCache.set(text, width);

	return width;
}

const graphemeSegmenter = new Intl.Segmenter(undefined, {
	granularity: "grapheme",
});

function graphemeAt(text: string, offset: number): string {
	const segment = graphemeSegmenter
		.segment(text.slice(offset))
		[Symbol.iterator]()
		.next();
	return segment.done ? "" : segment.value.segment;
}

// Clamp a line to a visible width, preserving ALL escape sequences (CSI colors,
// OSC hyperlinks/markers). Adds no ellipsis — it is used per-frame to guarantee
// a line can never exceed the terminal width and wrap onto the next row (which
// would desync the whole differential frame).
export function clampLineToWidth(text: string, width: number): string {
	let result = "";
	let visible = 0;
	let i = 0;
	while (i < text.length) {
		const ch = text[i];
		if (ch === "\x1b") {
			const next = text[i + 1];
			if (next === "[") {
				// CSI: ESC [ ... <final byte 0x40-0x7E>
				let j = i + 2;
				while (
					j < text.length &&
					!(text.charCodeAt(j) >= 0x40 && text.charCodeAt(j) <= 0x7e)
				)
					j++;
				result += text.slice(i, j + 1);
				i = j + 1;
				continue;
			}
			if (next === "]") {
				// OSC: ESC ] ... terminated by BEL (0x07) or ST (ESC \)
				let j = i + 2;
				while (
					j < text.length &&
					text[j] !== "\x07" &&
					!(text[j] === "\x1b" && text[j + 1] === "\\")
				)
					j++;
				const end = text[j] === "\x07" ? j + 1 : j + 2;
				result += text.slice(i, end);
				i = end;
				continue;
			}
			if (next === "_") {
				// APC (e.g. cursor marker): ESC _ ... ST/BEL
				let j = i + 2;
				while (
					j < text.length &&
					text[j] !== "\x07" &&
					!(text[j] === "\x1b" && text[j + 1] === "\\")
				)
					j++;
				const end = text[j] === "\x07" ? j + 1 : j + 2;
				result += text.slice(i, end);
				i = end;
				continue;
			}
			// Lone ESC — pass through.
			result += ch;
			i++;
			continue;
		}
		// Drop other C0 control bytes (NUL, etc.). A stray NUL — e.g. from
		// reading a binary or corrupted file — truncates the terminal frame and
		// freezes the TUI mid-render. Tabs are expanded to a single space.
		const code = text.charCodeAt(i);
		if (code < 0x20) {
			if (code === 0x09) {
				if (visible + 1 > width) break;
				result += " ";
				visible += 1;
			}
			i++;
			continue;
		}
		const grapheme = graphemeAt(text, i);
		if (!grapheme) break;
		const w = visibleWidth(grapheme);
		if (visible + w > width) break;
		result += grapheme;
		visible += w;
		i += grapheme.length;
	}
	return result;
}

// Ported from pi's compositeTuiLine: a single-pass extraction of the
// before/after segments around the overlay slot, instead of three-to-four
// independent clampLineToWidth/visibleWidth/skipColumns scans over the same
// line. Restricted to lines whose printable content is plain ASCII (no
// grapheme clustering needed — each char is exactly one column), which
// covers the overwhelming majority of rendered lines (styled English text).
// Anything with wide/multi-byte characters falls back to the slower
// generic path below, unchanged.
function isAsciiOnlyLine(text: string): boolean {
	let i = 0;
	while (i < text.length) {
		const ch = text[i];
		if (ch === "\x1b") {
			const next = text[i + 1];
			if (next === "[") {
				let j = i + 2;
				while (
					j < text.length &&
					!(text.charCodeAt(j) >= 0x40 && text.charCodeAt(j) <= 0x7e)
				)
					j++;
				i = j + 1;
				continue;
			}
			if (next === "]" || next === "_") {
				let j = i + 2;
				while (
					j < text.length &&
					text[j] !== "\x07" &&
					!(text[j] === "\x1b" && text[j + 1] === "\\")
				)
					j++;
				i = text[j] === "\x07" ? j + 1 : j + 2;
				continue;
			}
			return false;
		}
		const code = text.charCodeAt(i);
		if (code !== 0x09 && (code < 0x20 || code > 0x7e)) return false;
		i++;
	}
	return true;
}

function compositeTuiLineAsciiFast(
	baseLine: string,
	overlayLine: string,
	startCol: number,
	overlayWidth: number,
	totalWidth: number,
): string {
	let before = "";
	let beforeWidth = 0;
	// `after` is built in two independent stages, mirroring the reference
	// path's skipColumns(baseLine, afterStart) followed by a *separate*
	// clampLineToWidth(..., afterWidth) call on the result — not a single
	// combined boundary check. afterSkipped tracks the skipColumns stage
	// (visible < afterStart: drop everything, codes included, until that
	// threshold); once past it, afterWidth/afterDone track the clamp stage
	// against its own independent budget.
	let after = "";
	let afterWidth = 0;
	let afterDone = false;
	const afterStart = startCol + overlayWidth;
	const afterTargetWidth = Math.max(0, totalWidth - afterStart);
	let col = 0;
	let i = 0;
	while (i < baseLine.length) {
		const ch = baseLine[i];
		if (ch === "\x1b") {
			const next = baseLine[i + 1];
			let j = i + 2;
			if (next === "[") {
				while (
					j < baseLine.length &&
					!(baseLine.charCodeAt(j) >= 0x40 && baseLine.charCodeAt(j) <= 0x7e)
				)
					j++;
				j++;
			} else {
				while (
					j < baseLine.length &&
					baseLine[j] !== "\x07" &&
					!(baseLine[j] === "\x1b" && baseLine[j + 1] === "\\")
				)
					j++;
				j = baseLine[j] === "\x07" ? j + 1 : j + 2;
			}
			const code = baseLine.slice(i, j);
			// clampLineToWidth doesn't check its width bound until the next
			// actual character would be added, so a code sitting right at the
			// boundary (no printable char after it yet to force the decision)
			// still attaches to `before` — hence beforeWidth === col, not <.
			if (beforeWidth === col) before += code;
			// skipColumns keeps everything from the point visible >= afterStart
			// onward, codes included; the clamp stage after that has the same
			// trailing-code-at-boundary inclusion rule as clampLineToWidth.
			if (col >= afterStart && !afterDone) after += code;
			i = j;
			continue;
		}
		// Match clampLineToWidth: tabs become one space, other C0 control
		// bytes are dropped entirely (not copied, not counted as a column).
		const code = ch.charCodeAt(0);
		if (code < 0x20) {
			if (code === 0x09) {
				if (beforeWidth === col && col < startCol) {
					before += " ";
					beforeWidth++;
				}
				if (col >= afterStart && !afterDone) {
					if (afterWidth + 1 > afterTargetWidth) afterDone = true;
					else {
						after += " ";
						afterWidth++;
					}
				}
				col++;
			}
			i++;
			continue;
		}
		if (beforeWidth === col && col < startCol) {
			before += ch;
			beforeWidth++;
		}
		if (col >= afterStart && !afterDone) {
			if (afterWidth + 1 > afterTargetWidth) afterDone = true;
			else {
				after += ch;
				afterWidth++;
			}
		}
		col++;
		i++;
	}
	const beforePad = " ".repeat(Math.max(0, startCol - beforeWidth));
	const overlayClamped = clampLineToWidth(overlayLine, overlayWidth);
	const overlayVisible = visibleWidth(overlayClamped);
	const overlayPad = " ".repeat(Math.max(0, overlayWidth - overlayVisible));
	const afterPad = " ".repeat(Math.max(0, afterTargetWidth - afterWidth));
	return `${before}${beforePad}${RESET}${overlayClamped}${overlayPad}${RESET}${after}${afterPad}`;
}

// Place `overlayLine` (already clamped to `overlayWidth`) into `baseLine` at
// column `startCol`, padding both sides with spaces out to `totalWidth`.
// Resets styles at each boundary so a colored child never bleeds into its
// neighbor — used by Flex's row direction to compose siblings left-to-right.
export function compositeTuiLine(
	baseLine: string,
	overlayLine: string,
	startCol: number,
	overlayWidth: number,
	totalWidth: number,
): string {
	if (isAsciiOnlyLine(baseLine) && isAsciiOnlyLine(overlayLine)) {
		return compositeTuiLineAsciiFast(
			baseLine,
			overlayLine,
			startCol,
			overlayWidth,
			totalWidth,
		);
	}
	const before = clampLineToWidth(baseLine, startCol);
	const beforeWidth = visibleWidth(before);
	const beforePad = " ".repeat(Math.max(0, startCol - beforeWidth));
	const overlay = clampLineToWidth(overlayLine, overlayWidth);
	const overlayPad = " ".repeat(
		Math.max(0, overlayWidth - visibleWidth(overlay)),
	);
	const afterStart = startCol + overlayWidth;
	const afterWidth = Math.max(0, totalWidth - afterStart);
	const after =
		afterWidth > 0
			? clampLineToWidth(skipColumns(baseLine, afterStart), afterWidth)
			: "";
	const afterPad = " ".repeat(Math.max(0, afterWidth - visibleWidth(after)));
	return `${before}${beforePad}${RESET}${overlay}${overlayPad}${RESET}${after}${afterPad}`;
}

// Drop the leading `columns` of visible width from a styled line, preserving
// any escape sequences positioned at or after that point. Used to isolate the
// remainder of a base line after an overlay's slot in compositeTuiLine.
function skipColumns(text: string, columns: number): string {
	let visible = 0;
	let i = 0;
	while (i < text.length && visible < columns) {
		const ch = text[i];
		if (ch === "\x1b") {
			const next = text[i + 1];
			if (next === "[") {
				let j = i + 2;
				while (
					j < text.length &&
					!(text.charCodeAt(j) >= 0x40 && text.charCodeAt(j) <= 0x7e)
				)
					j++;
				i = j + 1;
				continue;
			}
			if (next === "]" || next === "_") {
				let j = i + 2;
				while (
					j < text.length &&
					text[j] !== "\x07" &&
					!(text[j] === "\x1b" && text[j + 1] === "\\")
				)
					j++;
				i = text[j] === "\x07" ? j + 1 : j + 2;
				continue;
			}
			i++;
			continue;
		}
		const codePoint = text.codePointAt(i);
		const character =
			codePoint === undefined ? ch : String.fromCodePoint(codePoint);
		visible += visibleWidth(character);
		i += character.length;
	}
	return text.slice(i);
}

// ── Spacer ───────────────────────────────────────────────────────────────────

export class Spacer implements Component {
	private height: number;

	constructor(height = 1) {
		this.height = height;
	}

	render(width: number): string[] {
		const line = " ".repeat(width);
		return Array(this.height).fill(line);
	}

	invalidate(): void {
		/* no-op */
	}
}

// ── Container ────────────────────────────────────────────────────────────────

export class Container implements Component {
	children: Component[] = [];

	addChild(component: Component): void {
		this.children.push(component);
	}

	removeChild(component: Component): void {
		const idx = this.children.indexOf(component);
		if (idx >= 0) this.children.splice(idx, 1);
	}

	clear(): void {
		this.children = [];
	}

	invalidate(): void {
		for (const child of this.children) {
			child.invalidate?.();
		}
	}

	render(width: number): string[] {
		const lines: string[] = [];
		for (const child of this.children) {
			for (const line of child.render(width)) {
				lines.push(line);
			}
		}
		return lines;
	}
}
