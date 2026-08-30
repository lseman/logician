// ── Ink TUI Core Rendering Primitives ───────────────────────────────────────
// ANSI-safe string utilities for the footer widget system. Mirrors the old TUI's
// primitives.ts but adapted for the Ink/React rendering model.

import stringWidth from "string-width";

/* ════════════════════════════════════════════════════════════════════════════
 *  Shared ANSI codes — single source of truth
 * ════════════════════════════════════════════════════════════════════════════ */

export const RESET = "\x1b[0m";
export const BOLD = "\x1b[1m";
export const DIM = "\x1b[2m";

/* ════════════════════════════════════════════════════════════════════════════
 *  Cursor marker — used by the input bar to park the hardware cursor
 * ════════════════════════════════════════════════════════════════════════════ */

export const CURSOR_MARKER = "\x1b_pi:c\x07";

/* ════════════════════════════════════════════════════════════════════════════
 *  Width utilities — ANSI-aware visible width with caching
 * ════════════════════════════════════════════════════════════════════════════ */

function isPrintableAscii(text: string): boolean {
	for (let i = 0; i < text.length; i++) {
		const code = text.charCodeAt(i);
		if (code < 0x20 || code > 0x7e) return false;
	}
	return true;
}

const WIDTH_CACHE_SIZE = 512;
const widthCache = new Map<string, number>();

/**
 * Compute the visible (display) width of a string, stripping ANSI escape
 * sequences and accounting for wide characters (CJK, emoji, etc.).
 * Uses an ASCII fast path + FIFO-bounded cache for repeated calls.
 */
export function visibleWidth(text: string): number {
	if (text.length === 0) return 0;
	if (isPrintableAscii(text)) return text.length;

	const cached = widthCache.get(text);
	if (cached !== undefined) return cached;

	const stripped = text
		.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "")
		.replace(/\x1b[\]_][\s\S]*?(?:\x07|\x1b\\)/g, "");

	const width = stringWidth(stripped);

	if (widthCache.size >= WIDTH_CACHE_SIZE) {
		const firstKey = widthCache.keys().next().value;
		if (firstKey !== undefined) widthCache.delete(firstKey);
	}
	widthCache.set(text, width);

	return width;
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Clamp line to terminal width — preserves ALL escape sequences
 * ════════════════════════════════════════════════════════════════════════════ */

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

/**
 * Clamp a line to a visible width, preserving ALL escape sequences (CSI colors,
 * OSC hyperlinks). Never adds ellipsis — used per-frame to guarantee a line
 * can never exceed the terminal width and wrap onto the next row.
 */
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
		if (code >= 0x20 && code <= 0x7e) {
			if (visible + 1 > width) break;
			result += text[i];
			visible += 1;
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

/* ════════════════════════════════════════════════════════════════════════════
 *  Text sanitization — remove control chars, collapse whitespace
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * Sanitize terminal text: strip control characters (except ANSI escapes),
 * collapse internal whitespace, and trim. Safe for use in widget labels.
 */
export function sanitizeTerminalText(text: string): string {
	// Strip all control characters except ANSI escape sequences
	const sanitized = text
		.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "") // keep ANSI escapes
		.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, ""); // strip control chars
	// Collapse whitespace and trim
	return sanitized.replace(/\s+/g, " ").trim();
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Composite line — overlay one line into another at a given column
 * ════════════════════════════════════════════════════════════════════════════ */

function compositeTuiLineAsciiSingle(
	baseLine: string,
	overlayLine: string,
	startCol: number,
	overlayWidth: number,
	totalWidth: number,
): string | null {
	let before = "";
	let beforeWidth = 0;
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
			if (beforeWidth === col) before += code;
			if (col >= afterStart && !afterDone) after += code;
			i = j;
			continue;
		}
		const code = ch.charCodeAt(0);
		if (code !== 0x09 && (code < 0x20 || code > 0x7e)) return null;
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

/**
 * Place `overlayLine` into `baseLine` at column `startCol`, padding both
 * sides with spaces out to `totalWidth`. Resets styles at boundaries.
 */
export function compositeTuiLine(
	baseLine: string,
	overlayLine: string,
	startCol: number,
	overlayWidth: number,
	totalWidth: number,
): string {
	const fast = compositeTuiLineAsciiSingle(
		baseLine,
		overlayLine,
		startCol,
		overlayWidth,
		totalWidth,
	);
	if (fast !== null) return fast;

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

/* ════════════════════════════════════════════════════════════════════════════
 *  Token formatting — shared across widgets
 * ════════════════════════════════════════════════════════════════════════════ */

/* ════════════════════════════════════════════════════════════════════════════
 *  Color text helpers — wrap text with ANSI color codes
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * Wrap `text` with an Ink-compatible color string.
 * Usage: colorText(theme.fg("accent"), "hello") → "\x1b[32mhello\x1b[0m"
 */
export function colorText(color: string | undefined, text: string): string {
	if (!color) return text;
	return `${color}${text}${RESET}`;
}

export function tokenStr(tokens: number): string {
	if (tokens >= 1_000_000) {
		const v = tokens / 1_000_000;
		return v % 1 === 0 ? `${Math.round(v)}M` : `${v.toFixed(1)}M`;
	}
	if (tokens >= 1000) {
		const v = tokens / 1000;
		return v % 1 === 0 ? `${Math.round(v)}k` : `${v.toFixed(1)}k`;
	}
	return String(tokens);
}
