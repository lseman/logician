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

// Simple visible width calculator (handles ANSI escape codes)
export function visibleWidth(text: string): number {
	return stringWidth(
		text
			.replace(/\x1b\[[0-?]*[ -\/]*[@-~]/g, "")
			.replace(/\x1b[\]_][\s\S]*?(?:\x07|\x1b\\)/g, ""),
	);
}

const graphemeSegmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });

function graphemeAt(text: string, offset: number): string {
	const segment = graphemeSegmenter.segment(text.slice(offset))[Symbol.iterator]().next();
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

// Place `overlayLine` (already clamped to `overlayWidth`) into `baseLine` at
// column `startCol`, padding both sides with spaces out to `totalWidth`.
// Resets styles at each boundary so a colored child never bleeds into its
// neighbor — used by HStack to compose siblings left-to-right on one row.
export function compositeTuiLine(
	baseLine: string,
	overlayLine: string,
	startCol: number,
	overlayWidth: number,
	totalWidth: number,
): string {
	const before = clampLineToWidth(baseLine, startCol);
	const beforeWidth = visibleWidth(before);
	const beforePad = " ".repeat(Math.max(0, startCol - beforeWidth));
	const overlay = clampLineToWidth(overlayLine, overlayWidth);
	const overlayPad = " ".repeat(Math.max(0, overlayWidth - visibleWidth(overlay)));
	const afterStart = startCol + overlayWidth;
	const afterWidth = Math.max(0, totalWidth - afterStart);
	const after = afterWidth > 0 ? clampLineToWidth(skipColumns(baseLine, afterStart), afterWidth) : "";
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
				while (j < text.length && !(text.charCodeAt(j) >= 0x40 && text.charCodeAt(j) <= 0x7e)) j++;
				i = j + 1;
				continue;
			}
			if (next === "]" || next === "_") {
				let j = i + 2;
				while (j < text.length && text[j] !== "\x07" && !(text[j] === "\x1b" && text[j + 1] === "\\")) j++;
				i = text[j] === "\x07" ? j + 1 : j + 2;
				continue;
			}
			i++;
			continue;
		}
		const codePoint = text.codePointAt(i);
		const character = codePoint === undefined ? ch : String.fromCodePoint(codePoint);
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
