// ── Input bar component ────────────────────────────────────────────────────────
// Full-featured single-line text input — undo/redo, kill ring, word nav,
// bracketed paste, history, grapheme-aware cursor. Mirrors pi TUI's input.

import {
	BOLD,
	type Component,
	CURSOR_MARKER,
	type Focusable,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import { getGraphemeSegmenter } from "../terminal/utils.ts";
import type { KillRing } from "./kill-ring.ts";
import type { UndoStack } from "./undo-stack.ts";
import { findWordBackward, findWordForward } from "./word-navigation.ts";

const segmenter = getGraphemeSegmenter();

function isArrow(data: string, direction: "A" | "B" | "C" | "D"): boolean {
	return (
		data === `\x1bO${direction}` ||
		new RegExp(`^\\x1b\\[(?:1(?:;\\d+)?)?${direction}$`).test(data)
	);
}

// ── Input bar ─────────────────────────────────────────────────────────────────

export interface InputBarOptions {
	prompt?: string;
	placeholder?: string;
}

export type InputSubmitIntent = "default" | "steer-now";

export class InputBar implements Component, Focusable {
	public focused = false;

	// State
	private _value = "";
	private cursor = 0; // grapheme index
	// Segmentation cache: re-running Intl.Segmenter over the whole buffer on
	// every keystroke is O(buffer length) per key, which compounds once a large
	// paste sits in the composer. Cached here and invalidated only when `value`
	// is reassigned via `_setValue`/`this.value =`.
	private _segsCache: string[] | null = null;
	private _segsCacheValue: string | null = null;

	private get value(): string {
		return this._value;
	}

	private set value(text: string) {
		this._value = text;
	}

	private _segments(text: string = this._value): string[] {
		if (text === this._segsCacheValue && this._segsCache !== null) {
			return this._segsCache;
		}
		const segs = [...segmenter.segment(text)].map(s => s.segment);
		if (text === this._value) {
			this._segsCache = segs;
			this._segsCacheValue = text;
		}
		return segs;
	}
	private history: string[] = [];
	private historyIndex: number | null = null;
	private historyUnsaved: string | null = null;
	private _prompt: string | undefined;
	private get _promptResolved(): string {
		if (this._prompt === undefined) {
			this._prompt = `  ${theme.fg("prompt", "")}${BOLD}› ${RESET}`;
		}
		return this._prompt;
	}
	private _placeholder = "Ask Logician…";
	private maxHistory = 100;

	// Kill ring & undo (injected by parent or default instances)
	private _killRing: KillRing | null = null;
	private _undoStack: UndoStack<{ value: string; cursor: number }> | null =
		null;

	// Bracketed paste
	private pasteBuffer = "";
	private isInPaste = false;
	private escapeArmed = false;

	// Rendering cache
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	// ── Callbacks ────────────────────────────────────────────────────────────

	onSubmit?: (text: string, intent: InputSubmitIntent) => void;
	onCancel?: () => void;
	onChange?: (text: string) => void;

	// ── Injected dependencies ────────────────────────────────────────────────

	setKillRing(kr: KillRing): void {
		this._killRing = kr;
	}

	setUndoStack(us: UndoStack<{ value: string; cursor: number }>): void {
		this._undoStack = us;
	}

	// ── Public API ─────────────────────────────────────────────────────────

	get valueText(): string {
		return this.value;
	}

	set valueText(text: string) {
		this._pushUndo();
		this.value = text;
		this.cursor = this._graphemeCount(text);
		this._invalidate();
	}

	get prompt(): string {
		return this._promptResolved;
	}

	set prompt(p: string) {
		this._prompt = p;
		this._invalidate();
	}

	get placeholder(): string {
		return this._placeholder;
	}

	set placeholder(p: string) {
		this._placeholder = p;
		this._invalidate();
	}

	// ── @-mention detection ────────────────────────────────────────────────

	/**
	 * The "@partial" token immediately before the cursor, if any (query text
	 * only, without the "@"). Returns null when the cursor isn't inside an
	 * active mention — e.g. after whitespace or when there's no "@" on the
	 * current line segment.
	 */
	getActiveMentionQuery(): string | null {
		const segs = this._segments();
		const before = segs.slice(0, this.cursor).join("");
		const at = before.lastIndexOf("@");
		if (at === -1) return null;
		const token = before.slice(at + 1);
		if (/\s/.test(token)) return null;
		return token;
	}

	/** Replace the active "@partial" token at the cursor with "@path ". */
	insertMention(path: string): void {
		const segs = this._segments();
		const before = segs.slice(0, this.cursor).join("");
		const after = segs.slice(this.cursor).join("");
		const at = before.lastIndexOf("@");
		if (at === -1) return;
		this._pushUndo();
		const newBefore = `${before.slice(0, at)}@${path} `;
		this.value = newBefore + after;
		this.cursor = this._graphemeCount(newBefore);
		this._invalidate();
	}

	// ── History ────────────────────────────────────────────────────────────

	pushHistory(text: string): void {
		if (
			this.history.length === 0 ||
			this.history[this.history.length - 1] !== text
		) {
			this.history.push(text);
			if (this.history.length > this.maxHistory) this.history.shift();
		}
		this.historyIndex = null;
		this.historyUnsaved = null;
	}

	historyPrev(): void {
		if (this.history.length === 0) return;
		if (this.historyIndex === null) {
			this.historyUnsaved = this.value;
		}
		this.historyIndex =
			this.historyIndex === null
				? this.history.length - 1
				: Math.max(0, this.historyIndex - 1);
		this.value = this.history[this.historyIndex] || "";
		this.cursor = this._graphemeCount(this.value);
		this._invalidate();
	}

	historyNext(): void {
		if (this.historyIndex === null) return;
		if (this.historyIndex === this.history.length - 1) {
			this.value = this.historyUnsaved || "";
			this.cursor = this._graphemeCount(this.value);
			this.historyIndex = null;
			this.historyUnsaved = null;
		} else {
			this.historyIndex++;
			this.value = this.history[this.historyIndex] || "";
			this.cursor = this._graphemeCount(this.value);
		}
		this._invalidate();
	}

	clearHistory(): void {
		this.historyIndex = null;
		this.historyUnsaved = null;
	}

	/** Submit the current composer value with an explicit delivery intent. */
	submit(intent: InputSubmitIntent = "default"): boolean {
		const text = this.value.trim();
		if (!text && this.value.length === 0) return false;
		this.pushHistory(text || this.value);
		const textToSubmit = text || this.value;
		this.value = "";
		this.cursor = 0;
		this._invalidate();
		if (!textToSubmit) return false;
		this.onSubmit?.(textToSubmit, intent);
		return true;
	}

	// ── Input handling ─────────────────────────────────────────────────────

	handleInput(data: string): void {
		if (data !== "\x1b") this.escapeArmed = false;

		// Some terminals batch multiple navigation keys into one stdin chunk.
		// Replay a pure arrow-key batch one key at a time instead of treating the
		// entire chunk as unknown input.
		const arrowBatch = data.match(/\x1b(?:O[ABCD]|\[(?:1(?:;\d+)?)?[ABCD])/g);
		if (arrowBatch && arrowBatch.length > 1 && arrowBatch.join("") === data) {
			for (const key of arrowBatch) this.handleInput(key);
			return;
		}

		// Fast typing, SSH, and tmux/mosh links routinely coalesce several plain
		// keystrokes into one stdin chunk. Below, only single-character chunks
		// reach the printable-character branch, so an unhandled multi-char burst
		// would otherwise be silently dropped instead of typed. Bracketed paste
		// (\x1b[200~) and any other escape-sequence chunk are handled by their own
		// branches below and must not be split here.
		if (data.length > 1 && !data.includes("\x1b") && !this.isInPaste) {
			for (const ch of data) this.handleInput(ch);
			return;
		}

		// ── Bracketed paste ──────────────────────────────────────────────────
		if (data.includes("\x1b[200~")) {
			this.isInPaste = true;
			this.pasteBuffer = "";
			data = data.replace("\x1b[200~", "");
			if (!data) return;
		}

		if (this.isInPaste) {
			this.pasteBuffer += data;
			const endIdx = this.pasteBuffer.indexOf("\x1b[201~");
			if (endIdx !== -1) {
				const pasteText = this.pasteBuffer.substring(0, endIdx);
				const remaining = this.pasteBuffer.substring(endIdx + 6);
				this._handlePaste(pasteText);
				this.isInPaste = false;
				this.pasteBuffer = "";
				if (remaining) this.handleInput(remaining);
				return;
			}
			return;
		}

		// ── Ctrl+U — delete to line start ────────────────────────────────────
		if (data === "\x15") {
			this._deleteToLineStart();
			return;
		}

		// ── Ctrl+W — delete word before cursor ───────────────────────────────
		if (data === "\x17") {
			this._deleteWordBackward();
			return;
		}

		// ── Ctrl+Y — yank from kill ring ─────────────────────────────────────
		if (data === "\x19") {
			this._yank();
			return;
		}

		// ── Ctrl+Z — undo ────────────────────────────────────────────────────
		if (data === "\x1a") {
			this.undo();
			return;
		}

		// ── Ctrl+R — redo ────────────────────────────────────────────────────
		if (data === "\x1e") {
			this.redo();
			return;
		}

		// ── Ctrl+Left / Alt+b — word left ────────────────────────────────────
		if (data === "\x1b[1;5D" || data === "\x1bb") {
			this.moveWordBackward();
			return;
		}

		// ── Ctrl+Right / Alt+f — word right ──────────────────────────────────
		if (data === "\x1b[1;5C" || data === "\x1bf") {
			this.moveWordForward();
			return;
		}

		// ── Left arrow — character left ──────────────────────────────────────
		if (isArrow(data, "D")) {
			if (this.cursor > 0) {
				this.cursor--;
				this._invalidate();
			}
			return;
		}

		// ── Right arrow — character right ────────────────────────────────────
		if (isArrow(data, "C")) {
			const totalGraphemes = this._graphemeCount(this.value);
			if (this.cursor < totalGraphemes) {
				this.cursor++;
				this._invalidate();
			}
			return;
		}

		// ── Up arrow — history prev ──────────────────────────────────────────
		if (isArrow(data, "A")) {
			if (!this._moveToAdjacentLine(-1)) this.historyPrev();
			return;
		}

		// ── Down arrow — history next ────────────────────────────────────────
		if (isArrow(data, "B")) {
			if (!this._moveToAdjacentLine(1)) this.historyNext();
			return;
		}

		// ── Escape ───────────────────────────────────────────────────────────
		if (data === "\x1b") {
			this._handleEscape();
			return;
		}

		// ── Ctrl+C — cancel ──────────────────────────────────────────────────
		if (data === "\x03") {
			this._cancel();
			return;
		}

		// ── Enter — submit ───────────────────────────────────────────────────
		if (data === "\r" || data === "\n") {
			this.submit();
			return;
		}

		// ── Backspace ────────────────────────────────────────────────────────
		if (data === "\x7f" || data === "\x08") {
			this._handleBackspace();
			return;
		}

		// ── Delete ───────────────────────────────────────────────────────────
		if (data === "\x1b[3~") {
			this._handleForwardDelete();
			return;
		}

		// ── Home ─────────────────────────────────────────────────────────────
		if (data === "\x1b[H" || data === "\x1b[1~") {
			this.cursor = 0;
			this._invalidate();
			return;
		}

		// ── End ──────────────────────────────────────────────────────────────
		if (data === "\x1b[F" || data === "\x1b[4~") {
			this.cursor = this._graphemeCount(this.value);
			this._invalidate();
			return;
		}

		// ── Tab — insert spaces ──────────────────────────────────────────────
		if (data === "\t") {
			this._insert("    ");
			return;
		}

		// ── Printable character ──────────────────────────────────────────────
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c < 0x20 || c === 0x7f) return;
			this._insert(data);
		}
	}

	// ── Private helpers ────────────────────────────────────────────────────

	private _graphemeCount(text: string): number {
		return this._segments(text).length;
	}

	private _graphemeSlice(text: string, from: number, to?: number): string {
		const segs = this._segments(text);
		const end = to !== undefined ? to : segs.length;
		return segs.slice(from, end).join("");
	}

	private _insert(ch: string): void {
		this._pushUndo();
		const segs = this._segments();
		const newSegs = [
			...segs.slice(0, this.cursor),
			...this._segments(ch),
			...segs.slice(this.cursor),
		];
		this.value = newSegs.join("");
		this.cursor += this._graphemeCount(ch);
		this._invalidate();
	}

	private _handleBackspace(): void {
		if (this.cursor === 0) return;
		this._pushUndo();
		const segs = this._segments().slice();
		segs.splice(this.cursor - 1, 1);
		this.value = segs.join("");
		this.cursor -= 1;
		this._invalidate();
	}

	private _handleForwardDelete(): void {
		const totalGraphemes = this._graphemeCount(this.value);
		if (this.cursor >= totalGraphemes) return;
		this._pushUndo();
		const segs = this._segments().slice();
		segs.splice(this.cursor, 1);
		this.value = segs.join("");
		this._invalidate();
	}

	/** Move vertically through pasted/multiline input while preserving column. */
	private _moveToAdjacentLine(direction: -1 | 1): boolean {
		const segments = this._segments();
		const beforeCursor = segments.slice(0, this.cursor);
		const previousBreak = beforeCursor.lastIndexOf("\n");
		const lineStart = previousBreak + 1;
		const column = this.cursor - lineStart;

		if (direction < 0) {
			if (lineStart === 0) return false;
			const previousLineEnd = lineStart - 1;
			const previousLineStart =
				segments.slice(0, previousLineEnd).lastIndexOf("\n") + 1;
			this.cursor = Math.min(previousLineStart + column, previousLineEnd);
		} else {
			const nextBreakOffset = segments.slice(this.cursor).indexOf("\n");
			if (nextBreakOffset < 0) return false;
			const nextLineStart = this.cursor + nextBreakOffset + 1;
			const followingBreakOffset = segments.slice(nextLineStart).indexOf("\n");
			const nextLineEnd =
				followingBreakOffset < 0
					? segments.length
					: nextLineStart + followingBreakOffset;
			this.cursor = Math.min(nextLineStart + column, nextLineEnd);
		}

		this._invalidate();
		return true;
	}

	private _deleteWordBackward(): void {
		if (this.cursor === 0) return;
		this._pushUndo();
		const oldCursor = this.cursor;
		this.cursor = findWordBackward(this.value, this.cursor);
		const deleted = this._graphemeSlice(this.value, this.cursor, oldCursor);
		this._killRing?.push(deleted, { prepend: true, accumulate: true });
		const segs = this._segments().slice();
		segs.splice(this.cursor, oldCursor - this.cursor);
		this.value = segs.join("");
		this._invalidate();
	}

	private _deleteToLineStart(): void {
		if (this.cursor === 0) return;
		this._pushUndo();
		const deleted = this._graphemeSlice(this.value, 0, this.cursor);
		this._killRing?.push(deleted, { prepend: true, accumulate: true });
		const segs = this._segments().slice();
		segs.splice(0, this.cursor);
		this.value = segs.join("");
		this.cursor = 0;
		this._invalidate();
	}

	private _yank(): void {
		const text = this._killRing?.peek();
		if (!text) return;
		this._pushUndo();
		this._insert(text);
	}

	private _handlePaste(pastedText: string): void {
		this._pushUndo();
		// Preserve newlines (multi-line paste like Pi). Only normalize tabs.
		const cleanText = pastedText
			.replace(/\r\n/g, "\n")
			.replace(/\r/g, "\n")
			.replace(/\t/g, "    ");
		this._insert(cleanText);
	}

	private _cancel(): void {
		if (this.value.length > 0) {
			this.value = "";
			this.cursor = 0;
			this._invalidate();
		}
		if (this.historyIndex !== null) {
			this.historyIndex = null;
			this.historyUnsaved = null;
			this._invalidate();
		}
		this.onCancel?.();
	}

	private _handleEscape(): void {
		if (this.escapeArmed) {
			this.escapeArmed = false;
			this.onCancel?.();
			return;
		}
		this.escapeArmed = true;
		if (this.value.length > 0) {
			this.value = "";
			this.cursor = 0;
			this._invalidate();
		}
		if (this.historyIndex !== null) {
			this.historyIndex = null;
			this.historyUnsaved = null;
			this._invalidate();
		}
	}

	// ── Undo / Redo ──────────────────────────────────────────────────────

	private _pushUndo(): void {
		if (this._undoStack) {
			this._undoStack.push({ value: this.value, cursor: this.cursor });
		}
	}

	undo(): void {
		if (!this._undoStack) return;
		const snap = this._undoStack.pop();
		if (!snap) return;
		this.value = snap.value;
		this.cursor = snap.cursor;
		this._invalidate();
	}

	redo(): void {
		if (!this._undoStack) return;
		const snap = this._undoStack.redo();
		if (!snap) return;
		this.value = snap.value;
		this.cursor = snap.cursor;
		this._invalidate();
	}

	hasUndo(): boolean {
		return this._undoStack?.hasPast() ?? false;
	}

	hasRedo(): boolean {
		return this._undoStack?.hasFuture() ?? false;
	}

	// ── Cursor movement ──────────────────────────────────────────────────

	moveWordBackward(): void {
		this.cursor = findWordBackward(this.value, this.cursor);
		this._invalidate();
	}

	moveWordForward(): void {
		this.cursor = findWordForward(this.value, this.cursor);
		this._invalidate();
	}

	// ── Rendering ────────────────────────────────────────────────────────

	_invalidate(): void {
		this.cachedLines = null;
		this.onChange?.(this.value);
	}

	invalidate(): void {
		this._invalidate();
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.cachedWidth = width;
		const prompt = this._promptResolved;
		const promptWidth = visibleWidth(prompt);
		const contentWidth = Math.max(1, width - promptWidth - 2);
		const displayText = this.value || this._placeholder;
		const isPlaceholder = !this.value;
		const logicalLines = displayText.split("\n");
		if (!isPlaceholder && logicalLines.length > 1) {
			this.cachedLines = this._renderMultiline(width, logicalLines);
			return this.cachedLines;
		}

		// Grapheme segments for cursor positioning
		const allSegments = isPlaceholder
			? this._segments(displayText)
			: this._segments();
		const graphCursor = Math.min(this.cursor, allSegments.length);
		const viewport = this._inputViewport(
			allSegments,
			graphCursor,
			contentWidth,
			isPlaceholder,
		);
		const segments = viewport.segments;
		const cursorInViewport = Math.max(0, graphCursor - viewport.start);

		// Build rendered segments with cursor
		const beforeCursor = segments.slice(0, cursorInViewport).join("");
		const atCursor =
			cursorInViewport < segments.length ? segments[cursorInViewport] : " ";
		const afterCursor = segments.slice(cursorInViewport + 1).join("");

		// Mark the edit position so the renderer can park the hardware cursor
		// there (consumed + stripped in tui-core). Inverse video draws the
		// visible cursor only when focused and not showing the placeholder, so
		// the prompt glyph never appears highlighted on an empty field.
		const cursorChar =
			isPlaceholder || !this.focused
				? `${CURSOR_MARKER}${atCursor}`
				: `${CURSOR_MARKER}\x1b[7m${atCursor}\x1b[27m`;

		// Build the line
		const color = isPlaceholder
			? theme.fg("inputPlaceholder", "")
			: theme.fg("inputText", "");
		const rawLine =
			prompt +
			(viewport.leftClipped
				? `${theme.fg("inputPlaceholder", "")}‹${RESET}`
				: "") +
			color +
			beforeCursor +
			cursorChar +
			afterCursor +
			"\x1b[0m" +
			(viewport.rightClipped
				? `${theme.fg("inputPlaceholder", "")}›${RESET}`
				: "");

		// Calculate visible width (strip CURSOR_MARKER for measurement)
		const cleanLine = rawLine.replace(CURSOR_MARKER, "");
		const lineWidth = visibleWidth(cleanLine);
		const finalLine = rawLine + " ".repeat(Math.max(0, width - lineWidth));

		// Give the composer a quiet visual boundary on normal-width terminals.
		// Narrow terminals keep the compact one-line editor.
		const header = width >= 36 ? this._renderComposerHeader(width) : null;
		this.cachedLines = header ? [header, finalLine] : [finalLine];
		return this.cachedLines;
	}

	/**
	 * Render a stable, bounded window around the active logical line. Long
	 * prompts remain readable without letting the composer consume the whole
	 * terminal; each logical line still gets the familiar horizontal viewport.
	 */
	private _renderMultiline(width: number, logicalLines: string[]): string[] {
		const header = width >= 36 ? [this._renderComposerHeader(width)] : [];
		const maxVisibleLines = width >= 52 ? 5 : 3;
		const beforeCursor = this._graphemeSlice(this.value, 0, this.cursor);
		const cursorLine = Math.min(
			logicalLines.length - 1,
			beforeCursor.split("\n").length - 1,
		);
		const cursorColumn = this._graphemeCount(
			beforeCursor.split("\n").at(-1) ?? "",
		);
		let start = Math.max(0, cursorLine - maxVisibleLines + 1);
		start = Math.min(start, Math.max(0, logicalLines.length - maxVisibleLines));
		const end = Math.min(logicalLines.length, start + maxVisibleLines);
		const hiddenAbove = start > 0;
		const hiddenBelow = end < logicalLines.length;

		const prompt = this._promptResolved;
		const promptWidth = visibleWidth(prompt);
		const continuation = " ".repeat(promptWidth);
		const contentWidth = Math.max(1, width - promptWidth - 1);
		const rows: string[] = [];

		for (let lineIndex = start; lineIndex < end; lineIndex++) {
			const lineSegments = [
				...segmenter.segment(logicalLines[lineIndex] ?? ""),
			].map(item => item.segment);
			const isCursorLine = lineIndex === cursorLine;
			const lineCursor = isCursorLine
				? Math.min(cursorColumn, lineSegments.length)
				: lineSegments.length;
			const viewport = this._inputViewport(
				lineSegments,
				lineCursor,
				contentWidth,
				false,
			);
			const cursorInViewport = Math.max(0, lineCursor - viewport.start);
			const visible = [...viewport.segments];
			let body: string;
			if (isCursorLine) {
				const before = visible.slice(0, cursorInViewport).join("");
				const at =
					cursorInViewport < visible.length ? visible[cursorInViewport] : " ";
				const after = visible.slice(cursorInViewport + 1).join("");
				const cursor = this.focused
					? `${CURSOR_MARKER}\x1b[7m${at}\x1b[27m`
					: `${CURSOR_MARKER}${at}`;
				body = before + cursor + after;
			} else {
				body = visible.join("");
			}

			const prefix = lineIndex === 0 ? prompt : continuation;
			const aboveMarker = hiddenAbove && lineIndex === start ? "↑" : "";
			const belowMarker = hiddenBelow && lineIndex === end - 1 ? "↓" : "";
			const leftMarker = viewport.leftClipped ? "‹" : aboveMarker;
			const rightMarker = viewport.rightClipped ? "›" : belowMarker;
			const raw =
				prefix +
				theme.fg("inputPlaceholder", leftMarker) +
				theme.fg("inputText", body) +
				theme.fg("inputPlaceholder", rightMarker) +
				RESET;
			const clean = raw.replace(CURSOR_MARKER, "");
			rows.push(raw + " ".repeat(Math.max(0, width - visibleWidth(clean))));
		}

		return [...header, ...rows];
	}

	private _renderComposerHeader(width: number): string {
		const hintText =
			width >= 72
				? this.value
					? "Enter send  ·  Ctrl+Enter steer now  ·  Esc clear  ·  Ctrl+O tools"
					: "/ Enter commands  ·  Ctrl+Enter steer now  ·  Ctrl+O tools"
				: width >= 52
					? this.value
						? "Enter send  ·  Ctrl+Enter steer now  ·  Esc clear"
						: "/ commands  ·  Ctrl+Enter steer now"
					: "Enter send  ·  Ctrl+Enter now";
		const hint = ` ${theme.fg("muted", hintText)} `;
		const hintWidth = visibleWidth(hint);
		const ruleWidth = Math.max(1, width - hintWidth);
		return theme.fg("borderMuted", "─".repeat(ruleWidth)) + hint;
	}

	private _inputViewport(
		segments: string[],
		cursor: number,
		width: number,
		isPlaceholder: boolean,
	): {
		segments: string[];
		start: number;
		leftClipped: boolean;
		rightClipped: boolean;
	} {
		if (isPlaceholder || visibleWidth(segments.join("")) <= width) {
			return {
				segments,
				start: 0,
				leftClipped: false,
				rightClipped: false,
			};
		}

		const target = Math.max(1, width - 2);
		let start = Math.max(0, cursor - target + 1);
		let end = Math.min(segments.length, Math.max(cursor + 1, start + target));

		while (
			visibleWidth(segments.slice(start, end).join("")) > target &&
			start < cursor
		) {
			start++;
		}
		while (
			end < segments.length &&
			visibleWidth(segments.slice(start, end + 1).join("")) <= target
		) {
			end++;
		}

		return {
			segments: segments.slice(start, end),
			start,
			leftClipped: start > 0,
			rightClipped: end < segments.length,
		};
	}
}
