// ── Input bar component ────────────────────────────────────────────────────────
// Full-featured single-line text input — undo/redo, kill ring, word nav,
// bracketed paste, history, grapheme-aware cursor. Mirrors pi TUI's input.

import type { Component, Focusable } from "../tui-core.ts";
import { CURSOR_MARKER, visibleWidth } from "../tui-core.ts";
import type { UndoStack } from "../undo-stack.ts";
import type { KillRing } from "../kill-ring.ts";
import { getGraphemeSegmenter } from "../utils.ts";
import { findWordBackward, findWordForward } from "../word-navigation.ts";

const segmenter = getGraphemeSegmenter();

// ── Input bar ─────────────────────────────────────────────────────────────────

export interface InputBarOptions {
    prompt?: string;
    placeholder?: string;
}

export class InputBar implements Component, Focusable {
    public focused = false;

    // State
    private value = "";
    private cursor = 0; // grapheme index
    private history: string[] = [];
    private historyIndex: number | null = null;
    private historyUnsaved: string | null = null;
    private _prompt = "\x1b[1m\x1b[38;5;111m› \x1b[0m";
    private _placeholder = "Type a message…";
    private maxHistory = 100;

    // Kill ring & undo (injected by parent or default instances)
    private _killRing: KillRing | null = null;
    private _undoStack: UndoStack<{ value: string; cursor: number }> | null =
        null;

    // Bracketed paste
    private pasteBuffer = "";
    private isInPaste = false;

    // Rendering cache
    private cachedLines: string[] | null = null;
    private cachedWidth = -1;

    // ── Callbacks ────────────────────────────────────────────────────────────

    onSubmit?: (text: string) => void;
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
        return this._prompt;
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

    // ── Input handling ─────────────────────────────────────────────────────

    handleInput(data: string): void {
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
                this._handlePaste(pasteText);
                this.isInPaste = false;
                this.pasteBuffer = "";
                const remaining = this.pasteBuffer.substring(endIdx + 6);
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

        // ── Ctrl+K — delete to line end ──────────────────────────────────────
        if (data === "\x0b") {
            this._deleteToLineEnd();
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

        // ── Up arrow — history prev ──────────────────────────────────────────
        if (data === "\x1b[A" || data === "\x1bOA") {
            this.historyPrev();
            return;
        }

        // ── Down arrow — history next ────────────────────────────────────────
        if (data === "\x1b[B" || data === "\x1bOB") {
            this.historyNext();
            return;
        }

        // ── Escape ───────────────────────────────────────────────────────────
        if (data === "\x1b") {
            this._cancel();
            return;
        }

        // ── Ctrl+C — cancel ──────────────────────────────────────────────────
        if (data === "\x03") {
            this._cancel();
            return;
        }

        // ── Enter — submit ───────────────────────────────────────────────────
        if (data === "\r" || data === "\n") {
            const text = this.value.trim();
            if (text || this.value.length > 0) {
                this.pushHistory(text || this.value);
                const textToSubmit = text || this.value;
                this.value = "";
                this.cursor = 0;
                this._invalidate();
                if (textToSubmit) {
                    this.onSubmit?.(textToSubmit);
                }
            }
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
        return [...segmenter.segment(text)].length;
    }

    private _graphemeSlice(text: string, from: number, to?: number): string {
        const segs = [...segmenter.segment(text)];
        const end = to !== undefined ? to : segs.length;
        return segs
            .slice(from, end)
            .map((s) => s.segment)
            .join("");
    }

    private _insert(ch: string): void {
        this._pushUndo();
        const segs = [...segmenter.segment(this.value)];
        const newSegs = [
            ...segs.slice(0, this.cursor),
            ...[...segmenter.segment(ch)],
            ...segs.slice(this.cursor),
        ];
        this.value = newSegs.map((s) => s.segment).join("");
        this.cursor += this._graphemeCount(ch);
        this._invalidate();
    }

    private _handleBackspace(): void {
        if (this.cursor === 0) return;
        this._pushUndo();
        const segs = [...segmenter.segment(this.value)];
        segs.splice(this.cursor - 1, 1);
        this.value = segs.map((s) => s.segment).join("");
        this.cursor -= 1;
        this._invalidate();
    }

    private _handleForwardDelete(): void {
        const totalGraphemes = this._graphemeCount(this.value);
        if (this.cursor >= totalGraphemes) return;
        this._pushUndo();
        const segs = [...segmenter.segment(this.value)];
        segs.splice(this.cursor, 1);
        this.value = segs.map((s) => s.segment).join("");
        this._invalidate();
    }

    private _deleteWordBackward(): void {
        if (this.cursor === 0) return;
        this._pushUndo();
        const oldCursor = this.cursor;
        this.cursor = findWordBackward(this.value, this.cursor);
        const deleted = this._graphemeSlice(this.value, this.cursor, oldCursor);
        this._killRing?.push(deleted, { prepend: true, accumulate: true });
        const segs = [...segmenter.segment(this.value)];
        segs.splice(this.cursor, oldCursor - this.cursor);
        this.value = segs.map((s) => s.segment).join("");
        this._invalidate();
    }

    private _deleteToLineStart(): void {
        if (this.cursor === 0) return;
        this._pushUndo();
        const deleted = this._graphemeSlice(this.value, 0, this.cursor);
        this._killRing?.push(deleted, { prepend: true, accumulate: true });
        const segs = [...segmenter.segment(this.value)];
        segs.splice(0, this.cursor);
        this.value = segs.map((s) => s.segment).join("");
        this.cursor = 0;
        this._invalidate();
    }

    private _deleteToLineEnd(): void {
        if (this.cursor >= this._graphemeCount(this.value)) return;
        this._pushUndo();
        const deleted = this._graphemeSlice(this.value, this.cursor);
        this._killRing?.push(deleted, { accumulate: true });
        const segs = [...segmenter.segment(this.value)];
        segs.splice(this.cursor);
        this.value = segs.map((s) => s.segment).join("");
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
        const cleanText = pastedText
            .replace(/\r\n/g, " ")
            .replace(/\r/g, " ")
            .replace(/\n/g, " ")
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
        const prompt = this._prompt;
        const promptWidth = visibleWidth(prompt);
        const contentWidth = Math.max(1, width - promptWidth - 1);
        const displayText = this.value || this._placeholder;
        const isPlaceholder = !this.value;

        // Grapheme segments for cursor positioning
        const allSegments = [...segmenter.segment(displayText)].map(
            (s) => s.segment,
        );
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
            cursorInViewport < segments.length
                ? segments[cursorInViewport]
                : " ";
        const afterCursor = segments.slice(cursorInViewport + 1).join("");

        // Cursor with inverse video (only when focused)
        const cursorChar = this.focused
            ? `${CURSOR_MARKER}\x1b[7m${atCursor}\x1b[27m`
            : atCursor;

        // Build the line
        const color = isPlaceholder ? "\x1b[38;5;244m" : "\x1b[38;5;159m";
        const rawLine =
            prompt +
            (viewport.leftClipped ? "\x1b[38;5;244m‹\x1b[0m" : "") +
            color +
            beforeCursor +
            cursorChar +
            afterCursor +
            "\x1b[0m" +
            (viewport.rightClipped ? "\x1b[38;5;244m›\x1b[0m" : "");

        // Calculate visible width (strip CURSOR_MARKER for measurement)
        const cleanLine = rawLine.replace(CURSOR_MARKER, "");
        const lineWidth = visibleWidth(cleanLine);
        const finalLine = rawLine + " ".repeat(Math.max(0, width - lineWidth));

        this.cachedLines = [finalLine];
        return this.cachedLines;
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
        let end = Math.min(
            segments.length,
            Math.max(cursor + 1, start + target),
        );

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
