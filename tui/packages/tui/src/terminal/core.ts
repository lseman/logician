// ── Minimal TUI core ──────────────────────────────────────────────────────────
// Differential rendering engine — minimal, no external deps

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

export interface RendererMetrics {
	bytesWritten: number;
	changedCells: number;
	cursorMoves: number;
	diffTimeMs: number;
	dirtyRegion: { top: number; bottom: number } | null;
	dirtyRows: number;
	frameTimeMs: number;
	layoutTimeMs: number;
	writeTimeMs: number;
}

const EMPTY_RENDERER_METRICS: RendererMetrics = {
	bytesWritten: 0,
	changedCells: 0,
	cursorMoves: 0,
	diffTimeMs: 0,
	dirtyRegion: null,
	dirtyRows: 0,
	frameTimeMs: 0,
	layoutTimeMs: 0,
	writeTimeMs: 0,
};

/**
 * Translate Kitty CSI-u Ctrl+letter reports back to the C0 bytes consumed by
 * existing keybindings. Ctrl+I and Ctrl+M stay encoded so they remain
 * distinguishable from Tab and Enter and can reach their dedicated bindings.
 */
export function normalizeKeyboardInput(data: string): string {
	return data
		// biome-ignore lint/suspicious/noControlCharactersInRegex: terminal CSI escape sequence
		.replace(/\x1b\[27(?:;1)?u/g, "\x1b")
		.replace(/\x1b\[(\d+);([56])u/g, (sequence, codepointText: string) => {
			const codepoint = Number(codepointText);
			const lowerCodepoint =
				codepoint >= 65 && codepoint <= 90 ? codepoint + 32 : codepoint;
			if (lowerCodepoint === 105 || lowerCodepoint === 109) return sequence;
			if (lowerCodepoint < 96 || lowerCodepoint > 127) return sequence;
			return String.fromCharCode(lowerCodepoint & 0x1f);
		});
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
	let width = 0;
	let i = 0;
	while (i < text.length) {
		const ch = text[i];
		if (ch === "\x1b") {
			const next = text[i + 1];
			// CSI: ESC [ ... final byte (0x40-0x7E)
			if (next === "[") {
				i += 2;
				while (i < text.length) {
					const fc = text.charCodeAt(i);
					if (fc >= 0x40 && fc <= 0x7e) break;
					i++;
				}
				i++;
				continue;
			}
			// OSC: ESC ] ... BEL (0x07) or ST (ESC \)
			if (next === "]") {
				i += 2;
				while (i < text.length) {
					if (text[i] === "\x07") break;
					if (text[i] === "\x1b" && text[i + 1] === "\\") {
						i++;
						break;
					}
					i++;
				}
				i++;
				if (text[i - 1] === "\x1b") i++; // skip ST backslash
				continue;
			}
			// APC: ESC _ ... BEL (0x07) or ST (ESC \)
			if (next === "_") {
				i += 2;
				while (i < text.length) {
					if (text[i] === "\x07") break;
					if (text[i] === "\x1b" && text[i + 1] === "\\") {
						i++;
						break;
					}
					i++;
				}
				i++;
				if (text[i - 1] === "\x1b") i++; // skip ST backslash
				continue;
			}
			// Lone ESC — pass through, count as zero width
			i++;
			continue;
		}
		// Drop other C0 control bytes (NUL, etc.) — zero width
		const code = ch.charCodeAt(0);
		if (code < 0x20) {
			i++;
			continue;
		}
		// Basic wide char detection (CJK ranges)
		width +=
			code >= 0x1100 &&
			(code <= 0x115f ||
				code === 0x2329 ||
				code === 0x232a ||
				(code >= 0x2e80 && code <= 0xa4cf && code !== 0x303f) ||
				(code >= 0xac00 && code <= 0xd7a3) ||
				(code >= 0xf900 && code <= 0xfaff) ||
				(code >= 0xfe10 && code <= 0xfe19) ||
				(code >= 0xfe30 && code <= 0xfe6f) ||
				(code >= 0xff00 && code <= 0xff60) ||
				(code >= 0xffe0 && code <= 0xffe6) ||
				(code >= 0x20000 && code <= 0x2fffd) ||
				(code >= 0x30000 && code <= 0x3fffd))
				? 2
				: 1;
		i++;
	}
	return width;
}

// Truncate text to fit within width
export function truncateToWidth(text: string, width: number): string {
	let result = "";
	let currentWidth = 0;
	let inEscape = false;

	for (let i = 0; i < text.length; i++) {
		if (text[i] === "\x1b" && text[i + 1] === "[") {
			inEscape = true;
			result += text[i];
		} else if (inEscape) {
			result += text[i];
			if (text[i] === "m") {
				inEscape = false;
			}
		} else {
			const charWidth = visibleWidth(text[i]);
			if (currentWidth + charWidth > width) {
				result += "...";
				break;
			}
			result += text[i];
			currentWidth += charWidth;
		}
	}

	return result;
}

// Clamp a line to a visible width, preserving ALL escape sequences (CSI colors,
// OSC hyperlinks/markers). Unlike truncateToWidth this adds no ellipsis — it is
// used per-frame to guarantee a line can never exceed the terminal width and
// wrap onto the next row (which would desync the whole differential frame).
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
		const w = visibleWidth(ch);
		if (visible + w > width) break;
		result += ch;
		visible += w;
		i++;
	}
	return result;
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

import { Buffer } from "node:buffer";
// ── Terminal input ───────────────────────────────────────────────────────────
// Keyboard comes from process.stdin in raw mode (pi-style). The bridge child's
// events arrive on its own stdout pipe, so stdin is free for the keyboard.
import process from "node:process";

// ── TUI — Differential rendering ────────────────────────────────────────────

export class TUI extends Container {
	private renderRequested = false;
	private renderTimer: ReturnType<typeof setTimeout> | null = null;
	private lastRenderAt = 0;
	private static readonly MIN_RENDER_INTERVAL_MS = 16;
	private stopped = false;
	private focusedComponent: Component | null = null;
	private overlayStack: Array<{
		component: Component;
		options?: OverlayOptions;
		preFocus: Component | null;
		hidden: boolean;
		focusOrder: number;
	}> = [];
	private focusOrderCounter = 0;
	private inputListeners: Set<
		(data: string) => { consume?: boolean; data?: string } | undefined
	> = new Set();
	private stdinHandler: ((data: string | Buffer) => void) | null = null;
	private wasRaw = false;
	private _scrollOffsetInternal: number = 0;
	private _viewportHeight: number = 0;
	private previousLines: string[] = [];
	private previousCursorRow = -1;
	private previousCursorCol = -1;
	private previousCursorVisible: boolean | null = null;
	private lastRenderMetrics: RendererMetrics = EMPTY_RENDERER_METRICS;
	private scrollableComponent: Scrollable | null = null;
	private inputBarComponent: Component | null = null;
	private fixedBottomComponent: Component | null = null;
	private fixedAboveInputComponent: Component | null = null;

	private _showHardwareCursor = true;

	constructor(_outStream: NodeJS.WriteStream, showCursor = true) {
		super();
		this._showHardwareCursor = showCursor;
	}

	setShowHardwareCursor(enabled: boolean): void {
		this._showHardwareCursor = enabled;
		this.requestRender();
	}

	get scrollOffset(): number {
		return this._scrollOffsetInternal;
	}

	get isAtBottom(): boolean {
		if (!this.scrollableComponent) return true;
		return this.scrollableComponent.isAtBottom;
	}

	setFocus(component: Component | null): void {
		if (isFocusable(this.focusedComponent)) {
			(this.focusedComponent as Focusable).focused = false;
		}
		this.focusedComponent = component;
		if (isFocusable(component)) {
			(component as Focusable).focused = true;
		}
	}

	showOverlay(
		component: Component,
		options?: OverlayOptions,
	): {
		hide: () => void;
		setHidden: (hidden: boolean) => void;
		focus: () => void;
	} {
		const entry = {
			component,
			options,
			preFocus: this.focusedComponent,
			hidden: false,
			focusOrder: ++this.focusOrderCounter,
		};
		this.overlayStack.push(entry);
		this.requestRender();

		return {
			hide: () => {
				const idx = this.overlayStack.indexOf(entry);
				if (idx >= 0) this.overlayStack.splice(idx, 1);
				this.requestRender();
			},
			setHidden: (hidden: boolean) => {
				entry.hidden = hidden;
				this.requestRender();
			},
			focus: () => {
				this.setFocus(component);
				entry.focusOrder = ++this.focusOrderCounter;
				this.requestRender();
			},
		};
	}

	hideOverlay(): void {
		this.overlayStack.pop();
		this.requestRender();
	}

	/** Remove a specific overlay from the stack and restore focus to its pre-focus target. */
	removeOverlay(component: Component): void {
		const idx = this.overlayStack.findIndex((e) => e.component === component);
		if (idx >= 0) {
			const entry = this.overlayStack[idx];
			this.overlayStack.splice(idx, 1);
			// Mark component invisible so input listeners stop consuming keys
			if (
				"visible" in entry.component &&
				typeof (entry.component as { visible?: unknown }).visible === "boolean"
			) {
				(entry.component as { visible: boolean }).visible = false;
			}
			// Restore focus to whatever was focused before this overlay was shown
			if (isFocusable(entry.preFocus)) {
				(entry.preFocus as Focusable).focused = false;
			}
			this.setFocus(entry.preFocus);
			this.requestRender();
		}
	}

	addInputListener(
		listener: (
			data: string,
		) => { consume?: boolean; data?: string } | undefined,
	): () => void {
		this.inputListeners.add(listener);
		return () => {
			this.inputListeners.delete(listener);
		};
	}

	start(): void {
		this.stopped = false;

		// Keyboard input: process.stdin in raw mode (pi-style). The bridge child's
		// events arrive on its own stdout pipe, so stdin is dedicated to keys.
		this.wasRaw = process.stdin.isRaw ?? false;
		process.stdin.setEncoding("utf-8");
		if (process.stdin.setRawMode) {
			try {
				process.stdin.setRawMode(true);
			} catch (_e: unknown) {
				// raw mode unavailable (e.g. piped stdin) — keys won't work but the UI
				// still renders; degrade gracefully rather than crash.
			}
		}
		process.stdin.resume();
		this.stdinHandler = (data: string | Buffer) => {
			const str = Buffer.isBuffer(data) ? data.toString("utf-8") : data;
			this.handleInput(normalizeKeyboardInput(str));
			this.requestRender();
		};
		process.stdin.on("data", this.stdinHandler);

		// Enter alternate screen buffer + hide cursor + enable bracketed paste.
		// Push Kitty's disambiguate-escape-codes keyboard mode when supported.
		// Unsupported terminals safely ignore it; supporting terminals can then
		// report Ctrl+M separately from Enter as CSI 109;5u.
		// The alt screen gives us a fixed canvas to redraw each frame from the
		// home position. Bracketed paste makes the terminal wrap pasted text in
		// \x1b[200~ … \x1b[201~ so the app can distinguish paste from typed input.
		this.write("\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l\x1b[?2004h\x1b[>1u");

		this.requestRender(true);
	}

	stop(): void {
		this.stopped = true;
		if (this.renderTimer) {
			clearTimeout(this.renderTimer);
			this.renderTimer = null;
		}
		this.disableMouse();
		// Show cursor + leave alternate screen + disable bracketed paste,
		// restoring the user's terminal.
		this.write("\x1b[<u\x1b[?25h\x1b[?1049l\x1b[?2004l");

		if (this.stdinHandler) {
			process.stdin.removeListener("data", this.stdinHandler);
			this.stdinHandler = null;
		}
		if (process.stdin.setRawMode) {
			try {
				process.stdin.setRawMode(this.wasRaw);
			} catch (_e: unknown) {
				// ignore
			}
		}
		process.stdin.pause();
	}

	requestRender(force = false): void {
		if (force) {
			this.previousLines = [];
			this.previousCursorRow = -1;
			this.previousCursorCol = -1;
			this.previousCursorVisible = null;
		}
		if (this.renderRequested) return;
		this.renderRequested = true;
		process.nextTick(() => this.scheduleRender());
	}

	private scheduleRender(): void {
		if (this.stopped || this.renderTimer || !this.renderRequested) return;
		const elapsed = Date.now() - this.lastRenderAt;
		const delay = Math.max(0, TUI.MIN_RENDER_INTERVAL_MS - elapsed);
		this.renderTimer = setTimeout(() => {
			this.renderTimer = null;
			if (this.stopped || !this.renderRequested) return;
			this.renderRequested = false;
			this.lastRenderAt = Date.now();
			this.doRender();
			if (this.renderRequested) this.scheduleRender();
		}, delay);
	}

	private static readonly WHEEL_STEP = 4;

	private handleInput(data: string): void {
		const hasVisibleOverlay = this.overlayStack.some((entry) => {
			if (entry.hidden) return false;
			if (
				"visible" in entry.component &&
				typeof (entry.component as { visible?: unknown }).visible === "boolean"
			) {
				return (entry.component as { visible: boolean }).visible;
			}
			return true;
		});

		// Handle SGR mouse events: \x1b[<button;column;row(M|m). A single stdin read
		// can batch several wheel ticks; coalesce them into one scroll so fast wheel
		// spins move proportionally without queuing a render per tick.
		if (data.startsWith("\x1b[<")) {
			const re = /\x1b\[<(\d+);(\d+);(\d+)([Mm])/g;
			let net = 0; // +down / -up, in wheel ticks
			let consumed = 0;
			let clicked = false;
			let m: RegExpExecArray | null;
			while ((m = re.exec(data)) !== null) {
				const btn = parseInt(m[1], 10);
				const column = parseInt(m[2], 10) - 1;
				const row = parseInt(m[3], 10) - 1;
				if (btn === 64)
					net -= 1; // wheel up → older content
				else if (btn === 65) net += 1; // wheel down → newer content
				else if (
					btn === 0 &&
					m[4] === "M" &&
					!hasVisibleOverlay &&
					row >= 0 &&
					row < this._viewportHeight
				) {
					clicked =
						this.scrollableComponent?.handleMouse?.(column, row) === true ||
						clicked;
				}
				consumed += m[0].length;
			}
			// Pure mouse chunk → apply coalesced scroll once, then stop.
			if (net !== 0 && consumed === data.length) {
				if (net > 0) this.scrollDown(net * TUI.WHEEL_STEP);
				else this.scrollUp(-net * TUI.WHEEL_STEP);
				return;
			}
			if (clicked && consumed === data.length) return;
			if (consumed === data.length) return; // mouse-only chunk, nothing to scroll
		}

		// Scroll keys are global in coding-agent TUIs: the transcript can move while
		// the prompt keeps focus. Plain arrows remain input/history navigation.
		// But if any overlay is visible, skip scrolling so the overlay gets first
		// crack at the keys (e.g. reasoner selector, plugin manager).
		if (!hasVisibleOverlay && this.scrollableComponent) {
			if (data === "\x1b[5~") {
				this.scrollUp(Math.max(4, Math.floor(this._viewportHeight * 0.8)));
				return;
			}
			if (data === "\x1b[6~") {
				this.scrollDown(Math.max(4, Math.floor(this._viewportHeight * 0.8)));
				return;
			}
			if (
				data === "\x1b[1;5H" ||
				(data === "\x1b[H" && !this.isInputFocused())
			) {
				this.scrollToTop();
				return;
			}
			if (
				data === "\x1b[1;5F" ||
				(data === "\x1b[F" && !this.isInputFocused())
			) {
				this.scrollToBottom();
				return;
			}

			// Handle arrow scrolling when not focused on input bar.
			const isInputFocused = this.focusedComponent === this.inputBarComponent;
			if (!isInputFocused) {
				if (data === "\x1b[A" || data === "\x1bOA") {
					/* Up arrow */ this.scrollUp(1);
					return;
				}
				if (data === "\x1b[B" || data === "\x1bOB") {
					/* Down arrow */ this.scrollDown(1);
					return;
				}
				if (data === "\x1b[H" || data === "\x1bOH") {
					/* Home */ this.scrollToTop();
					return;
				}
				if (data === "\x1b[F" || data === "\x1bOF") {
					/* End */ this.scrollToBottom();
					return;
				}
			}
		}
		for (const listener of this.inputListeners) {
			const result = listener(data);
			if (result?.consume) return;
		}
		// Fallback for overlays not owned by an application-level listener.
		if (data === "\x1b" && this.overlayStack.length > 0) {
			const top = this.overlayStack[this.overlayStack.length - 1];
			const comp = top.component;
			if (
				!top.hidden &&
				"handleInput" in comp &&
				typeof (comp as Record<string, unknown>).handleInput === "function"
			) {
				const action = (
					(comp as Record<string, unknown>).handleInput as (
						d: string,
					) => unknown
				)(data);
				const actionObj = action as { type?: string } | null;
				if (actionObj?.type === "close" || actionObj?.type === "cancel") {
					this.removeOverlay(top.component);
					return;
				}
			}
		}
		if (this.focusedComponent && "handleInput" in this.focusedComponent) {
			(
				this.focusedComponent as { handleInput: (data: string) => void }
			).handleInput(data);
		}
	}

	private doRender(): void {
		if (this.stopped) return;
		try {
			this._doRenderInner();
		} catch (err) {
			// Render crash: print a minimal error and fall back to a blank
			// screen so the terminal is left in a usable state rather than
			// showing corrupted escape sequences.
			const msg = err instanceof Error ? err.message : String(err);
			this.previousLines = [];
			this.previousCursorRow = -1;
			this.previousCursorCol = -1;
			this.previousCursorVisible = null;
			// Close every state the renderer may have left open, clear the
			// potentially partial frame, and leave a visible cursor. The next
			// render starts from an invalidated cache and therefore repaints.
			process.stderr.write(
				"\x1b[?2026l\x1b]8;;\x1b\\\x1b[0m\x1b[2J\x1b[H\x1b[?25h" +
					`\n\x1b[38;5;203m[TUI render error]\x1b[0m ${msg}\n`,
			);
			// eslint-disable-next-line no-console
			console.error("TUI render crash:", err);
		}
	}

	private _doRenderInner(): void {
		const frameStartedAt = performance.now();
		const termWidth = Math.max(1, process.stdout.columns || 80);
		const termHeight = Math.max(1, process.stdout.rows || 24);

		let inputLines: string[];
		try {
			inputLines = this.inputBarComponent
				? this.inputBarComponent.render(termWidth)
				: [" ".repeat(termWidth)];
		} catch (_e: unknown) {
			inputLines = [" ".repeat(termWidth)];
		}

		let statusLines: string[];
		try {
			statusLines = this.fixedBottomComponent
				? this.fixedBottomComponent.render(termWidth)
				: [" ".repeat(termWidth)];
		} catch (_e: unknown) {
			statusLines = [" ".repeat(termWidth)];
		}

		const inputHeight = Math.max(1, inputLines.length);
		const statusHeight = Math.max(1, statusLines.length);

		// Optional pinned region above the input bar (todo list). Zero lines = hidden.
		let aboveInputLines: string[] = [];
		try {
			aboveInputLines = this.fixedAboveInputComponent
				? this.fixedAboveInputComponent.render(termWidth)
				: [];
		} catch (_e: unknown) {
			aboveInputLines = [];
		}
		// Interactive selectors participate in the fixed composer stack instead
		// of floating over transcript content. This matches pinned TODO/queue
		// behavior and keeps the selector physically attached to the input.
		aboveInputLines.push(...this.renderAboveInputOverlays(termWidth));
		const aboveInputHeight = aboveInputLines.length;


		// Fixed layout: transcript + divider + [pinned + divider] + input bar + divider + status footer.
		const transcriptHeight = Math.max(
			1,
			termHeight -
				2 -
				aboveInputHeight -
				inputHeight -
				statusHeight,
		);
		const transcriptWidth = termWidth;

		// Build output lines with fixed layout
		const lines: string[] = [];

		// 1. Transcript area (scrollable)
		if (this.scrollableComponent) {
			(
				this.scrollableComponent as unknown as {
					setViewportHeight: (h: number) => void;
				}
			).setViewportHeight(transcriptHeight);
			this._viewportHeight = transcriptHeight;
			let transcriptLines: string[];
			try {
				transcriptLines = this.scrollableComponent.render(transcriptWidth);
			} catch (_e: unknown) {
				// Component render failed — fill with safe placeholder
				transcriptLines = Array(transcriptHeight)
					.fill(0)
					.map(() => " ".repeat(transcriptWidth));
			}
			const totalLines = Math.max(
				transcriptLines.length,
				this.scrollableComponent.totalHeight,
			);
			const maxScroll = Math.max(0, totalLines - transcriptHeight);
			// Use scrollable component's scrollOffset (set during render by scrollToBottom)
			const comp = this.scrollableComponent as Scrollable;
			const scrollOff = Math.min(maxScroll, Math.max(0, comp.scrollOffset));
			const visibleLines = (comp as unknown as { rendersViewport?: boolean })
				.rendersViewport
				? transcriptLines
				: transcriptLines.slice(scrollOff, scrollOff + transcriptHeight);

			// Pad transcript to fill its slot
			while (lines.length < transcriptHeight) {
				lines.push(
					lines.length < visibleLines.length
						? visibleLines[lines.length]
						: " ".repeat(termWidth),
				);
			}
		} else {
			// Fill transcript area with spaces
			for (let i = 0; i < transcriptHeight; i++) {
				lines.push(" ".repeat(termWidth));
			}
		}

		// 2. Separator line above input
		lines.push(`\x1b[38;5;236m${"─".repeat(termWidth)}\x1b[0m`);

		// 2b. Pinned region above input (todo list), when present
		for (let i = 0; i < aboveInputHeight; i++) {
			lines.push(aboveInputLines[i] || " ".repeat(termWidth));
		}


		// 3. Input bar (fixed)
		for (let i = 0; i < inputHeight; i++) {
			lines.push(inputLines[i] || " ".repeat(termWidth));
		}

		// 4. Separator line below input
		lines.push(`\x1b[38;5;236m${"─".repeat(termWidth)}\x1b[0m`);

		// 5. Status bar (fixed, at bottom)
		for (let i = 0; i < statusHeight; i++) {
			lines.push(statusLines[i] || " ".repeat(termWidth));
		}

		// Pad to termHeight if needed
		while (lines.length < termHeight) {
			lines.push(" ".repeat(termWidth));
		}

		// Compose overlays (slash popup, etc.)
		const finalLines = this.composeOverlays(
			lines,
			termWidth,
			termHeight,
			transcriptHeight,
		);
		const layoutFinishedAt = performance.now();

		// Leave the physical last column unused: writing it can put terminals into
		// pending-autowrap state and shift the next update down a row.
		const renderWidth = Math.max(1, termWidth - 1);
		let changes = "";
		let dirtyRows = 0;
		let changedCells = 0;
		let cursorMoves = 0;
		let dirtyTop = Number.POSITIVE_INFINITY;
		let dirtyBottom = -1;

		// The InputBar marks the edit position with CURSOR_MARKER. Find it so we
		// can park the hardware cursor exactly there, and strip it from output.
		let markerRow = -1;
		let markerCol = 0;

		for (let row = 0; row < termHeight; row++) {
			const prevLine = this.previousLines[row];
			const newLine = row < finalLines.length ? finalLines[row] : " ".repeat(termWidth);
			const hasMarker = newLine.includes(CURSOR_MARKER);

			// Extract cursor marker position before stripping
			if (hasMarker) {
				const markerIdx = newLine.indexOf(CURSOR_MARKER);
				markerRow = row;
				markerCol = visibleWidth(newLine.slice(0, markerIdx));
			}

			// Strip CURSOR_MARKER for cell parsing
			const cleanNew = newLine.replace(CURSOR_MARKER, "");
			const cleanPrev = prevLine?.replace(CURSOR_MARKER, "") ?? "";

			// Image protocols are commands rather than printable cells. Repaint
			// those rows atomically instead of trying to split their payload.
			if (isImageLine(cleanNew) || isImageLine(cleanPrev)) {
				if (cleanNew !== cleanPrev) {
					changes += `\x1b[${row + 1};1H\x1b[0m\x1b[2K`;
					changes += isImageLine(cleanNew)
						? cleanNew
						: clampLineToWidth(cleanNew, renderWidth);
					dirtyRows++;
					dirtyTop = Math.min(dirtyTop, row);
					dirtyBottom = Math.max(dirtyBottom, row);
					changedCells += renderWidth;
					cursorMoves++;
				}
				continue;
			}

			const lineDiff = diffTerminalLineWithMetrics(
				cleanPrev,
				cleanNew,
				row,
				renderWidth,
			);
			changes += lineDiff.output;
			if (lineDiff.changedCells > 0) {
				dirtyRows++;
				dirtyTop = Math.min(dirtyTop, row);
				dirtyBottom = Math.max(dirtyBottom, row);
			}
			changedCells += lineDiff.changedCells;
			cursorMoves += lineDiff.cursorMoves;
		}

		this.previousLines = finalLines;
		// Park the hardware cursor at the input's edit position (under the
		// visible InputBar cursor). Falls back to the input line's first column
		// only if no marker was emitted, which keeps the cursor off the footer.
		const fallbackRow = Math.min(
			termHeight,
			transcriptHeight + 2 + aboveInputHeight,
		);
		const cursorRow = markerRow >= 0 ? markerRow + 1 : fallbackRow;
		const cursorCol =
			markerRow >= 0 ? Math.min(termWidth, markerCol + 1) : 1;
		const cursorMoved =
			changes.length > 0 ||
			cursorRow !== this.previousCursorRow ||
			cursorCol !== this.previousCursorCol;
		const cursorUpdate = cursorMoved
			? `\x1b[${cursorRow};${cursorCol}H`
			: "";
		const visibilityChanged =
			this._showHardwareCursor !== this.previousCursorVisible;
		const visibilityUpdate = visibilityChanged
			? this._showHardwareCursor
				? "\x1b[?25h"
				: "\x1b[?25l"
			: "";
		const diffFinishedAt = performance.now();
		const writeStartedAt = performance.now();
		let bytesWritten = 0;
		if (changes) {
			// Cursor restoration is part of the synchronized frame, so the user
			// never observes it parked on the last streamed cell.
			const buffer =
				`\x1b[?2026h${changes}${cursorUpdate}${visibilityUpdate}` +
				"\x1b[?2026l";
			this.write(buffer);
			bytesWritten = Buffer.byteLength(buffer);
		} else {
			const terminalStateUpdate = cursorUpdate + visibilityUpdate;
			if (terminalStateUpdate) {
				this.write(terminalStateUpdate);
				bytesWritten = Buffer.byteLength(terminalStateUpdate);
			}
		}
		if (cursorMoved) {
			cursorMoves++;
			this.previousCursorRow = cursorRow;
			this.previousCursorCol = cursorCol;
		}
		if (visibilityChanged) {
			this.previousCursorVisible = this._showHardwareCursor;
		}
		const frameFinishedAt = performance.now();
		this.lastRenderMetrics = {
			bytesWritten,
			changedCells,
			cursorMoves,
			diffTimeMs: diffFinishedAt - layoutFinishedAt,
			dirtyRegion:
				dirtyBottom >= 0 ? { top: dirtyTop, bottom: dirtyBottom } : null,
			dirtyRows,
			frameTimeMs: frameFinishedAt - frameStartedAt,
			layoutTimeMs: layoutFinishedAt - frameStartedAt,
			writeTimeMs: frameFinishedAt - writeStartedAt,
		};
	}

	getLastRenderMetrics(): RendererMetrics {
		return { ...this.lastRenderMetrics };
	}

	private composeOverlays(
		lines: string[],
		termWidth: number,
		_termHeight: number,
		transcriptHeight: number,
	): string[] {
		const result = [...lines];

		const visibleEntries = this.overlayStack.filter((e) => {
			if (e.options?.anchor === "aboveInput") return false;
			if (e.hidden) return false;
			// Also check component's visible property if it has one
			if (
				"visible" in e.component &&
				typeof (e.component as { visible?: unknown }).visible === "boolean"
			) {
				return (e.component as { visible: boolean }).visible;
			}
			return true;
		});

		for (const entry of visibleEntries) {
			const leftAligned = entry.options?.align === "left";
			const overlayWidth = leftAligned
				? Math.max(1, termWidth)
				: Math.max(
						40,
						Math.min(
							termWidth - 8,
							entry.options?.maxHeight ? termWidth * 0.6 : termWidth - 8,
						),
					);
			const overlayLines = entry.component.render(Math.max(1, overlayWidth));
			const overlayHeight = Math.min(
				overlayLines.length,
				entry.options?.maxHeight || 999,
			);

			// Calculate row position within transcript area
			let row = 0;
			switch (entry.options?.anchor) {
				case "center":
					row = Math.max(0, Math.floor((transcriptHeight - overlayHeight) / 2));
					break;
				case "bottom":
					// Flush against the bottom of the transcript area, i.e. directly
					// above the separator + input bar.
					row = Math.max(0, transcriptHeight - overlayHeight);
					break;
				default:
					row = 0;
					break;
			}

			// Horizontal offset: left-aligned overlays hug the left edge; otherwise
			// center within the terminal.
			const margin = leftAligned
				? 0
				: Math.max(2, Math.floor((termWidth - overlayWidth) / 2));

			for (let i = 0; i < overlayHeight; i++) {
				const idx = row + i;
				if (idx >= 0 && idx < result.length) {
					const srcLine = overlayLines[i] || "";
					const srcVis = visibleWidth(srcLine);
					// Pad with spaces, then overlay the content at the correct offset
					const basePad = " ".repeat(margin);
					const afterPad = " ".repeat(Math.max(0, termWidth - margin - srcVis));
					result[idx] = basePad + srcLine + afterPad;
				}
			}
		}

		return result;
	}

	private renderAboveInputOverlays(termWidth: number): string[] {
		const entries = this.overlayStack.filter((entry) => {
			if (entry.hidden || entry.options?.anchor !== "aboveInput") return false;
			if (
				"visible" in entry.component &&
				typeof (entry.component as { visible?: unknown }).visible === "boolean"
			) {
				return (entry.component as { visible: boolean }).visible;
			}
			return true;
		});
		if (entries.length === 0) return [];

		// Only the most recently focused selector owns the composer region.
		const entry = entries.reduce((latest, candidate) =>
			candidate.focusOrder > latest.focusOrder ? candidate : latest,
		);
		const width = Math.max(1, termWidth - 1);
		const rendered = entry.component.render(width);
		const maxHeight = entry.options?.maxHeight ?? rendered.length;
		return rendered.slice(0, maxHeight).map((line) => {
			const clamped = clampLineToWidth(line, width);
			return (
				clamped + " ".repeat(Math.max(0, termWidth - visibleWidth(clamped)))
			);
		});
	}

	private write(data: string): void {
		try {
			process.stdout.write(data);
		} catch (_e: unknown) {
			// Silently ignore write errors
		}
	}

	// ── Scroll controls ───────────────────────────────────────────────────

	setScrollableComponent(comp: Scrollable | null): void {
		this.scrollableComponent = comp;
	}

	setInputBarComponent(comp: Component | null): void {
		this.inputBarComponent = comp;
	}

	setFixedBottomComponent(comp: Component | null): void {
		this.fixedBottomComponent = comp;
	}

	// Pinned region rendered directly above the input bar (e.g. the todo list).
	// Renders nothing when the component returns no lines.
	setFixedAboveInputComponent(comp: Component | null): void {
		this.fixedAboveInputComponent = comp;
	}

	private scrollUp(lines: number): void {
		if (!this.scrollableComponent) return;
		this.scrollableComponent.scroll(lines);
		this.requestRender();
	}

	private scrollDown(lines: number): void {
		if (!this.scrollableComponent) return;
		this.scrollableComponent.scroll(-lines);
		this.requestRender();
	}

	private scrollToTop(): void {
		if (!this.scrollableComponent) return;
		this.scrollableComponent.scrollOffset = 0;
		this.requestRender();
	}

	scrollToBottom(): void {
		if (!this.scrollableComponent) return;
		this.scrollableComponent.scrollToBottom();
		this.requestRender();
	}

	// ── Mouse tracking ───────────────────────────────────────────────────────────

	private mouseEnabled = false;

	enableMouse(): void {
		if (this.mouseEnabled) return;
		this.write("\x1b[?1006h"); // SGR mouse encoding
		this.write("\x1b[?1000h"); // button + wheel events only (no motion flood)
		this.mouseEnabled = true;
	}

	disableMouse(): void {
		if (!this.mouseEnabled) return;
		this.write("\x1b[?1000l");
		this.write("\x1b[?1006l");
		this.mouseEnabled = false;
	}

	private isInputFocused(): boolean {
		return this.focusedComponent === this.inputBarComponent;
	}
}

function isImageLine(line: string): boolean {
	return line.includes("\x1b_G") || line.includes("\x1b]1337;");
}

// ── Cell-level rendering ──────────────────────────────────────────────────────
// Each cell is { char: string, attr: string } where attr is the full ANSI
// attribute string (e.g. "\x1b[38;5;46m\x1b[1m"). We parse each rendered line
// into cells, compare cell-by-cell against the previous frame, and emit only
// the changes (cursor movement + attribute change + character write).

interface Cell {
	char: string;
	attr: string;
	continuation: boolean;
}

/**
 * Parse an ANSI-styled line into an array of cells. Each cell has a character
 * and the accumulated attribute string that applies to it. Handles CSI, OSC,
 * and APC escape sequences.
 */
function parseLineIntoCells(line: string, targetWidth: number): Cell[] {
	const cells: Cell[] = [];
	let attr = "";
	let i = 0;
	const len = line.length;

	while (i < len && cells.length < targetWidth) {
		const ch = line[i];

		if (ch === "\x1b") {
			const next = line[i + 1];
			if (next === "[") {
				// CSI sequence
				let j = i + 2;
				while (j < len) {
					const fc = line.charCodeAt(j);
					if (fc >= 0x40 && fc <= 0x7e) break;
					j++;
				}
				const seq = line.slice(i, j + 1);
				i = j + 1;

				// Only SGR changes cell appearance. Other CSI commands must not
				// leak into a style restoration sequence.
				if (seq.endsWith("m")) {
					if (seq === "\x1b[m" || seq === "\x1b[0m" || seq === "\x1b[0;0m") {
						attr = "";
					} else {
						attr += seq;
					}
				}
				continue;
			}
			if (next === "]") {
				// OSC sequence
				let j = i + 2;
				while (j < len) {
					if (line[j] === "\x07") break;
					if (line[j] === "\x1b" && line[j + 1] === "\\") {
						j++;
						break;
					}
					j++;
				}
				const end = line[j] === "\x07" ? j + 1 : Math.min(len, j + 2);
				const seq = line.slice(i, end);
				i = end;
				attr += seq;
				continue;
			}
			if (next === "_") {
				// APC sequence
				let j = i + 2;
				while (j < len) {
					if (line[j] === "\x07") break;
					if (line[j] === "\x1b" && line[j + 1] === "\\") {
						j++;
						break;
					}
					j++;
				}
				const end = line[j] === "\x07" ? j + 1 : Math.min(len, j + 2);
				i = end;
				continue;
			}
			// Lone ESC
			attr += ch;
			i++;
			continue;
		}

		// Drop C0 control bytes (except tab which we handle below)
		const code = ch.charCodeAt(0);
		if (code < 0x20 && code !== 0x09) {
			i++;
			continue;
		}

		// Tab: expand to one space
		if (code === 0x09) {
			if (cells.length < targetWidth) {
				cells.push({ char: " ", attr, continuation: false });
			}
			i++;
			continue;
		}

		const codePoint = line.codePointAt(i);
		if (codePoint === undefined) break;
		const char = String.fromCodePoint(codePoint);
		const width = visibleWidth(char);
		if (width > 0 && cells.length + width <= targetWidth) {
			cells.push({ char, attr, continuation: false });
			for (let column = 1; column < width; column++) {
				cells.push({ char: "", attr, continuation: true });
			}
		}
		i += char.length;
	}

	// Styling blank padding is visually irrelevant and makes every trailing cell
	// appear changed when a component happens to omit a final reset.
	while (cells.length < targetWidth) {
		cells.push({ char: " ", attr: "", continuation: false });
	}

	return cells;
}

/**
 * Generate ANSI escape sequence to transition from prevCells to newCells,
 * starting at the given row. Only emits cursor movement + attribute changes
 * + character writes for changed cells. Uses a smart strategy:
 *   1. Move cursor to first changed cell
 *   2. For each subsequent cell: if attr changed, emit attr; if char changed,
 *      emit char; if both, emit attr then char
 *   3. If we reach a run of unchanged cells, jump cursor past them
 */
export interface TerminalLineDiff {
	output: string;
	changedCells: number;
	cursorMoves: number;
}

function cellLevelDiff(
	prevCells: Cell[],
	newCells: Cell[],
	row: number,
): TerminalLineDiff {
	let out = "";
	const closeHyperlink = "\x1b]8;;\x1b\\";
	const changed = new Array<boolean>(newCells.length).fill(false);
	for (let i = 0; i < newCells.length; i++) {
		const prev = prevCells[i];
		changed[i] =
			!prev ||
			prev.char !== newCells[i].char ||
			prev.attr !== newCells[i].attr ||
			prev.continuation !== newCells[i].continuation;
	}

	// A terminal cannot address the second half of a wide glyph independently.
	// Expand changes leftward so replacing either half repaints the whole glyph.
	for (let i = 1; i < changed.length; i++) {
		if (
			changed[i] &&
			(newCells[i].continuation || prevCells[i]?.continuation)
		) {
			changed[i - 1] = true;
		}
	}

	let column = 0;
	let changedCells = 0;
	let cursorMoves = 0;
	while (column < newCells.length) {
		if (!changed[column]) {
			column++;
			continue;
		}
		const start = column;
		while (column < newCells.length && changed[column]) column++;
		changedCells += column - start;
		cursorMoves++;

		out += `\x1b[${row + 1};${start + 1}H${closeHyperlink}\x1b[0m`;
		let activeAttr = "";
		for (let i = start; i < column; i++) {
			const cell = newCells[i];
			if (cell.continuation) continue;
			if (cell.attr !== activeAttr) {
				out += `\x1b[0m${cell.attr}`;
				activeAttr = cell.attr;
			}
			out += cell.char;
		}
		out += closeHyperlink;
	}

	return { output: out, changedCells, cursorMoves };
}

/** Build the terminal update for one printable row. Exported for regression tests. */
export function diffTerminalLine(
	previousLine: string,
	nextLine: string,
	row: number,
	width: number,
): string {
	return diffTerminalLineWithMetrics(previousLine, nextLine, row, width).output;
}

export function diffTerminalLineWithMetrics(
	previousLine: string,
	nextLine: string,
	row: number,
	width: number,
): TerminalLineDiff {
	return cellLevelDiff(
		parseLineIntoCells(previousLine, width),
		parseLineIntoCells(nextLine, width),
		row,
	);
}

// ── Overlay options ──────────────────────────────────────────────────────────

export interface OverlayOptions {
	anchor?: "center" | "top" | "bottom" | "aboveInput";
	align?: "center" | "left";
	maxHeight?: number;
	nonCapturing?: boolean;
}
