// ── Minimal TUI core ──────────────────────────────────────────────────────────
// Input routing, scroll state, and overlay stack for the TUI. Ink
// (ink-app/) owns painting, diffing, resize, and alt-screen; this module
// only decides what should be on screen, never how to draw it.

// ── Interfaces ────────────────────────────────────────────────────────────────
import type { InkOverlayModelProvider } from "../overlays/ink-overlay-model.ts";

export interface InkTextSpan {
	text: string;
	color?: string;
	backgroundColor?: string;
	bold?: boolean;
	dim?: boolean;
	underline?: boolean;
	italic?: boolean;
	inverse?: boolean;
}

export type InkTextRow = readonly InkTextSpan[];

export interface InkTextComponent {
	getInkTextRows(width: number): InkTextRow[];
	invalidate?(): void;
}

export interface InkComposerModel {
	prompt: string;
	headerHint: string | null;
	beforeCursor: string;
	atCursor: string;
	afterCursor: string;
	isPlaceholder: boolean;
	leftClipped: boolean;
	rightClipped: boolean;
	cursorColumn: number;
	focused: boolean;
}

export interface InkComposerComponent {
	getInkComposerModel(width: number): InkComposerModel;
	invalidate?(): void;
}

/** Overlay state owner. Native Ink overlays do not require a string renderer. */
export interface OverlayComponent extends InkOverlayModelProvider {
	invalidate?(): void;
}

/**
 * The component model handed to Ink for each render. Ink sizes the
 * transcript-vs-dock split instead of consuming a pre-composited line array.
 */
export interface TUIComponentsFrame {
	termWidth: number;
	termHeight: number;
	scrollableComponent: Scrollable | null;
	inputBarComponent: InkComposerComponent | null;
	fixedBottomComponent: InkTextComponent | null;
	fixedAboveInputComponent: InkTextComponent | null;
	overlayStack: readonly {
		component: OverlayComponent;
		options?: OverlayOptions;
		hidden: boolean;
		focusOrder: number;
	}[];
	showHardwareCursor: boolean;
}

export interface Focusable {
	focused: boolean;
}

export interface Scrollable extends InkTextComponent {
	scrollOffset: number;
	scroll(delta: number): void;
	scrollToBottom(): void;
	setViewportHeight(height: number): void;
	isAtBottom: boolean;
	handleMouse?(column: number, row: number): boolean;
}

export function isFocusable(
	c: InkComposerComponent | InkTextComponent | OverlayComponent | null,
): c is (InkComposerComponent | InkTextComponent | OverlayComponent) & Focusable {
	return c !== null && "focused" in c;
}

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

export function inkTextRowText(row: InkTextRow): string {
	return row.map((span) => span.text).join("");
}

export function clampInkTextRow(row: InkTextRow, width: number): InkTextRow {
	const result: InkTextSpan[] = [];
	let remaining = Math.max(0, width);
	for (const span of row) {
		if (remaining <= 0) break;
		const text = clampLineToWidth(span.text, remaining);
		if (text) result.push({ ...span, text });
		remaining -= visibleWidth(text);
	}
	return result;
}

export function padInkTextRow(row: InkTextRow, width: number): InkTextRow {
	const clipped = clampInkTextRow(row, width);
	const padding = Math.max(0, width - visibleWidth(inkTextRowText(clipped)));
	return padding > 0 ? [...clipped, { text: " ".repeat(padding) }] : clipped;
}

/** Plain-text projection for model tests and width calculations. */
export function inkTextComponentLines(component: InkTextComponent, width: number): string[] {
	return component.getInkTextRows(width).map(inkTextRowText);
}

export function ansi256ToHex(index: number): string {
	const base = ["#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080", "#c0c0c0", "#808080", "#ff0000", "#00ff00", "#ffff00", "#0000ff", "#ff00ff", "#00ffff", "#ffffff"];
	if (index < 16) return base[Math.max(0, index)] ?? "#ffffff";
	if (index >= 232) {
		const value = 8 + (Math.min(255, index) - 232) * 10;
		return `#${value.toString(16).padStart(2, "0").repeat(3)}`;
	}
	const n = Math.min(231, index) - 16;
	const channel = (part: number): number => part === 0 ? 0 : 55 + part * 40;
	return `#${[Math.floor(n / 36), Math.floor((n % 36) / 6), n % 6].map((part) => channel(part).toString(16).padStart(2, "0")).join("")}`;
}

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
		// Walk by code point so surrogate pairs (astral chars, most emoji) count
		// as one character instead of two, which would otherwise desync width
		// bookkeeping (e.g. table column padding) from what the terminal draws.
		const codePoint = text.codePointAt(i);
		const char = codePoint === undefined ? ch : String.fromCodePoint(codePoint);
		if (codePoint !== undefined && codePoint >= 0xe000 && codePoint <= 0xf8ff) {
			i += char.length;
			continue;
		}
		width +=
			codePoint !== undefined &&
			codePoint >= 0x1100 &&
			(codePoint <= 0x115f ||
				codePoint === 0x2329 ||
				codePoint === 0x232a ||
				(codePoint >= 0x2e80 && codePoint <= 0xa4cf && codePoint !== 0x303f) ||
				(codePoint >= 0xac00 && codePoint <= 0xd7a3) ||
				(codePoint >= 0xf900 && codePoint <= 0xfaff) ||
				(codePoint >= 0xfe10 && codePoint <= 0xfe19) ||
				(codePoint >= 0xfe30 && codePoint <= 0xfe6f) ||
				(codePoint >= 0xff00 && codePoint <= 0xff60) ||
				(codePoint >= 0xffe0 && codePoint <= 0xffe6) ||
				(codePoint >= 0x20000 && codePoint <= 0x2fffd) ||
				(codePoint >= 0x30000 && codePoint <= 0x3fffd) ||
				// Emoji outside the CJK ranges above are still typically wide.
				(codePoint >= 0x1f000 && codePoint <= 0x1ffff))
				? 2
				: 1;
		i += char.length;
	}
	return width;
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
		const w = visibleWidth(ch);
		if (visible + w > width) break;
		result += ch;
		visible += w;
		i++;
	}
	return result;
}

// ── Container ────────────────────────────────────────────────────────────────

export class InkTextContainer implements InkTextComponent {
	private readonly children: InkTextComponent[] = [];

	addChild(component: InkTextComponent): void {
		this.children.push(component);
	}

	getInkTextRows(width: number): InkTextRow[] {
		const lines: InkTextRow[] = [];
		for (const child of this.children) {
			for (const line of child.getInkTextRows(width)) {
				lines.push(line);
			}
		}
		return lines;
	}
}

// ── Terminal input ───────────────────────────────────────────────────────────
// Ink owns stdin, raw mode, resize, and alt-screen; TUI only runs input
// routing / scroll / overlay state against frames delivered via
// onComponentsFrame.

// ── TUI — Input routing, scroll state, overlay stack ────────────────────────

export class TUI {
	private renderRequested = false;
	private renderTimer: ReturnType<typeof setTimeout> | null = null;
	private lastRenderAt = 0;
	private static readonly MIN_RENDER_INTERVAL_MS = 16;
	private started = false;
	private stopped = false;
	private focusedComponent: InkComposerComponent | InkTextComponent | OverlayComponent | null = null;
	private overlayStack: Array<{
		component: OverlayComponent;
		options?: OverlayOptions;
		preFocus: InkComposerComponent | InkTextComponent | OverlayComponent | null;
		hidden: boolean;
		focusOrder: number;
	}> = [];
	private focusOrderCounter = 0;
	private inputListeners: Set<
		(data: string) => { consume?: boolean; data?: string } | undefined
	> = new Set();
	private _viewportHeight: number = 0;
	private scrollableComponent: Scrollable | null = null;
	private inputBarComponent: InkComposerComponent | null = null;
	private fixedBottomComponent: InkTextComponent | null = null;
	private fixedAboveInputComponent: InkTextComponent | null = null;

	private _showHardwareCursor = true;
	private onComponentsFrame?: (frame: TUIComponentsFrame) => void;

	constructor(showCursor = true) {
		this._showHardwareCursor = showCursor;
	}

	/**
	 * Set or replace the components-frame sink after construction. Needed when
	 * the host renderer (e.g. an Ink component) only knows its own state
	 * setter after it mounts, which happens after LogicianTUI/TUI must
	 * already exist.
	 */
	setOnComponentsFrame(onComponentsFrame: (frame: TUIComponentsFrame) => void): void {
		this.onComponentsFrame = onComponentsFrame;
	}

	/** Feed raw stdin bytes -- Ink owns stdin and forwards them here. */
	feedInput(data: string): void {
		this.handleInput(data);
	}

	setShowHardwareCursor(enabled: boolean): void {
		this._showHardwareCursor = enabled;
		this.requestRender();
	}

	get isAtBottom(): boolean {
		if (!this.scrollableComponent) return true;
		return this.scrollableComponent.isAtBottom;
	}

	setFocus(component: InkComposerComponent | InkTextComponent | OverlayComponent | null): void {
		if (isFocusable(this.focusedComponent)) {
			(this.focusedComponent as Focusable).focused = false;
		}
		this.focusedComponent = component;
		if (isFocusable(component)) {
			(component as Focusable).focused = true;
		}
	}

	showOverlay(
		component: OverlayComponent,
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
	removeOverlay(component: OverlayComponent): void {
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

	/** Ink owns stdin, raw mode, resize, and alt-screen; this only starts the render-request scheduler. */
	start(): void {
		this.started = true;
		this.stopped = false;
		this.requestRender();
	}

	stop(): void {
		this.stopped = true;
		if (this.renderTimer) {
			clearTimeout(this.renderTimer);
			this.renderTimer = null;
		}
		this.disableMouse();
	}

	requestRender(): void {
		// Defer renders until Ink has mounted and is ready to receive them.
		if (!this.started) return;
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
			if (clicked && consumed === data.length) {
				this.requestRender();
				return;
			}
			if (consumed === data.length) return; // mouse-only chunk, nothing to scroll
		}

		// Fast repeated key presses (PageUp/PageDown/Home/End, held or double-
		// tapped) are routinely coalesced by SSH/tmux/mosh into one stdin chunk.
		// Every branch below matches a single sequence with strict equality, so
		// an unsplit multi-sequence chunk (e.g. "\x1b[6~\x1b[6~") would fail every
		// check and fall through unhandled. Replay recognized navigation
		// sequences one at a time when the whole chunk is made of 2+ of them.
		// Arrow keys are excluded — input-bar.ts already splits and handles a
		// pure arrow-key batch with input/history-navigation semantics.
		const navBatch = data.match(
			/\x1b\[(?:5~|6~|1;5H|1;5F|H|F)/g,
		);
		if (
			navBatch &&
			navBatch.length > 1 &&
			navBatch.join("") === data
		) {
			for (const key of navBatch) this.handleInput(key);
			return;
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
			const termWidth = Math.max(1, process.stdout.columns || 80);
			const termHeight = Math.max(1, process.stdout.rows || 24);
			this.onComponentsFrame?.({
				termWidth,
				termHeight,
				scrollableComponent: this.scrollableComponent,
				inputBarComponent: this.inputBarComponent,
				fixedBottomComponent: this.fixedBottomComponent,
				fixedAboveInputComponent: this.fixedAboveInputComponent,
				overlayStack: this.overlayStack,
				showHardwareCursor: this._showHardwareCursor,
			});
		} catch (err) {
			// A crash here would otherwise be silently swallowed by Ink's own
			// render cycle. Surface it to stderr, which Ink doesn't own, so it's
			// visible outside the alt-screen the next frame paints over.
			const msg = err instanceof Error ? err.message : String(err);
			process.stderr.write(`\n[TUI render error] ${msg}\n`);
			// eslint-disable-next-line no-console
			console.error("TUI render crash:", err);
		}
	}

	// ── Scroll controls ───────────────────────────────────────────────────

	setScrollableComponent(comp: Scrollable | null): void {
		this.scrollableComponent = comp;
	}

	/** Keep input hit-testing and transcript slicing on the same Ink-owned viewport. */
	setViewportHeight(height: number): void {
		this._viewportHeight = Math.max(0, height);
		this.scrollableComponent?.setViewportHeight(this._viewportHeight);
	}

	setInputBarComponent(comp: InkComposerComponent | null): void {
		this.inputBarComponent = comp;
	}

	setFixedBottomComponent(comp: InkTextComponent | null): void {
		this.fixedBottomComponent = comp;
	}

	// Pinned region rendered directly above the input bar (e.g. the todo list).
	// Renders nothing when the component returns no lines.
	setFixedAboveInputComponent(comp: InkTextComponent | null): void {
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
	// Ink has no mouse API, so TUI still owns this: raw mode-toggle sequences
	// written straight to stdout, orthogonal to Ink's own frame buffer.

	private mouseEnabled = false;

	private write(data: string): void {
		try {
			process.stdout.write(data);
		} catch (_e: unknown) {
			// Silently ignore write errors
		}
	}

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

export function isImageLine(line: string): boolean {
	return line.includes("\x1b_G") || line.includes("\x1b]1337;");
}

// ── Overlay options ──────────────────────────────────────────────────────────

export interface OverlayOptions {
	anchor?: "center" | "top" | "bottom" | "aboveInput";
	align?: "center" | "left";
	maxHeight?: number;
	nonCapturing?: boolean;
}
