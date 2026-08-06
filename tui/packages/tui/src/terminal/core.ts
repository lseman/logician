// ── Minimal TUI core ──────────────────────────────────────────────────────────
// Differential rendering engine — minimal, no external deps

// Component/Container/CURSOR_MARKER/ANSI string utilities live in
// primitives.ts, which the layout engine (rendering/layout.ts and its
// dependents) also imports. core.ts must not import the layout engine at
// module scope: primitives.ts <- layout.ts <- core.ts would otherwise close
// a cycle back through this file, and since stack.ts/scroll-view.ts extend
// Container at module-evaluation time, Node's ESM loader hits Container
// still in its temporal dead zone and crashes. The three layout-engine
// functions core.ts needs are only ever called inside method bodies, so a
// one-way, deferred-in-effect dependency (this file importing rendering/,
// never the reverse) is safe.
export {
	BOLD,
	type Component,
	Container,
	CURSOR_MARKER,
	clampLineToWidth,
	compositeTuiLine,
	DIM,
	type Focusable,
	isFocusable,
	RESET,
	type Scrollable,
	Spacer,
	visibleWidth,
} from "./primitives.ts";

import {
	getComponentBoxAt,
	getScrollViewsAt,
	type LayoutFrame,
	type LayoutRect,
	renderLayoutFrame,
} from "../rendering/layout.ts";
import {
	type Component,
	Container,
	CURSOR_MARKER,
	clampLineToWidth,
	type Focusable,
	isFocusable,
	type Scrollable,
	Spacer,
	visibleWidth,
} from "./primitives.ts";

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
	return (
		data
			// biome-ignore lint/suspicious/noControlCharactersInRegex: terminal CSI escape sequence
			.replace(/\x1b\[27(?:;1)?u/g, "\x1b")
			.replace(/\x1b\[(\d+);([56])u/g, (sequence, codepointText: string) => {
				const codepoint = Number(codepointText);
				const lowerCodepoint =
					codepoint >= 65 && codepoint <= 90 ? codepoint + 32 : codepoint;
				if (lowerCodepoint === 105 || lowerCodepoint === 109) return sequence;
				if (lowerCodepoint < 96 || lowerCodepoint > 127) return sequence;
				return String.fromCharCode(lowerCodepoint & 0x1f);
			})
	);
}

import { Buffer } from "node:buffer";
import { appendFileSync } from "node:fs";
// ── Terminal input ───────────────────────────────────────────────────────────
// Keyboard comes from process.stdin in raw mode (pi-style). The bridge child's
// events arrive on its own stdout pipe, so stdin is free for the keyboard.
import process from "node:process";

// ── Render debug logging ─────────────────────────────────────────────────────
// Opt-in, off by default: LOGICIAN_TUI_DEBUG_RENDER=1 appends one JSON line per
// frame to the given path (or ./logician-tui-render.log). Writing to a file
// rather than stdout/stderr keeps it out of the alt-screen buffer being
// measured. This is the only consumer of RendererMetrics today — without it
// the per-frame timing/dirty-row data core.ts already computes had nowhere to
// go, so profiling real sessions meant adding ad-hoc console.error calls.
const RENDER_DEBUG_ENABLED = process.env.LOGICIAN_TUI_DEBUG_RENDER === "1";
const RENDER_DEBUG_PATH =
	process.env.LOGICIAN_TUI_DEBUG_RENDER_PATH || "logician-tui-render.log";

function logRenderMetrics(metrics: RendererMetrics): void {
	try {
		appendFileSync(
			RENDER_DEBUG_PATH,
			`${JSON.stringify({ t: Date.now(), ...metrics })}\n`,
		);
	} catch {
		// Debug logging is best-effort; never let it break rendering.
	}
}

// ── TUI — Differential rendering ────────────────────────────────────────────

export class TUI extends Container {
	private renderRequested = false;
	private renderImmediateRequested = false;
	private renderTimer: ReturnType<typeof setTimeout> | null = null;
	private lastRenderFinishedAt = 0;
	private static readonly MIN_RENDER_INTERVAL_MS = 16;
	private started = false;
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
	private resizeHandler: (() => void) | null = null;
	private wasRaw = false;
	private _scrollOffsetInternal: number = 0;
	private _viewportHeight: number = 0;
	private previousLines: string[] = [];
	private previousCursorRow = -1;
	private previousCursorCol = -1;
	private previousCursorVisible: boolean | null = null;
	private lastRenderMetrics: RendererMetrics = EMPTY_RENDERER_METRICS;
	private layoutRoot: Component | null = null;
	private currentLayoutFrame: LayoutFrame | null = null;
	private paintedOverlayClickRects: Array<{
		rect: LayoutRect;
		onClick: () => void;
	}> = [];
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
		const idx = this.overlayStack.findIndex(e => e.component === component);
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
		this.started = true;
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
			// Composer/navigation feedback should not wait behind the streaming frame
			// budget. Upgrade any already-scheduled render to an immediate frame.
			this.requestRender(false, true);
		};
		process.stdin.on("data", this.stdinHandler);

		// A resize lands between two frames' worth of `previousLines`, which were
		// diffed against the old termWidth/termHeight. Cell columns and row count
		// both shift, so patching the old buffer against new geometry can leave
		// stale glyphs at now-meaningless coordinates. Force a full repaint so
		// the next frame always redraws from a clean slate at the new size.
		this.resizeHandler = () => this.requestRender(true);
		process.stdout.on("resize", this.resizeHandler);

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
		if (this.resizeHandler) {
			process.stdout.removeListener("resize", this.resizeHandler);
			this.resizeHandler = null;
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

	requestRender(force = false, immediate = false): void {
		// Defer renders until we've entered the alternate screen buffer.
		// RequestRender during construction would output to stdout before
		// alt-screen + clear, overlapping with startup theme text.
		if (!this.started) return;
		if (force) {
			this.previousLines = [];
			this.previousCursorRow = -1;
			this.previousCursorCol = -1;
			this.previousCursorVisible = null;
		}
		if (immediate) this.renderImmediateRequested = true;
		if (this.renderRequested) {
			if (immediate && this.renderTimer) {
				clearTimeout(this.renderTimer);
				this.renderTimer = null;
				process.nextTick(() => this.scheduleRender());
			}
			return;
		}
		this.renderRequested = true;
		process.nextTick(() => this.scheduleRender());
	}

	/**
	 * Commit one frame synchronously at an interaction boundary. Use sparingly
	 * for Enter/loading acknowledgement; continuous streaming stays frame-paced.
	 */
	renderNow(): void {
		if (!this.started || this.stopped) return;
		if (this.renderTimer) {
			clearTimeout(this.renderTimer);
			this.renderTimer = null;
		}
		this.renderRequested = false;
		this.renderImmediateRequested = false;
		this.doRender();
		this.lastRenderFinishedAt = performance.now();
		if (this.renderRequested) this.scheduleRender();
	}

	private scheduleRender(): void {
		if (this.stopped || this.renderTimer || !this.renderRequested) return;
		// Pace from the end of the previous frame. Measuring from frame start made
		// expensive streaming frames schedule their successor immediately, creating
		// bursts of layout work and terminal writes that could starve input handling.
		const elapsed = performance.now() - this.lastRenderFinishedAt;
		const delay = this.renderImmediateRequested
			? 0
			: Math.max(0, TUI.MIN_RENDER_INTERVAL_MS - elapsed);
		this.renderTimer = setTimeout(() => {
			this.renderTimer = null;
			if (this.stopped || !this.renderRequested) return;
			this.renderRequested = false;
			this.renderImmediateRequested = false;
			this.doRender();
			this.lastRenderFinishedAt = performance.now();
			if (this.renderRequested) this.scheduleRender();
		}, delay);
	}

	// Lines moved per wheel tick. Kept small so a single notch reads as a
	// glide rather than a jump; fast spins still move proportionally further
	// because multiple ticks batched into one stdin chunk are coalesced below.
	private static readonly WHEEL_STEP = 2;

	private handleInput(data: string): void {
		const hasVisibleOverlay = this.overlayStack.some(entry => {
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
			let wheelColumn = 0;
			let wheelRow = 0;
			let m: RegExpExecArray | null;
			while ((m = re.exec(data)) !== null) {
				const btn = parseInt(m[1], 10);
				const column = parseInt(m[2], 10) - 1;
				const row = parseInt(m[3], 10) - 1;
				if (btn === 64) {
					net -= 1; // wheel up → older content
					wheelColumn = column;
					wheelRow = row;
				} else if (btn === 65) {
					net += 1; // wheel down → newer content
					wheelColumn = column;
					wheelRow = row;
				} else if (btn === 0 && m[4] === "M") {
					if (this.layoutRoot) {
						clicked =
							this.routeClick(column, row, hasVisibleOverlay) || clicked;
					} else if (
						!hasVisibleOverlay &&
						row >= 0 &&
						row < this._viewportHeight
					) {
						clicked =
							this.scrollableComponent?.handleMouse?.(column, row) === true ||
							clicked;
					}
				}
				consumed += m[0].length;
			}
			// Pure mouse chunk → apply coalesced scroll once, then stop.
			if (net !== 0 && consumed === data.length) {
				if (this.layoutRoot) {
					// ScrollView.scrollBy: positive moves toward the end (down),
					// negative toward the start (up) — same sign as `net` already
					// uses (wheel up is -1, wheel down is +1), so no negation here.
					const delta = net * TUI.WHEEL_STEP;
					this.routeWheel(wheelColumn, wheelRow, delta);
				} else if (net > 0) {
					this.scrollDown(net * TUI.WHEEL_STEP);
				} else {
					this.scrollUp(-net * TUI.WHEEL_STEP);
				}
				return;
			}
			if (clicked && consumed === data.length) return;
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
		const navBatch = data.match(/\x1b\[(?:5~|6~|1;5H|1;5F|H|F)/g);
		if (navBatch && navBatch.length > 1 && navBatch.join("") === data) {
			for (const key of navBatch) this.handleInput(key);
			return;
		}

		// Scroll keys are global in coding-agent TUIs: the transcript can move while
		// the prompt keeps focus. Plain arrows remain input/history navigation.
		// But if any overlay is visible, skip scrolling so the overlay gets first
		// crack at the keys (e.g. reasoner selector, plugin manager).
		const primaryScrollView = this.layoutRoot
			? this.currentLayoutFrame?.primaryScrollView
			: undefined;
		if (!hasVisibleOverlay && (this.scrollableComponent || primaryScrollView)) {
			const pageStep = Math.max(4, Math.floor(this._viewportHeight * 0.8));
			if (data === "\x1b[5~") {
				if (primaryScrollView) primaryScrollView.scrollBy(-pageStep);
				else this.scrollUp(pageStep);
				this.requestRender();
				return;
			}
			if (data === "\x1b[6~") {
				if (primaryScrollView) primaryScrollView.scrollBy(pageStep);
				else this.scrollDown(pageStep);
				this.requestRender();
				return;
			}
			if (
				data === "\x1b[1;5H" ||
				(data === "\x1b[H" && !this.isInputFocused())
			) {
				if (primaryScrollView) primaryScrollView.scrollToStart();
				else this.scrollToTop();
				this.requestRender();
				return;
			}
			if (
				data === "\x1b[1;5F" ||
				(data === "\x1b[F" && !this.isInputFocused())
			) {
				if (primaryScrollView) primaryScrollView.scrollToEnd();
				else this.scrollToBottom();
				this.requestRender();
				return;
			}

			// Handle arrow scrolling when not focused on input bar.
			const isInputFocused = this.focusedComponent === this.inputBarComponent;
			if (!isInputFocused) {
				if (data === "\x1b[A" || data === "\x1bOA") {
					/* Up arrow */
					if (primaryScrollView) primaryScrollView.scrollBy(-1);
					else this.scrollUp(1);
					this.requestRender();
					return;
				}
				if (data === "\x1b[B" || data === "\x1bOB") {
					/* Down arrow */
					if (primaryScrollView) primaryScrollView.scrollBy(1);
					else this.scrollDown(1);
					this.requestRender();
					return;
				}
				if (data === "\x1b[H" || data === "\x1bOH") {
					/* Home */
					if (primaryScrollView) primaryScrollView.scrollToStart();
					else this.scrollToTop();
					this.requestRender();
					return;
				}
				if (data === "\x1b[F" || data === "\x1bOF") {
					/* End */
					if (primaryScrollView) primaryScrollView.scrollToEnd();
					else this.scrollToBottom();
					this.requestRender();
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

		if (this.layoutRoot) {
			this._doRenderInnerLayoutEngine(
				this.layoutRoot,
				termWidth,
				termHeight,
				frameStartedAt,
			);
			return;
		}

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
			termHeight - 2 - aboveInputHeight - inputHeight - statusHeight,
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
		// Park the hardware cursor at the input's edit position (under the
		// visible InputBar cursor). Falls back to the input line's first column
		// only if no marker was emitted, which keeps the cursor off the footer.
		const fallbackRow = Math.min(
			termHeight,
			transcriptHeight + 2 + aboveInputHeight,
		);
		this._commitFrame(
			finalLines,
			termWidth,
			termHeight,
			frameStartedAt,
			fallbackRow,
		);
	}

	/**
	 * Diff `finalLines` against the last committed frame, write only the
	 * changed cells, park the hardware cursor at CURSOR_MARKER (or
	 * `fallbackCursorRow` if no component emitted one), and record render
	 * metrics. Shared tail for both the legacy fixed-region layout and the
	 * constrained layout engine — everything upstream of this differs only in
	 * how `finalLines` and `fallbackCursorRow` were produced.
	 */
	private _commitFrame(
		finalLines: string[],
		termWidth: number,
		termHeight: number,
		frameStartedAt: number,
		fallbackCursorRow: number,
	): void {
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
			const newLine =
				row < finalLines.length ? finalLines[row] : " ".repeat(termWidth);
			const hasMarker = newLine.includes(CURSOR_MARKER);

			// Extract cursor marker position before stripping. Done for every
			// row, even ones whose text didn't change, since the marker can
			// land on such a row while other rows moved.
			if (hasMarker) {
				const markerIdx = newLine.indexOf(CURSOR_MARKER);
				markerRow = row;
				markerCol = visibleWidth(newLine.slice(0, markerIdx));
			}

			// Most rows are untouched most frames (static scrollback, blank
			// padding, unrelated status regions). Skip the marker-strip
			// allocation and cell parse entirely when the raw row is identical
			// to last frame — a row carrying the marker but unchanged text
			// still stripped to the same string both times, so re-diffing it
			// would be a no-op anyway.
			if (prevLine === newLine) continue;

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

			// Reachable when the raw lines differed only in marker position —
			// stripping the marker made both sides equal, so there's nothing
			// left to diff.
			if (cleanPrev === cleanNew) continue;

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
		const cursorRow = markerRow >= 0 ? markerRow + 1 : fallbackCursorRow;
		const cursorCol = markerRow >= 0 ? Math.min(termWidth, markerCol + 1) : 1;
		const cursorMoved =
			changes.length > 0 ||
			cursorRow !== this.previousCursorRow ||
			cursorCol !== this.previousCursorCol;
		const cursorUpdate = cursorMoved ? `\x1b[${cursorRow};${cursorCol}H` : "";
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
		if (RENDER_DEBUG_ENABLED) logRenderMetrics(this.lastRenderMetrics);
	}

	private _doRenderInnerLayoutEngine(
		root: Component,
		termWidth: number,
		termHeight: number,
		frameStartedAt: number,
	): void {
		// _commitFrame never writes the physical last column (avoids terminal
		// autowrap on write) — see renderWidth there. Lay out one column
		// narrower so nothing, including a full-width ScrollView's scrollbar,
		// ever targets that reserved column; pad back out to termWidth after,
		// matching what the legacy fixed layout always produced.
		const layoutWidth = Math.max(1, termWidth - 1);
		let frame: LayoutFrame;
		try {
			frame = renderLayoutFrame(root, layoutWidth, termHeight, () =>
				this.requestRender(),
			);
		} catch (_e: unknown) {
			frame = renderLayoutFrame(
				new Spacer(termHeight),
				layoutWidth,
				termHeight,
				() => this.requestRender(),
			);
		}
		this.currentLayoutFrame = frame;
		this._viewportHeight =
			frame.primaryScrollView?.viewportHeight ?? termHeight;
		const paddedLines = frame.lines.map(line => `${line} `);

		// composeOverlays' transcriptHeight parameter only feeds center/bottom
		// anchor math for non-aboveInput overlays; the primary scroll view's
		// viewport height is the layout-engine equivalent of "the transcript
		// area" those overlays float over.
		const finalLines = this.composeOverlays(
			paddedLines,
			termWidth,
			termHeight,
			this._viewportHeight,
		);
		this._commitFrame(
			finalLines,
			termWidth,
			termHeight,
			frameStartedAt,
			termHeight,
		);
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
		this.paintedOverlayClickRects = [];

		const visibleEntries = this.overlayStack.filter(e => {
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

			if (entry.options?.onClick) {
				this.paintedOverlayClickRects.push({
					rect: {
						x: margin,
						y: row,
						width: overlayWidth,
						height: overlayHeight,
					},
					onClick: entry.options.onClick,
				});
			}
		}

		return result;
	}

	private renderAboveInputOverlays(termWidth: number): string[] {
		const entries = this.overlayStack.filter(entry => {
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
		return rendered.slice(0, maxHeight).map(line => {
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

	// ── Layout root ───────────────────────────────────────────────────────
	// When set, _doRenderInner() builds the frame through the constrained
	// layout engine (rendering/layout.ts) instead of the hand-assembled
	// transcript/separator/pinned/input/status block below. Regions are then
	// whatever Flex/ScrollView tree the caller composes.

	setLayoutRoot(component: Component | null): void {
		this.layoutRoot = component;
		this.currentLayoutFrame = null;
		this.requestRender();
	}

	private aboveInputOverlaysComponent: Component | undefined;

	/**
	 * Stable Component wrapping renderAboveInputOverlays() — the most-
	 * recently-focused `anchor: "aboveInput"` overlay (slash popup, file
	 * mention, plugin/MCP manager). Lets a layout-engine stack entry pick up
	 * the same "one active picker, most-recent-focus-wins" behavior the
	 * legacy fixed layout renders inline.
	 */
	getAboveInputOverlaysComponent(): Component {
		if (!this.aboveInputOverlaysComponent) {
			this.aboveInputOverlaysComponent = {
				render: (width: number) => this.renderAboveInputOverlays(width),
			};
		}
		return this.aboveInputOverlaysComponent;
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

	/**
	 * Route a wheel event through the committed layout frame: hit-test at
	 * (column, row), offer `delta` to each ScrollView found there deepest-first,
	 * and stop once one consumes it unless it opts into `overscroll: "chain"`.
	 * Falls back to the frame's primary ScrollView if nothing under the
	 * pointer consumed the delta — mirrors the legacy scrollUp/scrollDown
	 * behavior where wheel input always reaches the transcript even when the
	 * pointer sits over a non-scrollable dock row.
	 */
	private routeWheel(column: number, row: number, delta: number): void {
		const frame = this.currentLayoutFrame;
		if (!frame) return;
		const hitScrollViews = getScrollViewsAt(frame, column, row);
		let remaining = delta;
		const seen = new Set<unknown>();
		for (const scrollView of hitScrollViews) {
			seen.add(scrollView);
			remaining = scrollView.scrollBy(remaining);
			if (remaining === 0 || scrollView.overscroll === "contain") break;
		}
		const primary = frame.primaryScrollView;
		if (remaining !== 0 && primary && !seen.has(primary))
			primary.scrollBy(remaining);
		this.requestRender();
	}

	/**
	 * Route a left-click in layout-engine mode. Overlay click regions (e.g.
	 * the "new output below" indicator) are checked first and work regardless
	 * of what else is visible, since they're cosmetic, non-modal affordances.
	 * Otherwise, when no modal overlay is capturing input, hit-test the
	 * primary ScrollView's child for a `handleMouse` capability (mirrors the
	 * legacy scrollableComponent?.handleMouse? path) — the click row is
	 * translated to content-relative coordinates using that component's own
	 * painted box, since content now renders at full unbounded height and is
	 * clipped by the ScrollView rather than self-clipped.
	 */
	private routeClick(
		column: number,
		row: number,
		hasVisibleOverlay: boolean,
	): boolean {
		for (const { rect, onClick } of this.paintedOverlayClickRects) {
			if (
				column >= rect.x &&
				column < rect.x + rect.width &&
				row >= rect.y &&
				row < rect.y + rect.height
			) {
				onClick();
				this.requestRender();
				return true;
			}
		}
		if (hasVisibleOverlay) return false;
		const frame = this.currentLayoutFrame;
		const primary = frame?.primaryScrollView;
		if (!frame || !primary) return false;
		const child = primary.contentComponent;
		if (!("handleMouse" in child)) return false;
		const box = getComponentBoxAt(frame, child, column, row);
		if (!box) return false;
		// box.rect.y already bakes in the ScrollView's scroll offset (layout.ts
		// translates the child box to y - scrollTop), so the click row only
		// needs the box's own origin subtracted — adding scrollTop again would
		// double-count it and hit the wrong content row entirely.
		const contentRow = row - box.rect.y;
		const contentColumn = column - box.rect.x;
		const handled = (
			child as unknown as { handleMouse: (c: number, r: number) => boolean }
		).handleMouse(contentColumn, contentRow);
		if (handled) this.requestRender();
		return handled;
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
	/** Invoked when a mouse click lands within this overlay's composited
	 * screen rect. Only wired up in layout-engine mode (setLayoutRoot). */
	onClick?: () => void;
}
