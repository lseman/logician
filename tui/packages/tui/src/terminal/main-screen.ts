// ── Main-screen TUI ──────────────────────────────────────────────────────────
// Sibling renderer to TUI (core.ts) that never enters the alternate screen
// buffer. It writes append-only into the terminal's main screen and lets the
// terminal's own scrollback hold history, instead of owning a fixed-height
// viewport that gets repainted every frame. There is no scroll/page/overlay-
// anchor concept tied to a viewport here — the terminal's native scrollback
// and mouse wheel do that job, so this class never enables mouse capture.
//
// Deliberately NOT sharing an extracted base class with TUI: the two modes'
// render loops are different enough (fixed-viewport cell-diff vs. append-only
// line-diff with relative cursor movement) that a shared base would mostly
// hold indirection. The public surface here matches what app/tui.ts and the
// rest of app/*.ts actually call on `TUI` today (verified by grep): start,
// stop, requestRender, renderNow, setFocus, showOverlay, hideOverlay,
// removeOverlay, addInputListener, setInputBarComponent (no-op),
// setLayoutRoot (no-op), getAboveInputOverlaysComponent, enableMouse (no-op),
// disableMouse (no-op).

import { Buffer } from "node:buffer";
import {
	type Component,
	Container,
	CURSOR_MARKER,
	isFocusable,
	visibleWidth,
} from "./primitives.ts";
import {
	normalizeKeyboardInput,
	parseSizeValue,
	type OverlayHandle,
	type OverlayOptions,
} from "./core.ts";

function isImageLine(line: string): boolean {
	return line.includes("\x1b_G") || line.includes("\x1b]1337;");
}

// Strip any trailing SGR/OSC state a line forgot to reset, so a whole-line
// rewrite never bleeds color into unrelated content above or below it in
// scrollback (append-only mode has no per-cell diff to contain this).
function applyLineReset(line: string): string {
	if (line.includes(CURSOR_MARKER)) return `${line}\x1b[0m`;
	return line;
}

export class TuiMainScreen extends Container {
	private renderRequested = false;
	private renderImmediateRequested = false;
	private renderTimer: ReturnType<typeof setTimeout> | null = null;
	private lastRenderFinishedAt = 0;
	private static readonly IDLE_RENDER_INTERVAL_MS = 16;
	private static readonly STREAMING_RENDER_INTERVAL_MS = 33;
	private isStreaming = false;
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

	// ── Append-only render state ─────────────────────────────────────────────
	private previousLines: string[] = [];
	private previousWidth = 0;
	private previousHeight = 0;
	private hardwareCursorRow = 0;
	private maxLinesRendered = 0;
	// Topmost row still reachable with relative cursor movement. Content
	// above this has already scrolled off the physical screen into the
	// terminal's native scrollback — a real cursor-up escape clamps at the
	// screen's top row rather than reaching further, so patching a change
	// there in place would silently misfire and stomp whatever's currently
	// visible. A change at or above this row must fall back to a fresh
	// reprint instead (see firstChanged < previousViewportTop below).
	private previousViewportTop = 0;

	private _showHardwareCursor = true;

	constructor(showCursor = true) {
		super();
		this._showHardwareCursor = showCursor;
	}

	readonly mode = "regular" as const;

	setShowHardwareCursor(enabled: boolean): void {
		this._showHardwareCursor = enabled;
		this.requestRender();
	}

	/** Set whether the transcript is actively streaming.
	 * During streaming, render interval increases from 16ms to 33ms
	 * (60fps → 30fps) to halve layout work when only 1-2 lines change.
	 * Switches back to 60fps for idle/spinner/key-interaction smoothness. */
	setIsStreaming(isStreaming: boolean): void {
		this.isStreaming = isStreaming;
	}

	setFocus(component: Component | null): void {
		if (isFocusable(this.focusedComponent)) {
			this.focusedComponent.focused = false;
		}
		this.focusedComponent = component;
		if (isFocusable(component)) {
			component.focused = true;
		}
	}

	// setLayoutRoot / setInputBarComponent are no-ops here: main-screen mode
	// always renders the flat child list (this.render()) and has no fixed
	// viewport for a layout-engine tree to be clipped against. app/tui.ts's
	// buildLayout() flat-mounts components directly onto this instance
	// instead of installing a layout root when this mode is active.
	setLayoutRoot(_component: Component | null): void {}
	setInputBarComponent(_component: Component | null): void {}

	private aboveInputOverlaysComponent: Component | undefined;

	getAboveInputOverlaysComponent(): Component {
		if (!this.aboveInputOverlaysComponent) {
			this.aboveInputOverlaysComponent = {
				render: (width: number) => this.renderAboveInputOverlays(width),
			};
		}
		return this.aboveInputOverlaysComponent;
	}

	showOverlay(
		component: Component,
		options?: OverlayOptions,
	): OverlayHandle {
		const entry = {
			component,
			options,
			preFocus: this.focusedComponent,
			hidden: false,
			focusOrder: ++this.focusOrderCounter,
		};
		this.overlayStack.push(entry);
		if (!options?.nonCapturing && this.isOverlayVisible(entry)) {
			this.setFocus(component);
		}
		this.requestRender();

		return {
			hide: () => {
				const idx = this.overlayStack.indexOf(entry);
				if (idx >= 0) {
					this.overlayStack.splice(idx, 1);
					if (this.focusedComponent === component) {
						this.setFocus(entry.preFocus);
					}
				}
				this.requestRender();
			},
			setHidden: (hidden: boolean) => {
				entry.hidden = hidden;
				this.requestRender();
			},
			isHidden: () => entry.hidden,
			isFocused: () => this.focusedComponent === component,
			focus: () => {
				this.setFocus(component);
				entry.focusOrder = ++this.focusOrderCounter;
				this.requestRender();
			},
			unfocus: () => {
				this.setFocus(entry.preFocus);
				this.requestRender();
			},
		};
	}

	hideOverlay(): void {
		const overlay = this.overlayStack[this.overlayStack.length - 1];
		if (overlay) this.overlayStack.pop();
		if (this.focusedComponent === overlay?.component) {
			this.setFocus(overlay?.preFocus);
		}
		this.requestRender();
	}

	private isOverlayVisible(entry: {
		component: Component;
		options?: OverlayOptions;
		hidden: boolean;
	}): boolean {
		if (entry.hidden) return false;
		// Components like the slash popup / file mention popup / plugin manager
		// stay mounted as overlays for the whole session but only actually show
		// content once invoked; their own `visible` flag is the real signal.
		if (
			"visible" in entry.component &&
			typeof (entry.component as { visible?: unknown }).visible === "boolean"
		) {
			return (entry.component as { visible: boolean }).visible;
		}
		if (entry.options?.visible) {
			return true;
		}
		return true;
	}

	removeOverlay(component: Component): void {
		const idx = this.overlayStack.findIndex(e => e.component === component);
		if (idx >= 0) {
			const entry = this.overlayStack[idx];
			this.overlayStack.splice(idx, 1);
			if (
				"visible" in entry.component &&
				typeof (entry.component as { visible?: unknown }).visible ===
					"boolean"
			) {
				(entry.component as { visible: boolean }).visible = false;
			}
			if (isFocusable(entry.preFocus)) {
				entry.preFocus.focused = false;
			}
			this.setFocus(entry.preFocus);
			this.requestRender();
		}
	}

	private renderAboveInputOverlays(termWidth: number): string[] {
		const entries = this.overlayStack.filter(entry => {
			if (entry.hidden || entry.options?.anchor !== "aboveInput") return false;
			if (entry.options?.visible && !entry.options.visible(termWidth, 1000)) {
				return false;
			}
			if (
				"visible" in entry.component &&
				typeof (entry.component as { visible?: unknown }).visible ===
					"boolean"
			) {
				return (entry.component as { visible: boolean }).visible;
			}
			return true;
		});
		if (entries.length === 0) return [];

		const entry = entries.reduce((latest, candidate) =>
			candidate.focusOrder > latest.focusOrder ? candidate : latest,
		);
		const width = Math.max(1, termWidth - 1);
		const rendered = entry.component.render(width);
		const maxHeight = parseSizeValue(entry.options?.maxHeight, 200) ?? rendered.length;
		return rendered.slice(0, maxHeight);
	}

	// Non-"aboveInput" overlays (center/bottom-anchored) have no coherent
	// position in an unbounded, append-only scrollback — there is no fixed
	// "bottom of the transcript" to anchor against. Skip them entirely rather
	// than guess a placement; only aboveInput overlays (slash popup, file
	// mention, plugin/MCP manager) render in this mode, appended after the
	// dock like everything else.

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

		this.wasRaw = process.stdin.isRaw ?? false;
		process.stdin.setEncoding("utf-8");
		if (process.stdin.setRawMode) {
			try {
				process.stdin.setRawMode(true);
			} catch (_e: unknown) {
				// raw mode unavailable (e.g. piped stdin) — keys won't work but the
				// UI still renders; degrade gracefully rather than crash.
			}
		}
		process.stdin.resume();

		// A resize can leave previousLines addressed against stale geometry —
		// force a full repaint (which for main-screen mode means: stop trying
		// relative cursor math against old wrapping and just print fresh from
		// here, same as pi's TuiMainScreen width/height-change handling).
		this.resizeHandler = () => {
			this.previousLines = [];
			this.previousWidth = 0;
			this.previousHeight = 0;
			this.requestRender(true);
		};
		process.stdout.on("resize", this.resizeHandler);

		// No alt-screen enter, no clear, no pre-scroll. Bracketed paste still
		// helps the input bar distinguish pasted text from typed input.
		// Render fresh from wherever the cursor already is, same as pi —
		// prior shell output stays exactly where it is, on screen and in
		// scrollback, untouched.
		process.stdout.write("\x1b[?2004h\x1b[>1u");

		this.stdinHandler = (data: string | Buffer) => {
			const str = Buffer.isBuffer(data) ? data.toString("utf-8") : data;
			// Kitty's disambiguate-escape-codes mode (enabled below via
			// \x1b[>1u) reports plain Escape as CSI 27u instead of a bare
			// 0x1b — without translating it back, every `data === "\x1b"`
			// check in the app (Esc-to-cancel, popup dismissal, etc.)
			// silently never matches.
			this.handleInput(normalizeKeyboardInput(str));
			this.requestRender(false, true);
		};
		process.stdin.on("data", this.stdinHandler);
		this.requestRender(true);
	}

	stop(): void {
		this.stopped = true;
		if (this.renderTimer) {
			clearTimeout(this.renderTimer);
			this.renderTimer = null;
		}

		// Leave the cursor below the last rendered line so the restored shell
		// prompt doesn't land mid-line.
		if (this.previousLines.length > 0) {
			const targetRow = this.previousLines.length;
			const lineDiff = targetRow - this.hardwareCursorRow;
			if (lineDiff > 0) process.stdout.write(`\x1b[${lineDiff}B`);
			else if (lineDiff < 0) process.stdout.write(`\x1b[${-lineDiff}A`);
			process.stdout.write("\r\n");
		}

		process.stdout.write("\x1b[<u\x1b[?25h\x1b[?2004l");

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

	// enableMouse/disableMouse are no-ops: main-screen mode leaves the
	// terminal's own mouse-wheel/scrollback handling untouched rather than
	// capturing SGR mouse events for an app-owned viewport that doesn't exist
	// here.
	enableMouse(): void {}
	disableMouse(): void {}

	requestRender(force = false, immediate = false): void {
		if (!this.started) return;
		if (force) {
			this.previousLines = [];
			this.previousWidth = 0;
			this.previousHeight = 0;
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
		const elapsed = performance.now() - this.lastRenderFinishedAt;
		const interval = this.isStreaming
			? TuiMainScreen.STREAMING_RENDER_INTERVAL_MS
			: TuiMainScreen.IDLE_RENDER_INTERVAL_MS;
		const delay = this.renderImmediateRequested
			? 0
			: Math.max(0, interval - elapsed);
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

	private handleInput(data: string): void {
		const hasVisibleOverlay = this.overlayStack.some(entry => {
			if (entry.hidden) return false;
			if (entry.options?.visible && !entry.options.visible(80, 24)) {
				return false;
			}
			if (
				"visible" in entry.component &&
				typeof (entry.component as { visible?: unknown }).visible ===
					"boolean"
			) {
				return (entry.component as { visible: boolean }).visible;
			}
			return true;
		});

		for (const listener of this.inputListeners) {
			const result = listener(data);
			if (result?.consume) return;
		}
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
		if (hasVisibleOverlay) {
			// Overlay handling above already had first crack via
			// addInputListener-registered listeners; nothing further to route.
		}
		if (this.focusedComponent && "handleInput" in this.focusedComponent) {
			(
				this.focusedComponent as { handleInput: (data: string) => void }
			).handleInput(data);
		}
	}

	// ── Render ────────────────────────────────────────────────────────────

	private doRender(): void {
		if (this.stopped) return;
		try {
			this._doRender();
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			this.previousLines = [];
			process.stderr.write(
				"\x1b]8;;\x1b\\\x1b[0m\x1b[?25h" +
					`\n\x1b[38;5;203m[TUI render error]\x1b[0m ${msg}\n`,
			);
			// eslint-disable-next-line no-console
			console.error("TUI render crash:", err);
		}
	}

	private extractCursorMarker(
		lines: string[],
	): { row: number; col: number } | null {
		for (let row = 0; row < lines.length; row++) {
			const idx = lines[row].indexOf(CURSOR_MARKER);
			if (idx !== -1) {
				return { row, col: visibleWidth(lines[row].slice(0, idx)) };
			}
		}
		return null;
	}

	private stripCursorMarker(lines: string[]): string[] {
		return lines.map(line =>
			line.includes(CURSOR_MARKER) ? line.replaceAll(CURSOR_MARKER, "") : line,
		);
	}

	private _doRender(): void {
		const width = Math.max(1, process.stdout.columns || 80);
		const height = Math.max(1, process.stdout.rows || 24);
		const widthChanged = this.previousWidth !== 0 && this.previousWidth !== width;
		const heightChanged =
			this.previousHeight !== 0 && this.previousHeight !== height;

		// aboveInput overlays (slash popup, file mention, plugin/MCP manager)
		// render through getAboveInputOverlaysComponent(), which app/tui.ts
		// mounts as a regular child in its correct document position — no
		// separate append here, or the popup would render twice (once in
		// place, once tacked onto the end after the status bar).
		let newLines = this.render(width);

		const cursorPos = this.extractCursorMarker(newLines);
		newLines = this.stripCursorMarker(newLines).map(applyLineReset);

		const write = (data: string): void => {
			try {
				process.stdout.write(data);
			} catch (_e: unknown) {
				// ignore
			}
		};

		const positionCursor = (): void => {
			if (!cursorPos || newLines.length === 0) {
				write("\x1b[?25l");
				return;
			}
			const targetRow = Math.max(
				0,
				Math.min(cursorPos.row, newLines.length - 1),
			);
			const rowDelta = targetRow - this.hardwareCursorRow;
			let buffer = "";
			if (rowDelta > 0) buffer += `\x1b[${rowDelta}B`;
			else if (rowDelta < 0) buffer += `\x1b[${-rowDelta}A`;
			buffer += `\x1b[${Math.max(0, cursorPos.col) + 1}G`;
			write(buffer);
			this.hardwareCursorRow = targetRow;
			write(this._showHardwareCursor ? "\x1b[?25h" : "\x1b[?25l");
		};

		const fullRender = (): void => {
			let buffer = "\x1b[?2026h";
			for (let i = 0; i < newLines.length; i++) {
				if (i > 0) buffer += "\r\n";
				buffer += newLines[i];
			}
			buffer += "\x1b[?2026l";
			write(buffer);
			this.hardwareCursorRow = Math.max(0, newLines.length - 1);
			this.maxLinesRendered = newLines.length;
			this.previousLines = newLines;
			this.previousWidth = width;
			this.previousHeight = height;
			this.previousViewportTop = Math.max(0, newLines.length - height);
			positionCursor();
		};

		// First render, or geometry changed: print fresh from here. There is
		// no clear — whatever the terminal already has above stays in
		// scrollback untouched.
		if (this.previousLines.length === 0 || widthChanged || heightChanged) {
			fullRender();
			return;
		}

		let firstChanged = -1;
		let lastChanged = -1;
		const maxLines = Math.max(newLines.length, this.previousLines.length);
		for (let i = 0; i < maxLines; i++) {
			const oldLine = i < this.previousLines.length ? this.previousLines[i] : "";
			const newLine = i < newLines.length ? newLines[i] : "";
			if (oldLine !== newLine) {
				if (firstChanged === -1) firstChanged = i;
				lastChanged = i;
			}
		}

		if (firstChanged === -1) {
			positionCursor();
			return;
		}

		// The changed row has already scrolled off the physical screen into
		// the terminal's native scrollback — relative cursor movement can only
		// reach what's still on-screen, so patching in place would clamp at
		// the top row and stomp whatever's currently visible (this is what
		// large content, e.g. /context, used to trip: a change far back in a
		// long transcript looked like the whole screen got wiped). Print
		// fresh from here instead.
		if (firstChanged < this.previousViewportTop) {
			fullRender();
			return;
		}

		const appendedLines = newLines.length > this.previousLines.length;
		if (appendedLines && firstChanged === this.previousLines.length) {
			lastChanged = newLines.length - 1;
		}

		let buffer = "\x1b[?2026h";
		const lineDiff = firstChanged - this.hardwareCursorRow;
		if (lineDiff > 0) buffer += `\x1b[${lineDiff}B`;
		else if (lineDiff < 0) buffer += `\x1b[${-lineDiff}A`;
		buffer += "\r";

		const renderEnd = Math.min(lastChanged, newLines.length - 1);
		for (let i = firstChanged; i <= renderEnd; i++) {
			if (i > firstChanged) buffer += "\r\n";
			const line = newLines[i];
			if (!isImageLine(line) && visibleWidth(line) > width) {
				this.stop();
				throw new Error(
					`Rendered line ${i} exceeds terminal width (${visibleWidth(line)} > ${width}). ` +
						"A custom TUI component didn't truncate its output — use visibleWidth()/clampLineToWidth().",
				);
			}
			buffer += "\x1b[2K";
			buffer += line;
		}

		let finalCursorRow = renderEnd;
		if (this.previousLines.length > newLines.length) {
			if (renderEnd < newLines.length - 1) {
				const moveDown = newLines.length - 1 - renderEnd;
				buffer += `\x1b[${moveDown}B`;
				finalCursorRow = newLines.length - 1;
			}
			const extraLines = this.previousLines.length - newLines.length;
			for (let i = newLines.length; i < this.previousLines.length; i++) {
				buffer += "\r\n\x1b[2K";
			}
			buffer += `\x1b[${extraLines}A`;
		}

		buffer += "\x1b[?2026l";
		write(buffer);

		this.hardwareCursorRow = finalCursorRow;
		this.maxLinesRendered = Math.max(this.maxLinesRendered, newLines.length);
		this.previousLines = newLines;
		this.previousWidth = width;
		this.previousHeight = height;
		positionCursor();
	}
}
