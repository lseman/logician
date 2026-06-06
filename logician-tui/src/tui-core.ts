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
}

export function isFocusable(c: Component | null): c is Component & Focusable {
    return c !== null && "focused" in c;
}

// ── Cursor marker ────────────────────────────────────────────────────────────

export const CURSOR_MARKER = "\x1b_pi:c\x07";

// ── Width utilities ──────────────────────────────────────────────────────────

// Simple visible width calculator (handles ANSI escape codes)
export function visibleWidth(text: string): number {
    let width = 0;
    let inEscape = false;
    for (let i = 0; i < text.length; i++) {
        if (text[i] === "\x1b" && text[i + 1] === "[") {
            inEscape = true;
            i += 1;
        } else if (inEscape) {
            if (text[i] === "m" || text[i] === "H" || text[i] === "f") {
                inEscape = false;
            }
        } else {
            // Basic wide char detection (CJK ranges)
            const code = text.charCodeAt(i);
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
        }
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

// ── Terminal input ───────────────────────────────────────────────────────────
// Keyboard comes from process.stdin in raw mode (pi-style). The bridge child's
// events arrive on its own stdout pipe, so stdin is free for the keyboard.
import process from "node:process";
import { Buffer } from "node:buffer";

// ── TUI — Differential rendering ────────────────────────────────────────────

const SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07";

export class TUI extends Container {
    private previousLines: string[] = [];
    private previousWidth = 0;
    private previousHeight = 0;
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
    private scrollableComponent: Scrollable | null = null;
    private inputBarComponent: Component | null = null;
    private fixedBottomComponent: Component | null = null;
    private fixedAboveInputComponent: Component | null = null;

    private _showHardwareCursor = true;

    constructor(outStream: NodeJS.WriteStream, showCursor = true) {
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
            } catch {
                // raw mode unavailable (e.g. piped stdin) — keys won't work but the UI
                // still renders; degrade gracefully rather than crash.
            }
        }
        process.stdin.resume();
        this.stdinHandler = (data: string | Buffer) => {
            const str = Buffer.isBuffer(data) ? data.toString("utf-8") : data;
            this.handleInput(str);
            this.requestRender();
        };
        process.stdin.on("data", this.stdinHandler);

        // Enter alternate screen buffer + hide cursor + enable bracketed paste.
        // The alt screen gives us a fixed canvas to redraw each frame from the
        // home position. Bracketed paste makes the terminal wrap pasted text in
        // \x1b[200~ … \x1b[201~ so the app can distinguish paste from typed input.
        this.write("\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l\x1b[?2004h");

        this.requestRender(true);
    }

    stop(): void {
        this.stopped = true;
        if (this.renderTimer) {
            clearTimeout(this.renderTimer);
            this.renderTimer = null;
        }
        // Show cursor + leave alternate screen + disable bracketed paste,
        // restoring the user's terminal.
        this.write("\x1b[?25h\x1b[?1049l\x1b[?2004l");

        if (this.stdinHandler) {
            process.stdin.removeListener("data", this.stdinHandler);
            this.stdinHandler = null;
        }
        if (process.stdin.setRawMode) {
            try {
                process.stdin.setRawMode(this.wasRaw);
            } catch {
                // ignore
            }
        }
        process.stdin.pause();
    }

    requestRender(force = false): void {
        if (force) {
            this.previousLines = [];
            this.previousWidth = -1;
            this.previousHeight = -1;
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
        // Handle SGR mouse events: \x1b[<button;column;row(M|m). A single stdin read
        // can batch several wheel ticks; coalesce them into one scroll so fast wheel
        // spins move proportionally without queuing a render per tick.
        if (data.startsWith("\x1b[<")) {
            const re = /\x1b\[<(\d+);\d+;\d+[Mm]/g;
            let net = 0; // +down / -up, in wheel ticks
            let consumed = 0;
            let m: RegExpExecArray | null;
            while ((m = re.exec(data)) !== null) {
                const btn = parseInt(m[1], 10);
                if (btn === 64)
                    net -= 1; // wheel up → older content
                else if (btn === 65) net += 1; // wheel down → newer content
                consumed += m[0].length;
            }
            // Pure mouse chunk → apply coalesced scroll once, then stop.
            if (net !== 0 && consumed === data.length) {
                if (net > 0) this.scrollDown(net * TUI.WHEEL_STEP);
                else this.scrollUp(-net * TUI.WHEEL_STEP);
                return;
            }
            if (consumed === data.length) return; // mouse-only chunk, nothing to scroll
        }

        // Scroll keys are global in coding-agent TUIs: the transcript can move while
        // the prompt keeps focus. Plain arrows remain input/history navigation.
        if (this.scrollableComponent) {
            if (data === "\x1b[5~") {
                this.scrollUp(
                    Math.max(4, Math.floor(this._viewportHeight * 0.8)),
                );
                return;
            }
            if (data === "\x1b[6~") {
                this.scrollDown(
                    Math.max(4, Math.floor(this._viewportHeight * 0.8)),
                );
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
        }

        // Handle arrow scrolling when not focused on input bar.
        const isInputFocused = this.focusedComponent === this.inputBarComponent;
        if (!isInputFocused && this.scrollableComponent) {
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
        for (const listener of this.inputListeners) {
            const result = listener(data);
            if (result?.consume) return;
        }
        if (this.focusedComponent && "handleInput" in this.focusedComponent) {
            (this.focusedComponent as any).handleInput(data);
        }
    }

    private doRender(): void {
        if (this.stopped) return;
        const termWidth = Math.max(1, process.stdout.columns || 80);
        const termHeight = Math.max(1, process.stdout.rows || 24);

        const inputLines = this.inputBarComponent
            ? this.inputBarComponent.render(termWidth)
            : [" ".repeat(termWidth)];
        const statusLines = this.fixedBottomComponent
            ? this.fixedBottomComponent.render(termWidth)
            : [" ".repeat(termWidth)];
        const inputHeight = Math.max(1, inputLines.length);
        const statusHeight = Math.max(1, statusLines.length);

        // Optional pinned region above the input bar (todo list). Zero lines = hidden.
        const aboveInputLines = this.fixedAboveInputComponent
            ? this.fixedAboveInputComponent.render(termWidth)
            : [];
        const aboveInputHeight = aboveInputLines.length;

        // Fixed layout: transcript + divider + [pinned] + input bar + divider + status footer.
        const transcriptHeight = Math.max(
            1,
            termHeight - 2 - aboveInputHeight - inputHeight - statusHeight,
        );
        const transcriptWidth = termWidth;

        // Build output lines with fixed layout
        const lines: string[] = [];

        // 1. Transcript area (scrollable)
        if (this.scrollableComponent) {
            (this.scrollableComponent as any).setViewportHeight(
                transcriptHeight,
            );
            this._viewportHeight = transcriptHeight;
            const transcriptLines =
                this.scrollableComponent.render(transcriptWidth);
            const totalLines = Math.max(
                transcriptLines.length,
                this.scrollableComponent.totalHeight,
            );
            const maxScroll = Math.max(0, totalLines - transcriptHeight);
            // Use scrollable component's scrollOffset (set during render by scrollToBottom)
            const comp = this.scrollableComponent as Scrollable;
            const scrollOff = Math.min(
                maxScroll,
                Math.max(0, comp.scrollOffset),
            );
            const visibleLines = (comp as any).rendersViewport
                ? transcriptLines
                : transcriptLines.slice(
                      scrollOff,
                      scrollOff + transcriptHeight,
                  );

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
        lines.push("\x1b[38;5;236m" + "─".repeat(termWidth) + "\x1b[0m");

        // 2b. Pinned region above input (todo list), when present
        for (let i = 0; i < aboveInputHeight; i++) {
            lines.push(aboveInputLines[i] || " ".repeat(termWidth));
        }

        // 3. Input bar (fixed)
        for (let i = 0; i < inputHeight; i++) {
            lines.push(inputLines[i] || " ".repeat(termWidth));
        }

        // 4. Separator line below input
        lines.push("\x1b[38;5;236m" + "─".repeat(termWidth) + "\x1b[0m");

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

        // Full redraw from the home position. The alternate screen buffer means we
        // own the whole canvas, so a synchronized home-and-repaint is both correct
        // and flicker-free — no fragile cursor arithmetic, no scrollback bleed.
        let buffer = "\x1b[?2026h"; // begin synchronized update
        buffer += "\x1b[H"; // home

        // The InputBar marks the edit position with CURSOR_MARKER. Find it so we
        // can park the hardware cursor exactly there, and strip it from output.
        let markerRow = -1;
        let markerCol = 0;

        for (let i = 0; i < termHeight; i++) {
            if (i > 0) buffer += "\r\n";
            buffer += "\x1b[2K"; // clear the whole line
            if (i < finalLines.length) {
                // Hard-truncate short of the physical last column. Many terminals set a
                // pending autowrap state when the last cell is written; during rapid
                // scroll/redraw transitions that can leave stale doubled fragments.
                let ln = finalLines[i];
                const markerIdx = ln.indexOf(CURSOR_MARKER);
                if (markerIdx >= 0) {
                    markerRow = i;
                    markerCol = visibleWidth(ln.slice(0, markerIdx));
                    ln = ln.replace(CURSOR_MARKER, "");
                }
                buffer += isImageLine(ln)
                    ? ln
                    : clampLineToWidth(ln, Math.max(1, termWidth - 1));
            }
        }

        buffer += "\x1b[?2026l"; // end synchronized update
        this.write(buffer);

        this.previousLines = finalLines;
        this.previousWidth = termWidth;
        this.previousHeight = termHeight;

        // Park the hardware cursor at the input's edit position (under the
        // visible InputBar cursor). Falls back to the input line's first column
        // only if no marker was emitted, which keeps the cursor off the footer.
        const fallbackRow = Math.min(
            termHeight,
            transcriptHeight + 2 + aboveInputHeight,
        );
        const cursorRow = markerRow >= 0 ? markerRow + 1 : fallbackRow;
        const cursorCol = markerRow >= 0 ? markerCol + 1 : 1;
        this.write(`\x1b[${cursorRow};${cursorCol}H`);
        this.write(this._showHardwareCursor ? "\x1b[?25h" : "\x1b[?25l");
    }

    private composeOverlays(
        lines: string[],
        termWidth: number,
        termHeight: number,
        transcriptHeight: number,
    ): string[] {
        const result = [...lines];

        const visibleEntries = this.overlayStack.filter((e) => {
            if (e.hidden) return false;
            // Also check component's visible property if it has one
            if (
                "visible" in e.component &&
                typeof (e.component as any).visible === "boolean"
            ) {
                return (e.component as any).visible;
            }
            return true;
        });

        for (const entry of visibleEntries) {
            const leftAligned = entry.options?.align === "left";
            const overlayWidth = leftAligned
                ? Math.max(40, termWidth - 1)
                : Math.max(
                      40,
                      Math.min(
                          termWidth - 8,
                          entry.options?.maxHeight
                              ? termWidth * 0.6
                              : termWidth - 8,
                      ),
                  );
            const overlayLines = entry.component.render(
                Math.max(1, overlayWidth),
            );
            const overlayHeight = Math.min(
                overlayLines.length,
                entry.options?.maxHeight || 999,
            );

            // Calculate row position within transcript area
            let row = 0;
            switch (entry.options?.anchor) {
                case "center":
                    row = Math.max(
                        0,
                        Math.floor((transcriptHeight - overlayHeight) / 2),
                    );
                    break;
                case "bottom":
                    // Flush against the bottom of the transcript area, i.e. directly
                    // above the separator + input bar.
                    row = Math.max(0, transcriptHeight - overlayHeight);
                    break;
                case "top":
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
                    const afterPad = " ".repeat(
                        Math.max(0, termWidth - margin - srcVis),
                    );
                    result[idx] = basePad + srcLine + afterPad;
                }
            }
        }

        return result;
    }

    private write(data: string): void {
        try {
            process.stdout.write(data);
        } catch {
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

// ── Overlay options ──────────────────────────────────────────────────────────

export interface OverlayOptions {
    anchor?: "center" | "top" | "bottom";
    align?: "center" | "left";
    maxHeight?: number;
    nonCapturing?: boolean;
}
