// ── Slash command popup ───────────────────────────────────────────────────────
// Overlay popup with fuzzy matching, usage hints, and Tab completion.

import { visibleWidth, type Component } from "../tui-core.ts";
import { filterSlashCommands, type SlashCommandDef } from "../slash-commands.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const HEADER_COLOR = "\x1b[38;5;159m"; // aqua
const SELECTED_COLOR = "\x1b[38;5;111m"; // green


export class SlashPopup implements Component {
    private commands: SlashCommandDef[] = [];
    private query = "";
    private selectedIndex = 0;
    private width = 70;
    public visible = false;
    private cachedLines: string[] | null = null;
    private cachedWidth = -1;
    private onSubmit?: (
        result: string | null,
        dispatch?: "quit",
        command?: string,
    ) => void;

    setCommands(commands: SlashCommandDef[]): void {
        this.commands = commands;
        this.selectedIndex = 0;
        this.invalidate();
    }

    setQuery(query: string): void {
        if (this.query === query) return;
        this.query = query;
        // Keep selection in range as the filtered list shrinks/grows.
        const n = this._getFiltered().length;
        if (this.selectedIndex >= n) this.selectedIndex = Math.max(0, n - 1);
        this.invalidate();
    }

    isVisibleOverlay(): boolean {
        return this.visible;
    }

    // ── Inline-autocomplete navigation (driven by the input bar) ─────────────

    moveSelection(delta: number): void {
        const n = this._getFiltered().length;
        if (n === 0) return;
        this.selectedIndex = (this.selectedIndex + delta + n) % n;
        this.invalidate();
    }

    /** Command string of the highlighted row, or null when no match. */
    currentCommand(): string | null {
        const filtered = this._getFiltered();
        return filtered.length > 0
            ? filtered[this.selectedIndex].command
            : null;
    }

    hasMatches(): boolean {
        return this._getFiltered().length > 0;
    }

    show(): void {
        this.visible = true;
        this.invalidate();
    }

    hide(): void {
        this.visible = false;
        this.query = "";
        this.invalidate();
    }

    handleInput(data: string): void {
        if (data === "\r" || data === "\n") {
            this._submit();
            return;
        }

        if (data === "\x1b" || data === "\x03") {
            this.hide();
            return;
        }

        if (data === "\x08" || data === "\x7f") {
            this.query = this.query.slice(0, -1);
            this.selectedIndex = 0;
            this.invalidate();
            return;
        }

        // Tab — accept current selection or complete to first match
        if (data === "\t") {
            const filtered = this._getFiltered();
            if (filtered.length > 0) {
                this.query = filtered[this.selectedIndex].command + " ";
                this.selectedIndex = 0;
            }
            this.invalidate();
            return;
        }

        // BackTab — previous item
        if (data === "\x1b[Z") {
            const filtered = this._getFiltered();
            if (filtered.length > 0) {
                this.selectedIndex =
                    (this.selectedIndex - 1 + filtered.length) %
                    filtered.length;
                this.invalidate();
            }
            return;
        }

        // Up arrow
        if (data === "\x1b[A" || data === "\x1bOA") {
            const filtered = this._getFiltered();
            if (filtered.length > 0) {
                this.selectedIndex =
                    (this.selectedIndex - 1 + filtered.length) %
                    filtered.length;
                this.invalidate();
            }
            return;
        }

        // Down arrow
        if (data === "\x1b[B" || data === "\x1bOB") {
            const filtered = this._getFiltered();
            if (filtered.length > 0) {
                this.selectedIndex = (this.selectedIndex + 1) % filtered.length;
                this.invalidate();
            }
            return;
        }

        // Printable character
        if (data.length === 1) {
            const c = data.charCodeAt(0);
            if (c >= 0x20 && c < 0x7f) {
                this.query += data;
                this.invalidate();
            }
        }
    }

    private _getFiltered(): SlashCommandDef[] {
        return filterSlashCommands(this.commands, this.query);
    }

    private _submit(): void {
        const filtered = this._getFiltered();
        let dispatchType: "quit" | undefined;
        if (filtered.length > 0) {
            const cmd = filtered[this.selectedIndex];
            const args = this.query.replace(/^\/\w+\s*/, "").trim();
            this._lastCommand = this.query;
            if (cmd.handler) {
                const result = cmd.handler(args);
                if (result) {
                    this._lastResult = String(result);
                }
            }
            if (cmd.bridgeHandler) {
                cmd.bridgeHandler(args);
            }
            dispatchType = cmd.dispatch === "quit" ? "quit" : undefined;
        }
        // Notify TUI about dispatch actions
        if (dispatchType === "quit" && this.onSubmit) {
            this.onSubmit(null, "quit", this._lastCommand);
        } else if (this._lastResult && this.onSubmit) {
            this.onSubmit(this._lastResult, undefined, this._lastCommand);
        }
        this.hide();
    }

    setOnSubmit(
        cb: (
            result: string | null,
            dispatch?: "quit",
            command?: string,
        ) => void,
    ) {
        this.onSubmit = cb;
    }

    _lastResult: string | null = null;
    _lastCommand: string = "";

    getLastCommand(): string {
        return this._lastCommand;
    }

    invalidate(): void {
        this.cachedLines = null;
    }

    render(width: number): string[] {
        if (width === this.cachedWidth && this.cachedLines !== null) {
            return this.cachedLines;
        }

        this.cachedWidth = width;
        this.width = width;

        if (!this.visible) return [];

        const filtered = this._getFiltered();
        const contentWidth = Math.min(80, Math.max(40, width - 4));
        const lines: string[] = [];

        // Title row — the typed text already shows in the input bar below, so here
        // we just label the menu and show the match count plus key hints.
        const count = filtered.length;
        const hint = `${DIM}↑↓ select · Tab complete · ⏎ run · Esc close${RESET}`;
        lines.push(
            ` ${HEADER_COLOR}commands${RESET}${DIM} (${count})${RESET}  ${hint}`,
        );

        // Command list
        for (let i = 0; i < filtered.length; i++) {
            const cmd = filtered[i];
            const isSelected = i === this.selectedIndex;
            const prefix = isSelected ? "▸ " : "  ";

            // Split command name and usage for alignment
            const parts = cmd.usage.split(" ");
            const cmdName = parts[0] || cmd.command;
            const usageSuffix = parts.slice(1).join(" ");

            let line = isSelected
                ? ` ${SELECTED_COLOR}${prefix}${BOLD}${cmdName}${RESET}${SELECTED_COLOR}`
                : ` ${prefix}${cmdName}`;

            // Add usage suffix if present
            if (usageSuffix) {
                line += ` ${DIM}${usageSuffix}${RESET}`;
            }

            // Add description
            if (cmd.description) {
                const descStart = visibleWidth(line) + 2;
                const descWidth = Math.max(1, contentWidth - descStart);
                if (descWidth > 0) {
                    line += `  ${DIM}${cmd.description.slice(0, descWidth)}${RESET}`;
                }
            }

            lines.push(line);
        }

        this.cachedLines = lines;
        return lines;
    }
}
