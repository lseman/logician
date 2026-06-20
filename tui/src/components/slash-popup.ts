// ── Slash command popup ───────────────────────────────────────────────────────
// Overlay popup with fuzzy matching, category grouping, arg hints, and examples.

import { theme } from "../theme.ts";
import {
	filterSlashCommands,
	groupByCategory,
	CATEGORY_ORDER,
	type SlashCommandDef,
	type SlashCommandCategory,
} from "../slash-commands.ts";
import { type Component, visibleWidth } from "../tui-core.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const getHeaderColor = (): string => theme.fg("header", "");
const getSelectedColor = (): string => theme.fg("selected", "");
const getCategoryColor = (cat: SlashCommandCategory): string => {
	const colors: Record<SlashCommandCategory, string> = {
		help: "\x1b[36m", session: "\x1b[33m", agent: "\x1b[35m",
		context: "\x1b[34m", rag: "\x1b[32m", skills: "\x1b[95m",
		reasoning: "\x1b[37m", display: "\x1b[93m", permissions: "\x1b[31m",
		shortcuts: "\x1b[36m", loop: "\x1b[94m", misc: "\x1b[90m",
	};
	return colors[cat] ?? "\x1b[90m";
};

interface RenderState {
	filtered: SlashCommandDef[];
	isFiltered: boolean;
	selectedCmd: SlashCommandDef | null;
	// For grouped display (when not filtered): ordered category headers with command indices
	groups: Array<{ category: SlashCommandCategory; start: number; count: number }>;
	// Map from flat index to command (with group headers in between)
	flatEntries: Array<{ cmd: SlashCommandDef; isHeader: boolean; category?: SlashCommandCategory }>;
	// Selection translated to flat index
	flatSelection: number;
}

export class SlashPopup implements Component {
	private commands: SlashCommandDef[] = [];
	private query = "";
	private selectedIndex = 0;
	public visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private onSubmit?: (
		result: string | null,
		dispatch?: "quit",
		command?: string,
	) => void;

	/** Prepare rendering state: groups commands by category or filters by query. */
	private _prepareRenderState(): RenderState {
		const filtered = filterSlashCommands(this.commands, this.query);
		const isFiltered = this.query.length > 1;

		if (isFiltered) {
			return {
				filtered,
				isFiltered: true,
				selectedCmd: filtered.length > 0 ? filtered[this.selectedIndex] : null,
				groups: [],
				flatEntries: filtered.map((cmd) => ({ cmd, isHeader: false })),
				flatSelection: Math.min(this.selectedIndex, filtered.length - 1),
			};
		}

		const groupsMap = groupByCategory(filtered);
		const groups: RenderState["groups"] = [];
		const flatEntries: RenderState["flatEntries"] = [];
		let idx = 0;
		for (const cat of CATEGORY_ORDER) {
			const cmds = groupsMap.get(cat);
			if (!cmds || cmds.length === 0) continue;
			const start = idx;
			flatEntries.push({ cmd: {} as SlashCommandDef, isHeader: true, category: cat });
			idx++;
			for (const cmd of cmds) {
				flatEntries.push({ cmd, isHeader: false });
				idx++;
			}
			groups.push({ category: cat, start, count: cmds.length });
		}

		return {
			filtered,
			isFiltered: false,
			selectedCmd: filtered.length > 0 ? filtered[this.selectedIndex] : null,
			groups,
			flatEntries,
			flatSelection: Math.min(this.selectedIndex, flatEntries.length - 1),
		};
	}

	setCommands(commands: SlashCommandDef[]): void {
		this.commands = commands;
		this.selectedIndex = 0;
		this.invalidate();
	}

	getCommands(): SlashCommandDef[] {
		return this.commands;
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
		return filtered.length > 0 ? filtered[this.selectedIndex].command : null;
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
				this.query = `${filtered[this.selectedIndex].command} `;
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
					(this.selectedIndex - 1 + filtered.length) % filtered.length;
				this.invalidate();
			}
			return;
		}

		// Up arrow
		if (data === "\x1b[A" || data === "\x1bOA") {
			const filtered = this._getFiltered();
			if (filtered.length > 0) {
				this.selectedIndex =
					(this.selectedIndex - 1 + filtered.length) % filtered.length;
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
		cb: (result: string | null, dispatch?: "quit", command?: string) => void,
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

		if (!this.visible) return [];

		const state = this._prepareRenderState();
		const contentWidth = Math.min(80, Math.max(40, width - 4));
		const lines: string[] = [];

		// Title row
		const count = state.filtered.length;
		const hint = `${DIM}↑↓ select · Tab complete · ⏎ run · Esc close${RESET}`;
		lines.push(
			` ${getHeaderColor()}commands${RESET}${DIM} (${count})${RESET}  ${hint}`,
		);

		if (state.groups.length > 0) {
			// Grouped display: category headers + commands
			for (const entry of state.flatEntries) {
				if (entry.isHeader) {
					const catColor = getCategoryColor(entry.category!);
					const catLabel = entry.category!.charAt(0).toUpperCase() + entry.category!.slice(1);
					lines.push(`${DIM}${catColor}── ${catLabel} ──${RESET}`);
					continue;
				}

				const cmd = entry.cmd;
				const idx = state.flatEntries.indexOf(entry);
				const isSelected = idx === state.flatSelection;
				const prefix = isSelected ? "▸ " : "  ";

				const cmdName = cmd.command;
				let line = isSelected
					? ` ${getSelectedColor()}${prefix}${BOLD}${cmdName}${RESET}${getSelectedColor()}`
					: ` ${prefix}${cmdName}`;

				// Arg hint in brackets
				if (cmd.argHint) {
					line += ` ${DIM}[${cmd.argHint}]${RESET}`;
				}

				// Description
				if (cmd.description) {
					const descStart = visibleWidth(line) + 2;
					const descWidth = Math.max(1, contentWidth - descStart);
					if (descWidth > 0) {
						line += `  ${DIM}${cmd.description.slice(0, descWidth)}${RESET}`;
					}
				}

				lines.push(line);
			}
		} else {
			// Flat filtered list
			for (let i = 0; i < state.filtered.length; i++) {
				const cmd = state.filtered[i];
				const isSelected = i === state.flatSelection;
				const prefix = isSelected ? "▸ " : "  ";
				const cmdName = cmd.command;

				let line = isSelected
					? ` ${getSelectedColor()}${prefix}${BOLD}${cmdName}${RESET}${getSelectedColor()}`
					: ` ${prefix}${cmdName}`;

				// Arg hint
				if (cmd.argHint) {
					line += ` ${DIM}[${cmd.argHint}]${RESET}`;
				}

				// Description
				if (cmd.description) {
					const descStart = visibleWidth(line) + 2;
					const descWidth = Math.max(1, contentWidth - descStart);
					if (descWidth > 0) {
						line += `  ${DIM}${cmd.description.slice(0, descWidth)}${RESET}`;
					}
				}

				lines.push(line);
			}
		}

		// Details panel for selected command (examples, arg info)
		if (state.selectedCmd && (state.selectedCmd.examples || state.selectedCmd.argHint)) {
			const sel = state.selectedCmd;
			lines.push(``);
			const detailsColor = getCategoryColor(sel.category ?? "misc");
			if (sel.argHint) {
				lines.push(`${DIM}Usage:${RESET} ${detailsColor}${sel.command}${RESET} ${BOLD}${sel.argHint}${RESET}`);
			}
			if (sel.examples && sel.examples.length > 0) {
				for (const ex of sel.examples) {
					lines.push(`${DIM}  Example:${RESET} ${DIM}${ex}${RESET}`);
				}
			}
		}

		this.cachedLines = lines;
		return lines;
	}
}
