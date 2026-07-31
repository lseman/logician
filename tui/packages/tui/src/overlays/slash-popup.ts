// ── Slash command popup ───────────────────────────────────────────────────────
// Overlay popup with fuzzy matching, category grouping, arg hints, and examples.

import {
	CATEGORY_ORDER,
	filterSlashCommands,
	groupByCategory,
	type SlashCommandCategory,
	type SlashCommandDef,
} from "@logician/coding-agent/commands";
import {
	BOLD,
	type Component,
	DIM,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import {
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
	renderListPopupFrame,
} from "./popup-utils.ts";

const MAX_VISIBLE_ENTRIES = 8;
const getSelectedColor = (): string => theme.fgRaw("selected");
const getCategoryColor = (cat: SlashCommandCategory): string => {
	const colors: Record<SlashCommandCategory, string> = {
		help: "\x1b[36m",
		session: "\x1b[33m",
		agent: "\x1b[35m",
		context: "\x1b[34m",
		skills: "\x1b[95m",
		reasoning: "\x1b[37m",
		display: "\x1b[93m",
		permissions: "\x1b[31m",
		shortcuts: "\x1b[36m",
		loop: "\x1b[94m",
		misc: "\x1b[90m",
	};
	return colors[cat] ?? "\x1b[90m";
};

interface RenderState {
	filtered: SlashCommandDef[];
	isFiltered: boolean;
	selectedCmd: SlashCommandDef | null;
	// For grouped display (when not filtered): ordered category headers with command indices
	groups: Array<{
		category: SlashCommandCategory;
		start: number;
		count: number;
	}>;
	// Map from flat index to command (with group headers in between)
	flatEntries: Array<{
		cmd: SlashCommandDef;
		isHeader: boolean;
		category?: SlashCommandCategory;
	}>;
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
		let commandIndex = 0;
		let flatSelection = 0;
		for (const cat of CATEGORY_ORDER) {
			const cmds = groupsMap.get(cat);
			if (!cmds || cmds.length === 0) continue;
			const start = idx;
			flatEntries.push({
				cmd: {} as SlashCommandDef,
				isHeader: true,
				category: cat,
			});
			idx++;
			for (const cmd of cmds) {
				if (commandIndex === this.selectedIndex) flatSelection = idx;
				flatEntries.push({ cmd, isHeader: false });
				idx++;
				commandIndex++;
			}
			groups.push({ category: cat, start, count: cmds.length });
		}

		return {
			filtered,
			isFiltered: false,
			selectedCmd: filtered.length > 0 ? filtered[this.selectedIndex] : null,
			groups,
			flatEntries,
			flatSelection,
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
		if (filtered.length > 0) {
			const cmd = filtered[this.selectedIndex];
			const args = this.query.replace(/^\/[^\s]+\s*/, "").trim();
			const raw = args ? `${cmd.command} ${args}` : cmd.command;
			this.submitRaw(raw);
		}
		this.hide();
	}

	/** Execute an exact command through the same path used by popup submission. */
	submitRaw(raw: string): boolean {
		this._lastResult = null;
		this._lastCommand = raw.trim();
		const commandName = this._lastCommand.split(/\s+/, 1)[0]?.toLowerCase();
		const cmd = this.commands.find(
			(command) => command.command.toLowerCase() === commandName,
		);
		if (!cmd) return false;
		const args = this._lastCommand.replace(/^\/[^\s]+\s*/, "").trim();

		if (cmd.dispatch === "quit") {
			if (cmd.handler) {
				const result = cmd.handler(args);
				if (result) this._lastResult = String(result);
			}
			cmd.bridgeHandler?.(args);
			this.onSubmit?.(null, "quit", this._lastCommand);
			return true;
		}

		// Establish the user-command turn BEFORE local handlers run.
		// Async handlers like /spawn emit tool/lifecycle/stream events that
		// attach to getCurrentTurn(); if the turn is created afterwards those
		// events land on a different card and the stream/final output vanish.
		this.onSubmit?.(null, undefined, this._lastCommand);

		if (cmd.handler) {
			const result = cmd.handler(args);
			if (result) this._lastResult = String(result);
		}
		cmd.bridgeHandler?.(args);

		// Deliver handler return text as a system notice without a second turn.
		if (this._lastResult) {
			this.onSubmit?.(this._lastResult, undefined, undefined);
		}
		return true;
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
		const popupWidth = Math.max(1, width);
		const contentWidth = Math.max(1, popupWidth - POPUP_FRAME_OVERHEAD);
		const lines: string[] = [];

		const count = state.filtered.length;

		if (state.groups.length > 0) {
			// Grouped display: category headers + commands
			const { items, hiddenAbove, hiddenBelow } = windowAroundSelection(
				state.flatEntries,
				state.flatSelection,
			);
			if (hiddenAbove > 0) {
				lines.push(`${DIM}  ↑ ${hiddenAbove} more above${RESET}`);
			}
			for (const entry of items) {
				if (entry.isHeader) {
					const category = entry.category ?? "misc";
					const catColor = getCategoryColor(category);
					const catLabel = category.charAt(0).toUpperCase() + category.slice(1);
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
			if (hiddenBelow > 0) {
				lines.push(`${DIM}  ↓ ${hiddenBelow} more below${RESET}`);
			}
		} else {
			// Flat filtered list
			const indexed = state.filtered.map((cmd, index) => ({ cmd, index }));
			const { items, hiddenAbove, hiddenBelow } = windowAroundSelection(
				indexed,
				state.flatSelection,
			);
			if (hiddenAbove > 0) {
				lines.push(`${DIM}  ↑ ${hiddenAbove} more above${RESET}`);
			}
			for (const item of items) {
				const { cmd, index } = item;
				const isSelected = index === state.flatSelection;
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
			if (hiddenBelow > 0) {
				lines.push(`${DIM}  ↓ ${hiddenBelow} more below${RESET}`);
			}
		}

		// Details panel for selected command (examples, arg info)
		if (
			state.selectedCmd &&
			(state.selectedCmd.examples || state.selectedCmd.argHint)
		) {
			const sel = state.selectedCmd;
			lines.push("");
			const detailsColor = getCategoryColor(sel.category ?? "misc");
			if (sel.argHint) {
				lines.push(
					`${DIM}Usage:${RESET} ${detailsColor}${sel.command}${RESET} ${BOLD}${sel.argHint}${RESET}`,
				);
			}
			if (sel.examples && sel.examples.length > 0) {
				for (const ex of sel.examples) {
					lines.push(`${DIM}  Example:${RESET} ${DIM}${ex}${RESET}`);
				}
			}
		}

		this.cachedLines = clampPopupLines(
			renderListPopupFrame({
				popupWidth,
				innerWidth: contentWidth,
				title: "commands",
				subtitle: ` (${count})`,
				hints: "↑↓ select · tab complete · enter run · esc close",
				bodyLines: lines,
				bottomText: state.selectedCmd?.description ?? "Run a Logician command.",
			}),
			width,
		);
		return this.cachedLines;
	}
}

function windowAroundSelection<T>(
	items: T[],
	selection: number,
): { items: T[]; hiddenAbove: number; hiddenBelow: number } {
	if (items.length <= MAX_VISIBLE_ENTRIES) {
		return { items, hiddenAbove: 0, hiddenBelow: 0 };
	}
	const half = Math.floor(MAX_VISIBLE_ENTRIES / 2);
	const start = Math.max(
		0,
		Math.min(selection - half, items.length - MAX_VISIBLE_ENTRIES),
	);
	const end = Math.min(items.length, start + MAX_VISIBLE_ENTRIES);
	return {
		items: items.slice(start, end),
		hiddenAbove: start,
		hiddenBelow: items.length - end,
	};
}
