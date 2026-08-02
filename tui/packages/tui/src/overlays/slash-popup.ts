// ── Slash command popup ───────────────────────────────────────────────────────
// Overlay popup with fuzzy matching, category grouping, arg hints, and examples.

import {
	CATEGORY_ORDER,
	filterSlashCommands,
	groupByCategory,
	type SlashCommandCategory,
	type SlashCommandDef,
} from "@logician/coding-agent/commands";
import type { InkListOverlayModel } from "./ink-overlay-model.ts";

const MAX_VISIBLE_ENTRIES = 8;

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

export class SlashPopup {
	private commands: SlashCommandDef[] = [];
	private query = "";
	private selectedIndex = 0;
	public visible = false;
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
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		const state = this._prepareRenderState();
		return {
			kind: "list",
			title: "commands",
			subtitle: ` (${state.filtered.length})`,
			hints: "↑↓ select · tab complete · enter run · esc close",
			items: state.filtered.map((command, index) => ({
				label: command.command,
				metadata: [command.argHint ? `[${command.argHint}]` : "", command.description]
					.filter(Boolean)
					.join("  "),
				selected: index === this.selectedIndex,
			})),
			emptyText: "No matching commands.",
			footer: state.selectedCmd?.description ?? "Run a Logician command.",
			selectedIndex: this.selectedIndex,
			maxRows: MAX_VISIBLE_ENTRIES,
		};
	}
}
