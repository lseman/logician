// ── Session browser overlay ──────────────────────────────────────────────────
// List, search, rename, and switch sessions. Powered by TuiSessionService,
// backed by agent-core's Session/SessionManager JSONL tree.

import type { TuiSessionService } from "@logician/agent-runtime/sessions";
import { BOLD, type Component, RESET, visibleWidth } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import {
	clampPopupLines,
	type ListItem,
	POPUP_FRAME_OVERHEAD,
	renderListItem,
	renderListPopupBody,
	renderSeparator,
	renderStatusLine,
} from "./popup-utils.ts";
import { SelectorController } from "./selector-controller.ts";

const getHeaderColor = (): string => theme.fg("header", "");
const getYellow = (): string => theme.fg("levelHigh", "");
const getRed = (): string => theme.fg("error", "");

export interface SessionInfo {
	id: string;
	name: string;
	preview: string;
	lastUpdated: string;
	messageCount: number;
}

// ── Actions emitted by the overlay ──────────────────────────────────────────

export type SessionManagerAction =
	| { type: "close" }
	| { type: "select"; sessionId: string }
	| { type: "rename"; sessionId: string; title: string }
	| { type: "delete"; sessionId: string }
	| { type: "new" };

// ── Mode ──────────────────────────────────────────────────────────────────────

type InputMode = "list" | "rename" | "delete-confirm" | "new";

export class SessionBrowserOverlay implements Component {
	private store: TuiSessionService | null = null;
	private sessions: SessionInfo[] = [];
	private _selection = new SelectorController();
	private visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private mode: InputMode = "list";
	private filter = "";
	private renameSessionId: string | null = null;
	private renameInput = "";
	private actionCallback: ((action: SessionManagerAction) => void) | null =
		null;

	setStore(store: TuiSessionService): void {
		this.store = store;
		this.refresh();
	}

	setActionCallback(cb: (action: SessionManagerAction) => void): void {
		this.actionCallback = cb;
	}

	refresh(): void {
		if (this.store) {
			this.sessions = this.store.listSessions();
		}
		this._selection.set(0, this.sessions.length);
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	show(): void {
		this.visible = true;
		this.mode = "list";
		this.filter = "";
		this.renameSessionId = null;
		this.renameInput = "";
		if (this.store) this.refresh();
		else this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.mode = "list";
		this.invalidate();
	}

	handleInput(data: string): void {
		switch (this.mode) {
			case "list":
				this.handleListInput(data);
				break;
			case "rename":
				this.handleRenameInput(data);
				break;
			case "delete-confirm":
				this.handleDeleteConfirm(data);
				break;
		}
	}

	// ── List mode ──────────────────────────────────────────────────────────

	private handleListInput(data: string): void {
		// Escape — close
		if (data === "\x1b" || data === "\x03") {
			this.actionCallback?.({ type: "close" });
			return;
		}

		// Enter — select
		if (data === "\r" || data === "\n") {
			this._select();
			return;
		}

		// Tab — select
		if (data === "\t") {
			this._select();
			return;
		}

		// Up / Down
		if (data === "\x1b[A" || data === "\x1bOA") {
			this._moveSelection(-1);
			return;
		}
		if (data === "\x1b[B" || data === "\x1bOB") {
			this._moveSelection(1);
			return;
		}

		// Ctrl+F — enter filter mode
		if (data === "\x06") {
			this.mode = "rename"; // reuse rename input for filter
			this.renameSessionId = null;
			this.renameInput = this.filter;
			this.invalidate();
			return;
		}

		// Ctrl+R — rename
		if (data === "\x12") {
			this._startRename();
			return;
		}

		// Ctrl+D — delete
		if (data === "\x04") {
			this._startDelete();
			return;
		}

		// Ctrl+N — new session
		if (data === "\x0e") {
			this.actionCallback?.({ type: "new" });
			return;
		}

		// Single-quote — enter filter
		if (data === "'") {
			this.mode = "rename";
			this.renameSessionId = null;
			this.renameInput = this.filter;
			this.invalidate();
			return;
		}

		// Backspace in filter
		if (data === "\x7f" || data === "\b") {
			if (this.renameInput.length > 0) {
				this.renameInput = this.renameInput.slice(0, -1);
				this._applyFilter();
				this.invalidate();
			}
			return;
		}

		// Regular characters — filter
		if (data.length === 1) {
			this.renameInput += data;
			this._applyFilter();
			this.invalidate();
			return;
		}
	}

	private _moveSelection(delta: number): void {
		if (this.sessions.length === 0) return;
		this._selection.move(delta, this.sessions.length);
		this.invalidate();
	}

	private _select(): void {
		if (
			this.sessions.length > 0 &&
			this._selection.index < this.sessions.length
		) {
			this.actionCallback?.({
				type: "select",
				sessionId: this.sessions[this._selection.index].id,
			});
		}
		this.hide();
	}

	private _startRename(): void {
		if (this.sessions.length === 0) return;
		const session = this.sessions[this._selection.index];
		this.mode = "rename";
		this.renameSessionId = session.id;
		this.renameInput = session.name;
		this.invalidate();
	}

	private _startDelete(): void {
		if (this.sessions.length === 0) return;
		this.mode = "delete-confirm";
		this.invalidate();
	}

	private _applyFilter(): void {
		this.filter = this.renameInput;
		const query = this.renameInput.toLowerCase();
		const all = this.store ? this.store.listSessions() : [];
		this.sessions = query
			? all.filter(
					s =>
						s.name.toLowerCase().includes(query) ||
						s.preview.toLowerCase().includes(query),
				)
			: all;
		this._selection.set(0, this.sessions.length);
	}

	// ── Rename mode ────────────────────────────────────────────────────────

	private handleRenameInput(data: string): void {
		// Escape — cancel
		if (data === "\x1b" || data === "\x03") {
			this.mode = "list";
			this.renameInput = "";
			this.renameSessionId = null;
			this.invalidate();
			return;
		}

		// Enter — confirm rename
		if (data === "\r" || data === "\n") {
			const newTitle = this.renameInput.trim();
			if (this.renameSessionId && newTitle.length > 0) {
				this.store?.renameSession(this.renameSessionId, newTitle);
				this.refresh();
			}
			this.mode = "list";
			this.renameInput = "";
			this.renameSessionId = null;
			this.invalidate();
			return;
		}

		// Backspace
		if (data === "\x7f" || data === "\b") {
			if (this.renameInput.length > 0) {
				this.renameInput = this.renameInput.slice(0, -1);
				this.invalidate();
			}
			return;
		}

		// Regular character
		if (data.length === 1) {
			this.renameInput += data;
			this.invalidate();
			return;
		}
	}

	// ── Delete confirm mode ────────────────────────────────────────────────

	private handleDeleteConfirm(data: string): void {
		if (data === "\x1b" || data === "\x03") {
			this.mode = "list";
			this.invalidate();
			return;
		}

		if (data === "y" || data === "Y") {
			if (this._selection.index < this.sessions.length) {
				this.actionCallback?.({
					type: "delete",
					sessionId: this.sessions[this._selection.index].id,
				});
				this.refresh();
			}
			this.mode = "list";
			this.invalidate();
			return;
		}

		this.mode = "list";
		this.invalidate();
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

		const popupWidth = Math.max(1, width);
		const innerWidth = Math.max(1, popupWidth - POPUP_FRAME_OVERHEAD);
		const lines: string[] = [];

		// ── List mode ────────────────────────────────────────────────────────
		if (this.mode === "list") {
			const headerFg = getHeaderColor();
			lines.push(
				`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`,
			);

			const titleText = "Sessions";
			const subtitleText = ` (${this.sessions.length} total)`;
			const hintsText =
				" enter switch · ' filter · ^R rename · ^D delete · ^N new";
			const titleLine = `${titleText}${theme.fg("muted", "")}${subtitleText}${hintsText}`;
			const titleVisible = visibleWidth(titleLine);
			const titlePad = Math.max(0, innerWidth - titleVisible);
			lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

			if (this.filter) {
				lines.push(renderStatusLine(`/filter: ${this.filter}`, innerWidth));
			}
			lines.push(renderSeparator(popupWidth));

			// Session list
			lines.push(
				...renderListPopupBody(
					this.sessions,
					this._selection,
					innerWidth,
					12,
					(s, i) => {
						const isSelected = i === this._selection.index;
						const item: ListItem = {
							label: s.name,
							metadata: `${s.messageCount}msg`,
							selected: isSelected,
						};
						return renderListItem(item, innerWidth);
					},
					"No sessions found.",
				),
			);

			lines.push(renderSeparator(popupWidth));
			lines.push(
				renderStatusLine(
					"Enter switch · ' filter · Ctrl+R rename · Ctrl+D delete · Ctrl+N new",
					innerWidth,
				),
			);
			lines.push(
				`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`,
			);
		}

		// ── Rename mode ──────────────────────────────────────────────────────
		if (this.mode === "rename") {
			const headerFg = getHeaderColor();
			lines.push(
				`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`,
			);

			if (this.renameSessionId !== null) {
				const title =
					this.sessions.find(s => s.id === this.renameSessionId)?.name ||
					"Untitled";
				const titleLine = `Rename: ${title}`;
				const titlePad = Math.max(0, innerWidth - visibleWidth(titleLine));
				lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
				lines.push(renderSeparator(popupWidth));
				const display = `${getYellow()}${this.renameInput}${RESET}_`;
				lines.push(renderStatusLine(display, innerWidth));
				lines.push(
					renderStatusLine("Enter to confirm, Esc to cancel", innerWidth),
				);
			} else {
				const titleLine = `Filter: ${this.sessions.length} matches`;
				const titlePad = Math.max(0, innerWidth - visibleWidth(titleLine));
				lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
				lines.push(renderSeparator(popupWidth));
				const display = `${getYellow()}${this.renameInput}${RESET}_`;
				lines.push(renderStatusLine(display, innerWidth));
				lines.push(
					renderStatusLine("Enter to apply, Esc to cancel", innerWidth),
				);
			}
			lines.push(renderSeparator(popupWidth));
			lines.push(
				`${headerFg}${"─".repeat(popupWidth)}${theme.fg("muted", "")}`,
			);
		}

		// ── Delete confirm mode ──────────────────────────────────────────────
		if (this.mode === "delete-confirm") {
			const session =
				this._selection.index < this.sessions.length
					? this.sessions[this._selection.index]
					: null;
			const redFg = getRed();
			lines.push(`${redFg}${"─".repeat(popupWidth)}${RESET}`);
			const titleLine = `${BOLD}${redFg}Delete session?${RESET}${session ? `: ${session.name}` : ""}`;
			const titlePad = Math.max(0, innerWidth - visibleWidth(titleLine));
			lines.push(`${redFg} ${titleLine}${" ".repeat(titlePad + 1)}`);
			lines.push(renderSeparator(popupWidth));
			lines.push(renderStatusLine("Y to confirm, Esc to cancel", innerWidth));
			lines.push(`${redFg}${"─".repeat(popupWidth)}${RESET}`);
		}

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}
