// ── Session browser overlay ──────────────────────────────────────────────────
// List, search, rename, and switch sessions. Powered by SessionStore.
// Not to be confused with agent-core's SessionManager, which manages an
// internal JSONL crash-recovery journal, not this UI.

import type { InkListOverlayModel } from "./ink-overlay-model.ts";
import type { SessionStore } from "@logician/coding-agent/sessions";

export interface SessionInfo {
	id: string;
	title: string;
	name: string | null;
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

export class SessionBrowserOverlay {
	private store: SessionStore | null = null;
	private sessions: SessionInfo[] = [];
	private selectedIndex = 0;
	private visible = false;
	private mode: InputMode = "list";
	private filter = "";
	private renameSessionId: string | null = null;
	private renameInput = "";
	private actionCallback: ((action: SessionManagerAction) => void) | null =
		null;

	setStore(store: SessionStore): void {
		this.store = store;
		this.refresh();
	}

	setActionCallback(cb: (action: SessionManagerAction) => void): void {
		this.actionCallback = cb;
	}

	refresh(): void {
		if (this.store) {
			const summaries = this.store.listSessions();
			this.sessions = summaries.map((s) => ({
				id: s.id,
				title: s.title,
				name: s.name,
				preview: s.preview,
				lastUpdated: s.lastUpdated,
				messageCount: s.messageCount,
			}));
		}
		this.selectedIndex = 0;
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
		this.selectedIndex =
			(this.selectedIndex + delta + this.sessions.length) %
			this.sessions.length;
		this.invalidate();
	}

	private _select(): void {
		if (this.sessions.length > 0 && this.selectedIndex < this.sessions.length) {
			this.actionCallback?.({
				type: "select",
				sessionId: this.sessions[this.selectedIndex].id,
			});
		}
		this.hide();
	}

	private _startRename(): void {
		if (this.sessions.length === 0) return;
		const session = this.sessions[this.selectedIndex];
		this.mode = "rename";
		this.renameSessionId = session.id;
		this.renameInput = session.title;
		this.invalidate();
	}

	private _startDelete(): void {
		if (this.sessions.length === 0) return;
		this.mode = "delete-confirm";
		this.invalidate();
	}

	private _applyFilter(): void {
		const query = this.renameInput.toLowerCase();
		if (!query) {
			this.sessions = this.store
				? this.store.listSessions().map((s) => ({
						id: s.id,
						title: s.title,
						name: s.name,
						preview: s.preview,
						lastUpdated: s.lastUpdated,
						messageCount: s.messageCount,
					}))
				: [];
		} else {
			const all = this.store ? this.store.listSessions() : [];
			this.sessions = all.filter(
				(s) =>
					s.title.toLowerCase().includes(query) ||
					(s.name ?? "").toLowerCase().includes(query) ||
					s.preview.toLowerCase().includes(query),
			);
		}
		this.selectedIndex = 0;
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
			if (this.selectedIndex < this.sessions.length) {
				this.actionCallback?.({
					type: "delete",
					sessionId: this.sessions[this.selectedIndex].id,
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
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		if (this.mode === "rename") {
			const filtering = this.renameSessionId === null;
			return {
				kind: "list",
				title: filtering ? "Filter sessions" : "Rename session",
				headerLines: [this.renameInput],
				items: [],
				emptyText: filtering ? `${this.sessions.length} matches` : "Enter a new title.",
				footer: "Enter confirm · Esc cancel",
				selectedIndex: 0,
			};
		}
		if (this.mode === "delete-confirm") {
			const session = this.sessions[this.selectedIndex];
			return {
				kind: "list",
				title: "Delete session?",
				subtitle: session ? ` · ${session.title}` : undefined,
				items: [],
				emptyText: "This cannot be undone.",
				footer: "Y confirm · Esc cancel",
				selectedIndex: 0,
			};
		}
		return {
			kind: "list",
			title: "Sessions",
			subtitle: ` (${this.sessions.length} total)`,
			hints: "enter switch · ' filter · ^R rename · ^D delete · ^N new",
			headerLines: this.filter ? [`Filter: ${this.filter}`] : undefined,
			items: this.sessions.map((session, index) => ({
				label: session.name ? `${session.name}  (${session.title})` : session.title,
				metadata: `${session.messageCount}msg`,
				selected: index === this.selectedIndex,
			})),
			emptyText: "No sessions found.",
			footer: "Enter switch · Ctrl+R rename · Ctrl+D delete · Ctrl+N new",
			selectedIndex: this.selectedIndex,
			maxRows: 12,
		};
	}

}
