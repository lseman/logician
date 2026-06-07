// ── Session manager overlay ──────────────────────────────────────────────────
// List, search, and switch sessions.

import { type Component, visibleWidth } from "../tui-core.ts";

const BORDERS = {
	top: "┌",
	topRight: "┐",
	bottom: "└",
	bottomRight: "┘",
	h: "─",
	v: "│",
};

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const HEADER_COLOR = "\x1b[38;5;159m";
const SELECTED_COLOR = "\x1b[38;5;111m";

export interface SessionInfo {
	id: string;
	title: string;
	preview: string;
	lastUpdated: string;
	messageCount: number;
}

export class SessionManager implements Component {
	private sessions: SessionInfo[] = [];
	private selectedIndex = 0;
	private visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

	setSessions(sessions: SessionInfo[]): void {
		this.sessions = sessions;
		this.selectedIndex = 0;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
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

		// Up arrow
		if (data === "\x1b[A" || data === "\x1bOA") {
			if (this.sessions.length > 0) {
				this.selectedIndex =
					(this.selectedIndex - 1 + this.sessions.length) %
					this.sessions.length;
				this.invalidate();
			}
			return;
		}

		// Down arrow
		if (data === "\x1b[B" || data === "\x1bOB") {
			if (this.sessions.length > 0) {
				this.selectedIndex = (this.selectedIndex + 1) % this.sessions.length;
				this.invalidate();
			}
			return;
		}

		// Tab — select current
		if (data === "\t") {
			this._submit();
			return;
		}
	}

	private _submit(): void {
		if (this.sessions.length > 0 && this.selectedIndex < this.sessions.length) {
			this._selectedSession = this.sessions[this.selectedIndex];
		}
		this.hide();
	}

	_selectedSession: SessionInfo | null = null;

	invalidate(): void {
		this.cachedLines = null;
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.cachedWidth = width;

		if (!this.visible) return [];

		const contentWidth = Math.max(40, width - 4);
		const lines: string[] = [];

		// Header
		lines.push(
			`${HEADER_COLOR}${BORDERS.top}${BORDERS.h.repeat(contentWidth)}${BORDERS.topRight}${RESET}`,
		);
		lines.push(
			`${BORDERS.v} ${BOLD}Sessions${DIM} (${this.sessions.length} total)${RESET}`,
		);
		lines.push(`${BORDERS.v}${BORDERS.h.repeat(contentWidth)}${BORDERS.v}`);

		// Session list
		for (let i = 0; i < this.sessions.length; i++) {
			const s = this.sessions[i];
			const isSelected = i === this.selectedIndex;
			const prefix = isSelected ? "▸ " : "  ";

			let line = "";

			if (isSelected) {
				line = `${BORDERS.v} ${SELECTED_COLOR}${prefix}${BOLD}${s.title}${RESET}${SELECTED_COLOR}`;
			} else {
				line = `${BORDERS.v}  ${s.title}`;
			}

			// Add metadata
			const meta = `${DIM}${s.messageCount}msg${RESET}`;
			const metaStart = visibleWidth(line) + 2;
			if (metaStart < contentWidth) {
				line += meta;
			}

			lines.push(line);
		}

		// Footer
		const footer = `${BORDERS.bottom}${BORDERS.h.repeat(contentWidth)}${BORDERS.bottomRight}`;
		lines.push(footer);

		this.cachedLines = lines;
		return lines;
	}
}
