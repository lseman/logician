// ── TrustPromptOverlay — project-directory trust prompt ──────────────────────
// Shown at TUI startup when the current directory (or an ancestor) contains
// trust-requiring resources (.logician/, extensions/, skills/, etc.).

import { visibleWidth, BOLD, DIM, RESET } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";
import {
	BOX,
	renderSeparator,
	renderStatusLine,
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
} from "./popup-utils.ts";

export type TrustChoice =
	| "trust"
	| "trust-parent"
	| "session-only"
	| "deny"
	| "deny-session";

export interface TrustPromptOverlayOptions {
	/** Working directory being asked about. */
	cwd: string;
	/** Trust-requiring resource paths found under cwd. */
	paths?: string[];
}

export interface TrustPromptAction {
	type: "trust-choice";
	choice: TrustChoice;
}

const OPTIONS: Array<{
	value: TrustChoice;
	label: string;
	hint?: string;
}> = [
	{ value: "trust", label: "Trust", hint: "persist" },
	{ value: "trust-parent", label: "Trust parent folder", hint: "persist" },
	{ value: "session-only", label: "Trust (this session only)" },
	{ value: "deny", label: "Do not trust", hint: "persist" },
	{ value: "deny-session", label: "Do not trust (this session only)" },
];

export class TrustPromptOverlay {
	private cwd = "";
	private paths: string[] = [];
	private selectedIndex = 0;
	private visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private onClose?: () => void;

	setOptions(opts: TrustPromptOverlayOptions): void {
		this.cwd = opts.cwd;
		this.paths = opts.paths ?? [];
		this.selectedIndex = 0;
		this.invalidate();
	}

	setOnClose(cb: () => void): void {
		this.onClose = cb;
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	isVisible(): boolean {
		return this.visible;
	}

	moveSelection(delta: number): void {
		const n = OPTIONS.length;
		this.selectedIndex = ((this.selectedIndex + delta) % n + n) % n;
		this.invalidate();
	}

	handleInput(data: string): TrustPromptAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03") {
			// Esc → deny for this session (safe default)
			this.hide();
			return { type: "trust-choice", choice: "deny-session" };
		}

		if (data === "\r" || data === "\n" || data === "\t") {
			const choice = OPTIONS[this.selectedIndex].value;
			this.hide();
			return { type: "trust-choice", choice };
		}

		if (data === "\x1b[A" || data === "\x1bOA" || data === "k" || data === "K") {
			this.moveSelection(-1);
			return null;
		}

		if (data === "\x1b[B" || data === "\x1bOB" || data === "j" || data === "J") {
			this.moveSelection(1);
			return null;
		}

		// Number keys 1-5 select directly
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c >= 0x31 && c <= 0x35) {
				this.selectedIndex = c - 0x31;
				const choice = OPTIONS[this.selectedIndex].value;
				this.hide();
				return { type: "trust-choice", choice };
			}
		}

		return null;
	}

	invalidate(): void {
		this.cachedLines = null;
	}

	// ── Rendering ─────────────────────────────────────────────────────────────

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.cachedWidth = width;
		if (!this.visible) return [];

		const popupWidth = Math.max(48, Math.min(width, 90));
		const innerWidth = Math.max(1, popupWidth - POPUP_FRAME_OVERHEAD);
		const headerFg = theme.fg("header", "");
		const lines: string[] = [];

		// ── Top border ──
		lines.push(`${headerFg}${BOX.horiz.repeat(popupWidth)}${RESET}`);

		// ── Title row ──
		const titleText = "Trust project folder";
		const hintsText = " ↑↓ navigate · enter confirm · esc deny";
		const titleLine = `${titleText}${theme.fg("muted", "")}${hintsText}`;
		const titleVisible = visibleWidth(titleLine);
		const titlePad = Math.max(0, innerWidth - titleVisible);
		lines.push(`${headerFg} ${titleLine}${" ".repeat(titlePad + 1)}`);

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Question line ──
		const icon = `${theme.fg("header", "")}❯${RESET}`;
		const cwdLine = `${icon} ${BOLD}${this.cwd}${RESET}`;
		const cwdVisible = visibleWidth(cwdLine);
		const cwdPad = Math.max(0, innerWidth - cwdVisible);
		lines.push(` ${cwdLine}${" ".repeat(cwdPad + 1)}`);

		// ── Description ──
		const desc =
			"This allows Logician to load local settings, extensions, skills, and execute project resources.";
		lines.push(renderStatusLine(desc, innerWidth, theme.fg("muted", "")));

		// ── Resource paths (if any) ──
		if (this.paths.length > 0) {
			const maxPaths = 5;
			const shown = this.paths.slice(0, maxPaths);
			lines.push(renderSeparator(popupWidth));
			for (const p of shown) {
				const pathText = `  ${DIM}•${RESET} ${p}`;
				const pathVisible = visibleWidth(pathText);
				const pathPad = Math.max(0, innerWidth - pathVisible);
				lines.push(` ${pathText}${" ".repeat(pathPad + 1)}`);
			}
			if (this.paths.length > maxPaths) {
				const more = `  ${DIM}… ${this.paths.length - maxPaths} more${RESET}`;
				lines.push(` ${more}${" ".repeat(innerWidth - visibleWidth(more) + 1)}`);
			}
		}

		// ── Separator ──
		lines.push(renderSeparator(popupWidth));

		// ── Options ──
		const maxRows = Math.min(5, popupWidth - 12);
		const start = 0;
		const end = Math.min(OPTIONS.length, start + maxRows);

		for (let i = start; i < end; i++) {
			const opt = OPTIONS[i];
			const isSelected = i === this.selectedIndex;
			const bg = isSelected ? theme.fgAsBg("selected") : "";
			const segReset = isSelected ? `${RESET}${bg}` : RESET;
			const blackBg = "\x1b[38;5;16m";

			let left = "";
			if (isSelected) {
				left += `${bg}${blackBg}${BOLD}▸${segReset} `;
			} else {
				left += `${DIM}${i + 1}${RESET}  `;
			}

			const label = isSelected ? `${bg}${blackBg}${BOLD}${opt.label}${segReset}` : opt.label;
			left += label;

			if (opt.hint && isSelected) {
				const hint = `${bg}${blackBg} [${opt.hint}]${segReset}`;
				const leftVisible = visibleWidth(left);
				const hintVisible = visibleWidth(hint);
				const gap = Math.max(1, innerWidth - leftVisible - hintVisible);
				const content = `${left}${bg}${" ".repeat(gap)}${hint}${bg}${" ".repeat(2)}`;
				lines.push(`${bg}${" ".repeat(1)}${content}${bg}${" ".repeat(1)}${RESET}`);
			} else if (opt.hint && !isSelected) {
				const leftVisible = visibleWidth(left);
				const hint = `${DIM}[${opt.hint}]${RESET}`;
				const hintVisible = visibleWidth(hint);
				const gap = Math.max(1, innerWidth - leftVisible - hintVisible);
				const content = `${left}${" ".repeat(gap)}${hint}`;
				const padRight = Math.max(0, innerWidth - visibleWidth(content));
				lines.push(` ${content}${" ".repeat(padRight + 1)}`);
			} else {
				const leftVisible = visibleWidth(left);
				const padRight = Math.max(0, innerWidth - leftVisible);
				lines.push(` ${left}${" ".repeat(padRight + 1)}`);
			}
		}

		// ── Bottom hint ──
		lines.push(renderSeparator(popupWidth));
		const defaultHint = "Default: Esc → deny for this session";
		lines.push(renderStatusLine(defaultHint, innerWidth));

		// ── Bottom border ──
		lines.push(`${headerFg}${BOX.horiz.repeat(popupWidth)}${RESET}`);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}
