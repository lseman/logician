// ── TrustPromptOverlay — project-directory trust prompt ──────────────────────
// Shown at TUI startup when the current directory (or an ancestor) contains
// trust-requiring resources (.logician/, extensions/, skills/, etc.).

import {
	visibleWidth,
	clampLineToWidth,
	BOLD,
	DIM,
	RESET,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import { BOX, clampPopupLines } from "./popup-utils.ts";

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
	description: string;
	key: string;
}> = [
	{
		value: "trust",
		label: "Trust this folder",
		description: "Remember this exact workspace",
		key: "y",
	},
	{
		value: "trust-parent",
		label: "Trust parent folder",
		description: "Remember the parent and its workspaces",
		key: "p",
	},
	{
		value: "session-only",
		label: "Trust for this session",
		description: "Allow now without saving",
		key: "s",
	},
	{
		value: "deny",
		label: "Do not trust",
		description: "Remember this folder as blocked",
		key: "n",
	},
	{
		value: "deny-session",
		label: "Exit without saving",
		description: "Keep the folder untrusted",
		key: "esc",
	},
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
			const shortcut = data === "N"
				? "deny-session"
				: ({
						y: "trust",
						Y: "trust",
						p: "trust-parent",
						P: "trust-parent",
						s: "session-only",
						S: "session-only",
						n: "deny",
					} as Record<string, TrustChoice | undefined>)[data];
			if (shortcut) {
				this.hide();
				return { type: "trust-choice", choice: shortcut };
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

		const popupWidth = Math.max(24, Math.min(width, 78));
		const innerWidth = Math.max(1, popupWidth - 4);
		const border = theme.fg("border", "");
		const lines: string[] = [];
		const row = (content = ""): string => {
			const clipped = clampLineToWidth(content, innerWidth);
			const padding = " ".repeat(
				Math.max(0, innerWidth - visibleWidth(clipped)),
			);
			return `${border}${BOX.vert}${RESET} ${clipped}${padding} ${border}${BOX.vert}${RESET}`;
		};
		const separator = (): string =>
			`${border}${BOX.separator}${BOX.horiz.repeat(popupWidth - 2)}${BOX.sepRight}${RESET}`;

		lines.push(
			`${border}${BOX.topLeft}${BOX.horiz.repeat(popupWidth - 2)}${BOX.topRight}${RESET}`,
		);
		lines.push(
			row(
				`${theme.fg("warning", "◆")} ${BOLD}${theme.fg("header", "TRUST THIS WORKSPACE?")}${RESET}`,
			),
		);
		lines.push(row(`${theme.fg("muted", "Folder")}  ${theme.fg("text", this.cwd)}`));
		lines.push(separator());
		lines.push(
			row(
				"Local configuration can change agent instructions and permit project tools.",
			),
		);
		lines.push(
			row(
				`${DIM}Only continue if you recognize and trust this folder.${RESET}`,
			),
		);

		if (this.paths.length > 0) {
			const maxPaths = 3;
			const shown = this.paths.slice(0, maxPaths);
			lines.push(separator());
			lines.push(row(theme.fg("muted", "LOCAL RESOURCES")));
			for (const p of shown) {
				lines.push(row(`${theme.fg("separator", "│")} ${p}`));
			}
			if (this.paths.length > maxPaths) {
				lines.push(
					row(`${DIM}… and ${this.paths.length - maxPaths} more${RESET}`),
				);
			}
		}

		lines.push(separator());
		for (let i = 0; i < OPTIONS.length; i++) {
			const opt = OPTIONS[i];
			const isSelected = i === this.selectedIndex;
			const marker = isSelected
				? theme.fg("active", "›")
				: theme.fg("dim", `${i + 1}`);
			const label = isSelected
				? `${BOLD}${theme.fg("text", opt.label)}${RESET}`
				: theme.fg("muted", opt.label);
			const shortcut = theme.fg("dim", opt.key);
			const left = `${marker} ${label}`;
			const right = `${opt.description}  ${shortcut}`;
			const gap = Math.max(
				1,
				innerWidth - visibleWidth(left) - visibleWidth(right),
			);
			lines.push(row(`${left}${" ".repeat(gap)}${theme.fg("dim", right)}`));
		}

		lines.push(separator());
		lines.push(
			row(
				`${theme.fg("muted", "↑↓ navigate")}  ·  ${theme.fg("muted", "Enter confirm")}  ·  ${theme.fg("muted", "Esc exit safely")}`,
			),
		);
		lines.push(
			`${border}${BOX.bottomLeft}${BOX.horiz.repeat(popupWidth - 2)}${BOX.bottomRight}${RESET}`,
		);

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}
