// ── TrustPromptOverlay — project-directory trust prompt ──────────────────────
// Shown at TUI startup when the current directory (or an ancestor) contains
// trust-requiring resources (.logician/, extensions/, skills/, etc.).

import type { InkListOverlayModel } from "./ink-overlay-model.ts";

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
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		return {
			kind: "list",
			title: "Trust this workspace?",
			subtitle: this.cwd ? ` · ${this.cwd}` : undefined,
			hints: "↑↓ select · enter confirm · esc deny",
			headerLines: this.paths.length ? this.paths : undefined,
			items: OPTIONS.map((option, index) => ({
				label: option.label,
				metadata: option.description,
				selected: index === this.selectedIndex,
			})),
			emptyText: "No trust choices available.",
			footer: "Only trust folders whose contents you understand.",
			selectedIndex: this.selectedIndex,
		};
	}

	// ── Rendering ─────────────────────────────────────────────────────────────

}
