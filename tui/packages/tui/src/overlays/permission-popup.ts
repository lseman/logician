// ── PermissionPopup — tool permission overlay popup ──────────────────────────
// Rounded-corner overlay for tool permission requests with selectable
// allow-once / always-allow / deny options.

import type { InkListOverlayModel } from "./ink-overlay-model.ts";

export interface PermissionChoice {
	/** Value sent back to the agent: "allow" | "always" | "deny" */
	value: "allow" | "always" | "deny";
	/** Display label */
	label: string;
	/** Short description */
	description?: string;
}

export type PermissionPopupAction =
	| { type: "select"; choice: PermissionChoice }
	| { type: "close" };

export class PermissionPopup {
	private toolName = "";
	private toolArgs = "";
	private choices: PermissionChoice[] = [];
	private selectedIndex = 0;
	public visible = false;

	setToolInfo(toolName: string, toolArgs: string): void {
		this.toolName = toolName;
		this.toolArgs = toolArgs;
		this.selectedIndex = 0;
		this.invalidate();
	}

	setChoices(choices: PermissionChoice[]): void {
		this.choices = choices;
		this.selectedIndex = 0;
		this.invalidate();
	}

	getSelected(): PermissionChoice | null {
		if (this.choices.length === 0) return null;
		return this.choices[this.selectedIndex];
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	// ── Navigation ─────────────────────────────────────────────────────────

	moveSelection(delta: number): void {
		if (this.choices.length === 0) return;
		this.selectedIndex =
			(this.selectedIndex + delta + this.choices.length) % this.choices.length;
		this.invalidate();
	}

	// ── Input handling (called from the TUI input listener) ─────────────────

	handleInput(data: string): PermissionPopupAction | null {
		if (!this.visible) return null;

		if (data === "\x1b" || data === "\x03") {
			return { type: "close" };
		}

		if (data === "\r" || data === "\n") {
			const selected = this.getSelected();
			if (selected) return { type: "select", choice: selected };
			return null;
		}

		if (data === "\x1b[A" || data === "\x1bOA") {
			this.moveSelection(-1);
			return null;
		}

		if (data === "\x1b[B" || data === "\x1bOB") {
			this.moveSelection(1);
			return null;
		}

		// Number keys 1-9 — select that option directly
		if (data.length === 1) {
			const c = data.charCodeAt(0);
			if (c >= 0x31 && c <= 0x39) {
				const idx = c - 0x31;
				if (idx < this.choices.length) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
				return null;
			}
		}

		// Single-letter shortcuts: a=allow, v=always, n=deny
		if (data.length === 1) {
			const lower = data.toLowerCase();
			if (lower === "a" && this.choices.some((c) => c.value === "allow")) {
				const idx = this.choices.findIndex((c) => c.value === "allow");
				if (idx !== -1) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
			}
			if (lower === "v" && this.choices.some((c) => c.value === "always")) {
				const idx = this.choices.findIndex((c) => c.value === "always");
				if (idx !== -1) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
			}
			if (lower === "n" && this.choices.some((c) => c.value === "deny")) {
				const idx = this.choices.findIndex((c) => c.value === "deny");
				if (idx !== -1) {
					this.selectedIndex = idx;
					return { type: "select", choice: this.choices[idx] };
				}
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
			title: "Permission Required",
			subtitle: this.toolName ? ` · ${this.toolName}` : undefined,
			hints: "↑↓ select · enter confirm · esc deny",
			headerLines: this.toolArgs ? [this.toolArgs] : undefined,
			items: this.choices.map((choice, index) => ({
				label: choice.label,
				metadata: choice.description,
				selected: index === this.selectedIndex,
			})),
			emptyText: "No permission choices available.",
			footer: "Choose how Logician may run this tool.",
			selectedIndex: this.selectedIndex,
		};
	}

}
