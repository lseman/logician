// ── ReasonerSelectorOverlay ──────────────────────────────────────────────────────
// Overlay for selecting an active reasoning mode.
// Mirrors PluginManagerOverlay pattern: list, select, confirm, close.
// Reasoner selection applies to the next turn (never mutates an in-flight run).

import { SelectorController } from "./selector-controller.ts";
import type { InkListOverlayModel } from "./ink-overlay-model.ts";
import { parsePopupListNav } from "./popup-utils.ts";

export interface ReasonerInfo {
	id: string;
	name: string;
	description: string;
	active: boolean;
}

export type ReasonerSelectorAction =
	| { type: "select"; reasoner: ReasonerInfo }
	| { type: "close" };

export class ReasonerSelectorOverlay {
	public visible = false;
	private reasoners: ReasonerInfo[] = [];
	private selection = new SelectorController();
	private message = "";

	setReasoners(reasoners: ReasonerInfo[]): void {
		this.reasoners = reasoners;
		this.selection.set(this.selection.index, this.reasoners.length);
		this.invalidate();
	}

	setMessage(message: string): void {
		this.message = message;
		this.invalidate();
	}

	show(): void {
		this.visible = true;
		this.invalidate();
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): ReasonerSelectorAction | null {
		if (!this.visible) return null;

		const nav = parsePopupListNav(data);
		if (nav?.type === "close") return { type: "close" };
		if (nav?.type === "confirm") {
			const reasoner = this.reasoners[this.selection.index];
			return reasoner ? { type: "select", reasoner } : { type: "close" };
		}
		if (nav?.type === "move") {
			this.moveSelection(nav.delta);
		}
		return null;
	}

	invalidate(): void {
		// State is read directly by the Ink renderer.
	}

	getInkOverlayModel(): InkListOverlayModel {
		return {
			kind: "list",
			title: "Reasoning Mode",
			subtitle: ` (${this.reasoners.length})`,
			hints: "↑↓ select · enter confirm · esc close",
			items: this.reasoners.map((reasoner, index) => ({
				label: reasoner.name,
				metadata: reasoner.active
					? `${reasoner.description}  active`
					: reasoner.description,
				selected: index === this.selection.index,
				current: reasoner.active,
			})),
			emptyText: "No reasoning modes available.",
			footer: this.message || "Select a reasoning mode for the next turn.",
			selectedIndex: this.selection.index,
		};
	}

	private moveSelection(delta: number): void {
		const n = this.reasoners.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
