// ── ReasonerSelectorOverlay ──────────────────────────────────────────────────────
// Overlay for selecting an active reasoning mode.
// Mirrors PluginManagerOverlay pattern: list, select, confirm, close.
// Reasoner selection applies to the next turn (never mutates an in-flight run).

import { type Component } from "../layers/core/tui-core.ts";
import { SelectorController } from "./selector-controller.ts";
import {
	renderListItem,
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
	parsePopupListNav,
	renderListPopupFrame,
	renderListPopupBody,
	type ListItem,
} from "./popup-utils.ts";

export interface ReasonerInfo {
	id: string;
	name: string;
	description: string;
	active: boolean;
}

export type ReasonerSelectorAction =
	| { type: "select"; reasoner: ReasonerInfo }
	| { type: "close" };

export class ReasonerSelectorOverlay implements Component {
	public visible = false;
	private reasoners: ReasonerInfo[] = [];
	private selection = new SelectorController();
	private message = "";
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;

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

		const bodyLines = renderListPopupBody(
			this.reasoners,
			this.selection,
			innerWidth,
			10,
			(r, i) => {
				const item: ListItem = {
					label: r.name,
					metadata: r.active ? `${r.description}  active` : r.description,
					selected: i === this.selection.index,
					statusDot: r.active ? "active" : undefined,
				};
				return renderListItem(item, innerWidth);
			},
			"No reasoning modes available.",
		);

		const lines = renderListPopupFrame({
			popupWidth,
			innerWidth,
			title: "Reasoning Mode",
			subtitle: ` (${this.reasoners.length})`,
			hints: " ↑↓ select · enter confirm · esc close",
			bodyLines,
			bottomText: this.message || "Select a reasoning mode for the next turn.",
		});

		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}

	private moveSelection(delta: number): void {
		const n = this.reasoners.length;
		if (!n) return;
		this.selection.move(delta, n);
		this.invalidate();
	}
}
