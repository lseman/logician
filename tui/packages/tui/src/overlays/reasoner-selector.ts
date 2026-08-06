// ── ReasonerSelectorOverlay ──────────────────────────────────────────────────────
// Overlay for selecting an active reasoning mode.
// Reasoner selection applies to the next turn (never mutates an in-flight run).

import { ListSelectorOverlay, type ListItem } from "./popup-utils.ts";

export interface ReasonerInfo {
	id: string;
	name: string;
	description: string;
	active: boolean;
}

export type ReasonerSelectorAction =
	| { type: "select"; reasoner: ReasonerInfo }
	| { type: "close" };

export class ReasonerSelectorOverlay extends ListSelectorOverlay<ReasonerInfo> {
	constructor() {
		super({
			title: "Reasoning Mode",
			emptyText: "No reasoning modes available.",
			defaultMessage: "Select a reasoning mode for the next turn.",
			toItem: (r, i, selectedIndex): ListItem => ({
				label: r.name,
				metadata: r.active ? `${r.description}  active` : r.description,
				selected: i === selectedIndex,
				statusDot: r.active ? "active" : undefined,
			}),
		});
	}

	setReasoners(reasoners: ReasonerInfo[]): void {
		this.setItems(reasoners);
	}

	handleInput(data: string): ReasonerSelectorAction | null {
		const action = this.handleListInput(data);
		if (!action) return null;
		return action.type === "select" ? { type: "select", reasoner: action.item } : action;
	}
}
