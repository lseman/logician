// ── ReasonerSelectorOverlay ──────────────────────────────────────────────────────
// Overlay for selecting an active reasoning mode.
// Reasoner selection applies to the next turn (never mutates an in-flight run).

import { createListSelector, type ListSelectorOverlay } from "./popup-utils.ts";

export interface ReasonerInfo {
	id: string;
	name: string;
	description: string;
	active: boolean;
}

export type ReasonerSelectorAction =
	| { type: "select"; item: ReasonerInfo }
	| { type: "close" };

export const ReasonerSelectorOverlay = createListSelector<ReasonerInfo>({
	title: "Reasoning Mode",
	emptyText: "No reasoning modes available.",
	defaultMessage: "Select a reasoning mode for the next turn.",
	toItem: function (
		this: ListSelectorOverlay<ReasonerInfo>,
		r,
		i,
		selectedIndex,
	) {
		return {
			label: r.name,
			metadata: r.active ? `${r.description}  active` : r.description,
			selected: i === selectedIndex,
			statusDot: r.active ? "active" : undefined,
		};
	},
});
