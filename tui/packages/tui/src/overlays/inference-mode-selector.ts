// ── InferenceModeSelector — inference-mode selection popup ──────────────────
// Rounded-corner overlay for selecting an inference mode from the configured
// list. Uses the shared popup-utils design system.

import { type ListItem, ListSelectorOverlay } from "./popup-utils.ts";

// Re-exported from agent-core; imported here to avoid circular dep.
export interface InferenceModeDef {
	label: string;
	description: string;
	thinking: boolean;
	useProviderDefaults: boolean;
	params: {
		temperature: number;
		top_p: number;
		top_k: number;
		min_p: number;
		presence_penalty: number;
		repetition_penalty: number;
	};
}

export interface InferenceModeInfo {
	id: string;
	label: string;
	description: string;
	thinking: boolean;
	useProviderDefaults: boolean;
	params: InferenceModeDef["params"];
}

export type InferenceModeSelectorAction =
	| { type: "select"; mode: InferenceModeInfo }
	| { type: "close" };

const MODE_ORDER = [
	"auto",
	"none",
	"thinking-general",
	"thinking-coding",
	"instruct-general",
	"instruct-reasoning",
	"instruct-coding",
	"deterministic",
	"creative",
	"analytical",
] as const;

export class InferenceModeSelector extends ListSelectorOverlay<InferenceModeInfo> {
	private _activeId = "instruct-general";

	constructor() {
		super({
			title: "Inference Mode",
			emptyText: "No inference modes configured.",
			defaultMessage: "Select an inference mode for this session.",
			toItem: (m, i, selectedIndex): ListItem => ({
				label: m.label,
				metadata: m.description,
				selected: i === selectedIndex,
				current: m.id === this._activeId,
			}),
		});
	}

	setModes(modes: InferenceModeInfo[], activeId: string): void {
		this._activeId = activeId;
		const activeIndex = modes.findIndex(m => m.id === activeId);
		this.setItems(modes, activeIndex >= 0 ? activeIndex : this.selection.index);
	}

	handleInput(data: string): InferenceModeSelectorAction | null {
		const action = this.handleListInput(data);
		if (!action) return null;
		return action.type === "select"
			? { type: "select", mode: action.item }
			: action;
	}
}

/** Return modes sorted by the canonical order, with unknown ones appended. */
export function sortInferenceModesByIds(ids: string[]): string[] {
	const sorted = [...ids].sort((a, b) => {
		const ai = MODE_ORDER.indexOf(a as (typeof MODE_ORDER)[number]);
		const bi = MODE_ORDER.indexOf(b as (typeof MODE_ORDER)[number]);
		if (ai >= 0 && bi >= 0) return ai - bi;
		if (ai >= 0) return -1;
		if (bi >= 0) return 1;
		return a.localeCompare(b);
	});
	return sorted;
}
