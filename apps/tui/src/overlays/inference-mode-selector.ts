// ── InferenceModeSelector — inference-mode selection popup ──────────────────
// Uses the shared ListSelectorOverlay factory; inference mode definitions live
// here (not in the controller) so consumers don't duplicate the config.

import { createListSelector, type ListSelectorOverlay } from "./popup-utils.ts";

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
	| { type: "select"; item: InferenceModeInfo }
	| { type: "close" };

// Canonical ordering for display; unknown modes are appended alphabetically.
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

// ── Inference mode definitions ──────────────────────────────────────────────

export const INFERENCE_MODES: InferenceModeInfo[] = [
	{
		id: "auto",
		label: "Auto",
		description: "Auto-select from task phase",
		thinking: true,
		useProviderDefaults: false,
		params: {
			temperature: 0.7,
			top_p: 0.8,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 1.0,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "none",
		label: "Provider",
		description: "Let the provider use its own defaults",
		thinking: false,
		useProviderDefaults: true,
		params: {
			temperature: 0.7,
			top_p: 0.8,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 0.0,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "thinking-general",
		label: "Think Gen",
		description: "General thinking — high creativity",
		thinking: true,
		useProviderDefaults: false,
		params: {
			temperature: 1.0,
			top_p: 0.95,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 1.5,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "thinking-coding",
		label: "Think Code",
		description: "Precise coding — lower temp",
		thinking: true,
		useProviderDefaults: false,
		params: {
			temperature: 0.6,
			top_p: 0.95,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 0.0,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "instruct-general",
		label: "Instruct",
		description: "Non-thinking — balanced",
		thinking: false,
		useProviderDefaults: false,
		params: {
			temperature: 0.7,
			top_p: 0.8,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 1.5,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "instruct-reasoning",
		label: "Reason",
		description: "Non-thinking — high temp",
		thinking: false,
		useProviderDefaults: false,
		params: {
			temperature: 1.0,
			top_p: 0.95,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 1.5,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "instruct-coding",
		label: "Code",
		description: "Non-thinking — precise output",
		thinking: false,
		useProviderDefaults: false,
		params: {
			temperature: 0.3,
			top_p: 0.9,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 0.0,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "deterministic",
		label: "Exact",
		description: "Near-zero temp — reproducible",
		thinking: false,
		useProviderDefaults: false,
		params: {
			temperature: 0.0,
			top_p: 0.0,
			top_k: 1,
			min_p: 0.0,
			presence_penalty: 0.0,
			repetition_penalty: 1.0,
		},
	},
	{
		id: "creative",
		label: "Creative",
		description: "Ultra-high temp — brainstorm",
		thinking: false,
		useProviderDefaults: false,
		params: {
			temperature: 1.3,
			top_p: 0.99,
			top_k: 40,
			min_p: 0.0,
			presence_penalty: 2.0,
			repetition_penalty: 0.9,
		},
	},
	{
		id: "analytical",
		label: "Analyze",
		description: "Low temp — code review",
		thinking: false,
		useProviderDefaults: false,
		params: {
			temperature: 0.2,
			top_p: 0.7,
			top_k: 20,
			min_p: 0.0,
			presence_penalty: 0.5,
			repetition_penalty: 1.1,
		},
	},
];

export const InferenceModeSelector = createListSelector<InferenceModeInfo>({
	title: "Inference Mode",
	emptyText: "No inference modes configured.",
	defaultMessage: "Select an inference mode for this session.",
	toItem: function (
		this: ListSelectorOverlay<InferenceModeInfo>,
		m,
		i,
		selectedIndex,
	) {
		return {
			label: m.label,
			metadata: m.description,
			selected: i === selectedIndex,
			current: m.id === this.activeId,
		};
	},
});
