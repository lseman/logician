// ── Inference Modes ────────────────────────────────────────────────────────────
// Predefined sampling parameter sets toggled via Ctrl+M.
// Each mode packs temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty.

export type InferenceMode =
	| "thinking-general"
	| "thinking-coding"
	| "instruct-general"
	| "instruct-reasoning";

export interface SamplingParams {
	temperature: number;
	top_p: number;
	top_k: number;
	min_p: number;
	presence_penalty: number;
	repetition_penalty: number;
}

export interface InferenceModeDef {
	/** Human-readable label shown in the status bar and /settings output. */
	label: string;
	/** Short description for the mode selector. */
	description: string;
	/** Whether this mode enables chain-of-thought / extended thinking. */
	thinking: boolean;
	/** The sampling parameters to send with each model request. */
	params: SamplingParams;
}

export const INFERENCE_MODES: ReadonlyMap<InferenceMode, InferenceModeDef> = new Map([
	[
		"thinking-general",
		{
			label: "Think Gen",
			description: "General thinking — high creativity, strong diversity push.",
			thinking: true,
			params: {
				temperature: 1.0,
				top_p: 0.95,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.5,
				repetition_penalty: 1.0,
			},
		},
	],
	[
		"thinking-coding",
		{
			label: "Think Code",
			description: "Precise coding — lower temperature, no presence penalty.",
			thinking: true,
			params: {
				temperature: 0.6,
				top_p: 0.95,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 0.0,
				repetition_penalty: 1.0,
			},
		},
	],
	[
		"instruct-general",
		{
			label: "Instruct",
			description: "Non-thinking general tasks — balanced sampling.",
			thinking: false,
			params: {
				temperature: 0.7,
				top_p: 0.8,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.5,
				repetition_penalty: 1.0,
			},
		},
	],
	[
		"instruct-reasoning",
		{
			label: "Reason",
			description: "Non-thinking reasoning — high temperature for exploration.",
			thinking: false,
			params: {
				temperature: 1.0,
				top_p: 0.95,
				top_k: 20,
				min_p: 0.0,
				presence_penalty: 1.5,
				repetition_penalty: 1.0,
			},
		},
	],
]);

const MODE_ORDER: InferenceMode[] = [
	"thinking-general",
	"thinking-coding",
	"instruct-general",
	"instruct-reasoning",
];

/** Default mode when no explicit mode is set. */
export const DEFAULT_MODE: InferenceMode = "instruct-general";

/** Get mode definition by name, or undefined if unknown. */
export function getInferenceMode(mode: InferenceMode): InferenceModeDef | undefined {
	return INFERENCE_MODES.get(mode);
}

/** Cycle to the next inference mode. Returns the new mode name. */
export function cycleInferenceMode(current: InferenceMode): InferenceMode {
	const idx = MODE_ORDER.indexOf(current);
	const nextIdx = (idx + 1) % MODE_ORDER.length;
	return MODE_ORDER[nextIdx];
}

/** Check if a string is a valid inference mode name. */
export function isValidInferenceMode(v: string): v is InferenceMode {
	return INFERENCE_MODES.has(v as InferenceMode);
}
