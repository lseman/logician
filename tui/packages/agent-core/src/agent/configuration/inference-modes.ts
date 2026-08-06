// ── Inference Modes ────────────────────────────────────────────────────────────
// Predefined sampling parameter sets toggled via Ctrl+M.
// Each mode packs temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty.

export type InferenceMode =
	| "auto"
	| "thinking-general"
	| "thinking-coding"
	| "instruct-general"
	| "instruct-reasoning"
	| "instruct-coding"
	| "deterministic"
	| "creative"
	| "analytical";

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

export const INFERENCE_MODES: ReadonlyMap<InferenceMode, InferenceModeDef> =
	new Map([
		[
			"auto",
			{
				label: "Auto",
				description:
					"Automatically selects a preset from the live task phase and evidence.",
				thinking: true,
				// Adaptive is resolved before requests; these are safe fallback values.
				params: {
					temperature: 0.7,
					top_p: 0.8,
					top_k: 20,
					min_p: 0.0,
					presence_penalty: 1.0,
					repetition_penalty: 1.0,
				},
			},
		],
		[
			"thinking-general",
			{
				label: "Think Gen",
				description:
					"General thinking — high creativity, strong diversity push.",
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
				description:
					"Non-thinking reasoning — high temperature for exploration.",
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
		[
			"instruct-coding",
			{
				label: "Code",
				description:
					"Non-thinking coding — low temperature, no presence penalty for precise output.",
				thinking: false,
				params: {
					temperature: 0.3,
					top_p: 0.9,
					top_k: 20,
					min_p: 0.0,
					presence_penalty: 0.0,
					repetition_penalty: 1.0,
				},
			},
		],
		[
			"deterministic",
			{
				label: "Exact",
				description:
					"Deterministic — near-zero temperature for reproducible, factual outputs.",
				thinking: false,
				params: {
					temperature: 0.0,
					top_p: 0.0,
					top_k: 1,
					min_p: 0.0,
					presence_penalty: 0.0,
					repetition_penalty: 1.0,
				},
			},
		],
		[
			"creative",
			{
				label: "Creative",
				description:
					"High creativity — ultra-high temperature for brainstorming and ideation.",
				thinking: false,
				params: {
					temperature: 1.3,
					top_p: 0.99,
					top_k: 40,
					min_p: 0.0,
					presence_penalty: 2.0,
					repetition_penalty: 0.9,
				},
			},
		],
		[
			"analytical",
			{
				label: "Analyze",
				description:
					"Careful analysis — low temperature, tight top_p for code review and comparison.",
				thinking: false,
				params: {
					temperature: 0.2,
					top_p: 0.7,
					top_k: 20,
					min_p: 0.0,
					presence_penalty: 0.5,
					repetition_penalty: 1.1,
				},
			},
		],
	]);

const MODE_ORDER: InferenceMode[] = [
	"auto",
	"thinking-general",
	"thinking-coding",
	"instruct-general",
	"instruct-reasoning",
	"instruct-coding",
	"deterministic",
	"creative",
	"analytical",
];

/** Default mode when no explicit mode is set. */
export const DEFAULT_MODE: InferenceMode = "instruct-general";

/** Get mode definition by name, or undefined if unknown. */
export function getInferenceMode(
	mode: InferenceMode,
): InferenceModeDef | undefined {
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
