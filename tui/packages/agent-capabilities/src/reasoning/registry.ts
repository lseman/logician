// ── Reasoner Registry ──────────────────────────────────────────────────────────
// Adapted from Python src/reasoners/registry.py.
//
// Factory: get_reasoner(name, llm, **config) → Reasoner instance.

import type { LLMBackend } from "@logician/agent-core/agent/backend.ts";
import { AutoCoTReasoner } from "./auto-cot.js";
import type { Reasoner, ReasonerConfig, ReasonerConstructor } from "./base.js";
import { BestOfNReasoner } from "./best-of-n.js";
import { CoVeReasoner } from "./cover.js";
import { GoTReasoner } from "./got.js";
import { InContextCoTReasoner } from "./in-context-cot.js";
import { ReflexionReasoner } from "./reflexion.js";
import { SelfConsistencyReasoner } from "./self-consistency.js";
import { SSRReasoner } from "./ssr.js";
import { ToTReasoner } from "./tot.js";

// ── Reasoner metadata ────────────────────────────────────────────────────────
// Each entry: { name, description, defaultConfig }

export interface ReasonerMeta {
	name: string;
	description: string;
	defaultConfig: ReasonerConfig;
}

export const REASONER_METADATA: Record<string, ReasonerMeta> = {
	none: {
		name: "None",
		description: "Default ReAct loop, no extra reasoning phase",
		defaultConfig: {},
	},
	ssr: {
		name: "Socratic Self-Refinement",
		description: "Decompose → verify steps → refine weakest (SSR)",
		defaultConfig: { maxIterations: 3, mSamples: 8 },
	},
	tot: {
		name: "Tree of Thoughts",
		description: "Beam search over reasoning paths with scoring",
		defaultConfig: { beamWidth: 6, maxDepth: 10, branchFactor: 3 },
	},
	got: {
		name: "Graph of Thoughts",
		description:
			"Graph-based reasoning with divergence, convergence, and pruning",
		defaultConfig: {
			beamWidth: 6,
			maxDepth: 10,
			branchFactor: 3,
			mergeThreshold: 0.7,
		},
	},
	reflexion: {
		name: "Reflexion",
		description: "Attempt → critique → rewrite loop",
		defaultConfig: { maxTrials: 3 },
	},
	self_consistency: {
		name: "Self-Consistency",
		description: "N independent samples, majority vote",
		defaultConfig: { nRollouts: 32, temperature: 0.8 },
	},
	best_of_n: {
		name: "Best-of-N",
		description: "N samples, LLM-scored, pick best",
		defaultConfig: { n: 8, temperature: 0.8 },
	},
	auto_cot: {
		name: "Auto-CoT",
		description: "Generate exemplars, then answer with few-shot",
		defaultConfig: {},
	},
	in_context_cot: {
		name: "In-Context CoT",
		description: "Hand-crafted CoT examples as few-shot",
		defaultConfig: {},
	},
	cover: {
		name: "Chain of Verification",
		description: "Generate response, plan verification steps, execute, generate final verified response",
		defaultConfig: { maxVerificationSteps: 3 },
	},
};

export const REASONER_REGISTRY: Record<string, ReasonerConstructor> = {
	ssr: SSRReasoner,
	tot: ToTReasoner,
	got: GoTReasoner,
	reflexion: ReflexionReasoner,
	self_consistency: SelfConsistencyReasoner,
	sc: SelfConsistencyReasoner,
	best_of_n: BestOfNReasoner,
	auto_cot: AutoCoTReasoner,
	in_context_cot: InContextCoTReasoner,
	cover: CoVeReasoner,
};

/** Get all registered reasoner IDs (including "none"). */
export function getReasonerIds(): string[] {
	return Object.keys(REASONER_METADATA);
}

/** Get metadata for a reasoner by ID. */
export function getReasonerMeta(id: string): ReasonerMeta | undefined {
	return REASONER_METADATA[id];
}

export function get_reasoner(
	name: string,
	llm: LLMBackend,
	config: ReasonerConfig = {},
): Reasoner {
	const cls = REASONER_REGISTRY[name.toLowerCase()];
	if (!cls) {
		throw new Error(
			`Unknown reasoner '${name}'. Registered: ${Object.keys(REASONER_REGISTRY).join(", ")}`,
		);
	}
	return new cls(llm, config);
}
