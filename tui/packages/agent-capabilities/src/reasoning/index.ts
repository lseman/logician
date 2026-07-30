// ── Reasoners Barrel Export ────────────────────────────────────────────────────
// Opt-in structured reasoning. Import only when needed.

export { AutoCoTReasoner } from "./auto-cot.ts";
export {
	BaseReasoner,
	type Reasoner,
	type ReasonerConfig,
	type ReasoningTrace,
} from "./base.ts";
export { BestOfNReasoner } from "./best-of-n.ts";
export { CoVeReasoner } from "./cover.ts";
export { GoTReasoner } from "./got.ts";
export { InContextCoTReasoner } from "./in-context-cot.ts";
export { ReflexionReasoner } from "./reflexion.ts";
export {
	get_reasoner,
	getReasonerIds,
	getReasonerMeta,
	REASONER_METADATA,
	REASONER_REGISTRY,
	type ReasonerMeta,
} from "./registry.ts";
export { SelfConsistencyReasoner } from "./self-consistency.ts";
export { type SocraticStep, SSRReasoner } from "./ssr.ts";
export { ToTReasoner } from "./tot.ts";
