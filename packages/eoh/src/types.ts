// ── EoH (Evolution of Heuristics) types ──────────────────────────────────────
// Data structures for the EoH algorithm (arxiv 2401.02051).
// A heuristic = {thought (natural language), code (executable fn), fitness}.

export interface Heuristic {
	id: string;
	thought: string;
	code: string;
	fitness: number;
	generation: number;
	createdBy: EohOperator;
	parentIds: string[];
}

export type EohOperator =
	| "init"
	| "e1_diversity"
	| "e2_convergence"
	| "m1_improve"
	| "m2_tune"
	| "m3_simplify";

export interface EohProblem {
	/** Short identifier, used in prompts. */
	name: string;
	/** Full problem description fed to LLM. */
	description: string;
	/** Expected function signature the LLM must produce. */
	functionSignature: string;
	/** Problem instances used to evaluate fitness (JSON-serializable). */
	instances: unknown[];
	/**
	 * Evaluate one heuristic on one instance. Returns a scalar score (higher = better).
	 * Called per-instance; fitness = mean across all instances.
	 */
	evaluateInstance: (fnCode: string, instance: unknown) => Promise<number>;
}

export interface EohConfig {
	/** Population size N. */
	populationSize: number;
	/** Number of parents p for E1/E2 operators. */
	numParents: number;
	/** Max generations to run (0 = unlimited, stopped manually). */
	maxGenerations: number;
	/** Fitness evaluation timeout per instance (ms). */
	evalTimeoutMs: number;
	/** LLM model to use for heuristic generation. */
	model?: string;
	/** LLM base URL. */
	baseUrl?: string;
}

export interface EohState {
	population: Heuristic[];
	generation: number;
	totalLLMCalls: number;
	problem: EohProblem | null;
	config: EohConfig;
	running: boolean;
}

export interface EohGenerateResult {
	thought: string;
	code: string;
}

// ── JSONL persistence types ──────────────────────────────────────────────────

export interface EohRunEntry {
	run: number;
	thought: string;
	code: string;
	fitness: number;
	generation: number;
	createdBy: EohOperator;
	parentIds: string[];
	status: "keep" | "discard" | "crash";
	description: string;
	timestamp: number;
	segment: number;
}

export interface EohConfigEntry {
	type: "eoh_config";
	name?: string;
	populationSize?: number;
	maxGenerations?: number;
	bestDirection?: "lower" | "higher";
}

export interface ReconstructedEohState {
	name: string | null;
	populationSize: number;
	maxGenerations: number;
	bestDirection: "lower" | "higher";
	currentSegment: number;
	results: EohRunEntry[];
}
