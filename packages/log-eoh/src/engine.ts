/**
 * EoH Evolution Engine.
 *
 * Core EoH algorithm loop (arxiv 2401.02051):
 *   1. Initialize population N heuristics via LLM
 *   2. Each generation: apply 5 operators × N parents = 5N candidates
 *   3. Evaluate fitness on instance set
 *   4. Keep top-N into next generation
 */

import {
	evaluateHeuristic,
	parseHeuristicOutput,
	validateCode,
} from "./evaluator.ts";
import { callLLM } from "./llm.ts";
import {
	populationStats,
	rankPopulation,
	selectNextGeneration,
	selectParents,
} from "./population.ts";
import {
	promptE1Diversity,
	promptE2Convergence,
	promptInit,
	promptM1Improve,
	promptM2Tune,
	promptM3Simplify,
} from "./prompts.ts";
import type {
	EohConfig,
	EohOperator,
	EohProblem,
	EohState,
	Heuristic,
} from "./types.ts";

let _idCounter = 0;
function newId(): string {
	return `h_${Date.now()}_${_idCounter++}`;
}

export type EohProgressEvent =
	| { type: "generation_start"; generation: number }
	| {
			type: "generation_end";
			generation: number;
			stats: ReturnType<typeof populationStats>;
	  }
	| { type: "heuristic_evaluated"; heuristic: Heuristic; operator: EohOperator }
	| { type: "heuristic_failed"; reason: string; operator: EohOperator }
	| { type: "init_complete"; population: Heuristic[] }
	| { type: "stopped" };

export type EohProgressHandler = (event: EohProgressEvent) => void;

export class EohEngine {
	private state: EohState;
	private onProgress?: EohProgressHandler;
	private stopRequested = false;

	constructor(config: Partial<EohConfig> = {}) {
		this.state = {
			population: [],
			generation: 0,
			totalLLMCalls: 0,
			problem: null,
			config: {
				populationSize: config.populationSize ?? 10,
				numParents: config.numParents ?? 3,
				maxGenerations: config.maxGenerations ?? 0,
				evalTimeoutMs: config.evalTimeoutMs ?? 10_000,
				model: config.model,
				baseUrl: config.baseUrl,
			},
			running: false,
		};
	}

	setProblem(problem: EohProblem): void {
		this.state.problem = problem;
		this.state.population = [];
		this.state.generation = 0;
	}

	setProgressHandler(handler: EohProgressHandler): void {
		this.onProgress = handler;
	}

	getState(): Readonly<EohState> {
		return this.state;
	}

	getBestHeuristic(): Heuristic | null {
		const ranked = rankPopulation(this.state.population);
		return ranked[0] ?? null;
	}

	setMaxGenerations(maxGenerations: number): void {
		if (!Number.isInteger(maxGenerations) || maxGenerations < 0) {
			throw new Error("maxGenerations must be a non-negative integer");
		}
		this.state.config.maxGenerations = maxGenerations;
	}

	/** Seed evolution from an existing, already evaluated heuristic. */
	seedHeuristic(
		code: string,
		fitness: number,
		thought = "Current implementation",
	): Heuristic {
		if (!Number.isFinite(fitness))
			throw new Error("Seed fitness must be finite");
		const heuristic: Heuristic = {
			id: newId(),
			thought,
			code,
			fitness,
			generation: 0,
			createdBy: "init",
			parentIds: [],
		};
		this.state.population = [heuristic];
		return heuristic;
	}

	stop(): void {
		this.stopRequested = true;
	}

	/** Run initialization: generate N heuristics from scratch. */
	async initialize(baseUrl: string, model: string): Promise<void> {
		if (!this.state.problem) throw new Error("No problem set");
		this.state.config.baseUrl = baseUrl;
		this.state.config.model = model;

		const { populationSize, evalTimeoutMs } = this.state.config;
		const problem = this.state.problem;
		const population: Heuristic[] = [];

		for (let i = 0; i < populationSize && !this.stopRequested; i++) {
			const h = await this.generateHeuristic(
				"init",
				promptInit(
					problem,
					population.map(h => h.thought),
				),
				[],
				problem,
				evalTimeoutMs,
			);
			if (h) population.push(h);
		}

		this.state.population = rankPopulation(population);
		this.emit({ type: "init_complete", population: this.state.population });
	}

	/** Run one generation of evolution. Returns new candidates generated. */
	async runGeneration(): Promise<Heuristic[]> {
		if (!this.state.problem) throw new Error("No problem set");
		if (!this.state.config.baseUrl || !this.state.config.model)
			throw new Error("No LLM config");
		if (this.state.population.length === 0)
			throw new Error("Population empty — call initialize() first");

		const { numParents, evalTimeoutMs } = this.state.config;
		const problem = this.state.problem;
		const sorted = rankPopulation(this.state.population);
		const candidates: Heuristic[] = [];

		this.state.generation++;
		this.emit({ type: "generation_start", generation: this.state.generation });

		// 5 operators, each applied to the full population (N parents sampled per call)
		const operatorRuns: Array<{ op: EohOperator; parentsNeeded: number }> = [
			{ op: "e1_diversity", parentsNeeded: numParents },
			{ op: "e2_convergence", parentsNeeded: numParents },
			{ op: "m1_improve", parentsNeeded: 1 },
			{ op: "m2_tune", parentsNeeded: 1 },
			{ op: "m3_simplify", parentsNeeded: 1 },
		];

		for (const { op, parentsNeeded } of operatorRuns) {
			if (this.stopRequested) break;
			// Run N candidates per operator (one per population slot)
			for (
				let i = 0;
				i < this.state.config.populationSize && !this.stopRequested;
				i++
			) {
				const parents = selectParents(sorted, parentsNeeded);
				const messages = buildOperatorPrompt(op, problem, parents);
				const h = await this.generateHeuristic(
					op,
					messages,
					parents,
					problem,
					evalTimeoutMs,
				);
				if (h) candidates.push(h);
			}
		}

		this.state.population = selectNextGeneration(
			this.state.population,
			candidates,
			this.state.config.populationSize,
		);

		this.emit({
			type: "generation_end",
			generation: this.state.generation,
			stats: populationStats(this.state.population),
		});

		return candidates;
	}

	/** Run evolution loop until maxGenerations or stopped. */
	async run(baseUrl: string, model: string): Promise<void> {
		this.stopRequested = false;
		this.state.running = true;
		try {
			if (this.state.population.length === 0) {
				await this.initialize(baseUrl, model);
			}
			const { maxGenerations } = this.state.config;
			while (
				!this.stopRequested &&
				(maxGenerations === 0 || this.state.generation < maxGenerations)
			) {
				await this.runGeneration();
			}
		} finally {
			this.state.running = false;
			this.emit({ type: "stopped" });
		}
	}

	// ── Internal ──────────────────────────────────────────────────────────────

	private async generateHeuristic(
		operator: EohOperator,
		messages: Array<{ role: string; content: string }>,
		parents: Heuristic[],
		problem: EohProblem,
		evalTimeoutMs: number,
	): Promise<Heuristic | null> {
		try {
			const raw = await callLLM({
				baseUrl: this.state.config.baseUrl!,
				model: this.state.config.model!,
				messages,
				temperature: 0.8,
				maxTokens: 2048,
			});
			this.state.totalLLMCalls++;

			const parsed = parseHeuristicOutput(raw);
			if (!parsed) {
				this.emit({
					type: "heuristic_failed",
					reason: "parse failed",
					operator,
				});
				return null;
			}

			const sigLine = problem.functionSignature
				.trim()
				.split("(")[0]
				.replace("def ", "")
				.trim();
			const validationError = validateCode(parsed.code, sigLine);
			if (validationError) {
				this.emit({
					type: "heuristic_failed",
					reason: `validation: ${validationError}`,
					operator,
				});
				return null;
			}

			const fitness = await evaluateHeuristic(
				parsed.code,
				problem,
				evalTimeoutMs,
			);
			if (!Number.isFinite(fitness)) {
				this.emit({
					type: "heuristic_failed",
					reason: "eval failed (non-finite fitness)",
					operator,
				});
				return null;
			}

			const heuristic: Heuristic = {
				id: newId(),
				thought: parsed.thought,
				code: parsed.code,
				fitness,
				generation: this.state.generation,
				createdBy: operator,
				parentIds: parents.map(p => p.id),
			};

			this.emit({ type: "heuristic_evaluated", heuristic, operator });
			return heuristic;
		} catch (err) {
			const reason = err instanceof Error ? err.message : String(err);
			this.emit({ type: "heuristic_failed", reason, operator });
			return null;
		}
	}

	private emit(event: EohProgressEvent): void {
		this.onProgress?.(event);
	}
}

function buildOperatorPrompt(
	op: EohOperator,
	problem: EohProblem,
	parents: Heuristic[],
): Array<{ role: string; content: string }> {
	switch (op) {
		case "e1_diversity":
			return promptE1Diversity(problem, parents);
		case "e2_convergence":
			return promptE2Convergence(problem, parents);
		case "m1_improve":
			return promptM1Improve(problem, parents[0]);
		case "m2_tune":
			return promptM2Tune(problem, parents[0]);
		case "m3_simplify":
			return promptM3Simplify(problem, parents[0]);
		case "init":
			return promptInit(problem, []);
	}
}
