/** Owns the Evolution of Heuristics command lifecycle for one target file. */

import path from "node:path";
import type { RuntimeEvent } from "@logician/agent-protocol";
import { EohEngine, type EohProgressEvent } from "@logician/eoh/engine";
import { populationStats } from "@logician/eoh/population";
import {
	applyEohCandidate,
	type EohFileTarget,
	evaluateEohCandidate,
	loadEohFile,
} from "./file.ts";

export interface EohControllerDeps {
	cwd: string;
	emit: (event: RuntimeEvent) => void;
	getBaseUrl: () => string;
	getCurrentModel: () => string;
}

export class EohController {
	private engine: EohEngine | null = null;
	private progressHandler: ((event: EohProgressEvent) => void) | null = null;
	private target: EohFileTarget | null = null;
	private appliedFitness = Number.NEGATIVE_INFINITY;
	private generationImproved = false;
	private staleGenerations = 0;
	private preparing = false;
	private runId = 0;

	constructor(private readonly deps: EohControllerDeps) {}

	/** Initialize the engine if not already created. */
	private getEngine(): EohEngine {
		if (!this.engine) {
			this.engine = new EohEngine();
		}
		if (!this.progressHandler) {
			this.progressHandler = (event: EohProgressEvent) => {
				if (event.type === "generation_start") {
					this.generationImproved = false;
				}
				if (
					event.type === "heuristic_evaluated" &&
					this.target &&
					event.heuristic.fitness > this.appliedFitness
				) {
					try {
						applyEohCandidate(this.target, event.heuristic.code);
						this.appliedFitness = event.heuristic.fitness;
						this.generationImproved = true;
						this.deps.emit({
							type: "notice",
							level: "success",
							label: "EoH",
							text: `Applied fitness ${event.heuristic.fitness.toFixed(6)} · ${path.relative(this.deps.cwd, this.target.path)}`,
						});
					} catch (error) {
						this.deps.emit({
							type: "notice",
							level: "error",
							label: "EoH",
							text: `Could not apply candidate: ${error instanceof Error ? error.message : String(error)}`,
						});
					}
				}
				if (event.type === "generation_end") {
					this.staleGenerations = this.generationImproved
						? 0
						: this.staleGenerations + 1;
					if (this.staleGenerations >= 3) {
						this.engine?.stop();
						this.deps.emit({
							type: "notice",
							level: "info",
							label: "EoH",
							text: "Converged: no improvement for 3 generations.",
						});
					}
				}
				this.deps.emit({
					...event,
				} as unknown as RuntimeEvent);
			};
		}
		this.engine.setProgressHandler(this.progressHandler);
		return this.engine;
	}

	private async startFile(rawPath: string, generations: number): Promise<void> {
		const runId = ++this.runId;
		this.preparing = true;
		try {
			const target = await loadEohFile(rawPath, this.deps.cwd);
			if (runId !== this.runId) return;
			this.engine = new EohEngine({
				populationSize: 6,
				numParents: 3,
				maxGenerations: generations,
				evalTimeoutMs: 30_000,
			});
			this.target = target;
			this.appliedFitness = Number.NEGATIVE_INFINITY;
			this.generationImproved = false;
			this.staleGenerations = 0;
			this.progressHandler = null;
			const engine = this.getEngine();
			engine.setProblem({
				name: path.basename(target.path),
				description:
					`${target.description}\n\nImprove only the heuristic function. ` +
					"The file's evaluate(heuristic) function returns fitness; higher is better.",
				functionSignature: target.functionSignature,
				instances: [null],
				evaluateInstance: code => evaluateEohCandidate(target, code, 30_000),
			});
			const initialFitness = await evaluateEohCandidate(
				target,
				target.heuristicCode,
				30_000,
			);
			if (runId !== this.runId) return;
			engine.seedHeuristic(
				target.heuristicCode,
				initialFitness,
				"Current file implementation",
			);
			this.appliedFitness = initialFitness;
			this.deps.emit({
				type: "notice",
				level: "info",
				label: "EoH",
				text: `Baseline fitness ${initialFitness.toFixed(6)} · evolving ${path.relative(this.deps.cwd, target.path)}`,
			});
			const model = this.deps.getCurrentModel() || undefined;
			if (!model) throw new Error("No model configured for EoH");
			await engine.run(this.deps.getBaseUrl(), model);
		} catch (error) {
			this.deps.emit({
				type: "notice",
				level: "error",
				label: "EoH",
				text: error instanceof Error ? error.message : String(error),
			});
		} finally {
			if (runId === this.runId) this.preparing = false;
		}
	}

	/** EoH command: /eoh <file.py> [generations] | stop | status | best | reset */
	command(raw: string): string {
		const trimmed = raw.trim();
		if (!trimmed) {
			return [
				"EoH: Evolution of Heuristics (arxiv 2401.02051)",
				"Usage:",
				"  /eoh <heuristic.py> [generations] - Start/resume file evolution",
				"  /eoh stop                       - Stop evolution",
				"  /eoh status                     - Show current status",
				"  /eoh best                       - Show best heuristic",
				"  /eoh reset                      - Reset EoH state",
				"",
				"The file must wrap def heuristic(...) in # EOH-BEGIN / # EOH-END",
				"and define evaluate(heuristic) -> float after that region.",
			].join("\n");
		}

		const [action, ...rest] = trimmed.split(/\s+/);
		const args = rest.join(" ");

		switch (action.toLowerCase()) {
			case "start": {
				const [file, generationArg] = rest;
				if (!file) return "Usage: /eoh <heuristic.py> [generations]";
				const generations = Number.parseInt(generationArg ?? "", 10) || 20;
				if (this.preparing || this.engine?.getState().running) {
					return "EoH evolution already running";
				}
				void this.startFile(file, generations);
				return `Preparing ${file} · max ${generations} generations · convergence patience 3`;
			}

			case "stop": {
				this.runId++;
				this.preparing = false;
				const engine = this.getEngine();
				engine.stop();
				return "EoH stop signal sent";
			}

			case "status": {
				const engine = this.getEngine();
				const state = engine.getState();
				const stats = populationStats(state.population);
				return [
					"EoH Status",
					`  Running: ${state.running || this.preparing}`,
					`  Generation: ${state.generation}`,
					`  LLM calls: ${state.totalLLMCalls}`,
					`  Population: ${stats.size}`,
					`  Best fitness: ${stats.best.toFixed(4)}`,
					`  Mean fitness: ${stats.mean.toFixed(4)}`,
					`  Worst fitness: ${stats.worst.toFixed(4)}`,
					this.target
						? `  File: ${path.relative(this.deps.cwd, this.target.path)}`
						: "  File: none",
					`  Convergence: ${this.staleGenerations}/3 stale generations`,
				].join("\n");
			}

			case "best": {
				const engine = this.getEngine();
				const best = engine.getBestHeuristic();
				if (!best) return "No heuristics yet — run /eoh start first";
				return [
					`Best heuristic (fitness=${best.fitness.toFixed(4)}, gen=${best.generation}, by=${best.createdBy}):`,
					"",
					"Thought:",
					best.thought,
					"",
					"Code:",
					"```python",
					best.code,
					"```",
				].join("\n");
			}

			case "reset": {
				this.runId++;
				this.engine?.stop();
				this.engine = null;
				this.progressHandler = null;
				this.target = null;
				this.appliedFitness = Number.NEGATIVE_INFINITY;
				this.staleGenerations = 0;
				this.preparing = false;
				return "EoH state reset";
			}

			default: {
				if (!action.toLowerCase().endsWith(".py")) {
					return `Unknown EoH action: ${action}. Use /eoh for usage.`;
				}
				const generations = Number.parseInt(args, 10) || 20;
				if (this.preparing || this.engine?.getState().running) {
					return "EoH evolution already running";
				}
				void this.startFile(action, generations);
				return `Preparing ${action} · max ${generations} generations · convergence patience 3`;
			}
		}
	}
}
