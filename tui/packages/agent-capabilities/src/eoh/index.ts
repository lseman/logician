// ── Evolution of Heuristics (EoH) Extension ───────────────────────────────────
// Implements EoH (arxiv 2401.02051) as a Logician extension.
// Exposes:
//   - Tool: `evolve_heuristics` — LLM-callable to run/control evolution
//   - Commands: /eoh-start, /eoh-stop, /eoh-best, /eoh-status, /eoh-set-problem

import type { ExtensionAPI } from "@logician/agent-core/extensions/types.ts";
import { EohEngine } from "./engine.ts";
import type { EohProblem, EohConfig } from "./types.ts";
import { populationStats } from "./population.ts";

// ── Built-in demo problem: Online Bin Packing ─────────────────────────────────
// Fitness = lb / n (ratio to lower bound; 1.0 = optimal).

const BIN_PACKING_PROBLEM: EohProblem = {
	name: "Online Bin Packing",
	description: `Given items of various sizes (0 < size ≤ 1) arriving online, pack them into bins of capacity 1.0 using a heuristic function. The heuristic selects which existing open bin to place the current item in, or opens a new bin.`,
	functionSignature: `def select_bin(item_size: float, bins: list[float]) -> int:`,
	instances: generateBinPackingInstances(10),
	evaluateInstance: async (code: string, instance: unknown) => {
		return evalBinPackingHeuristic(code, instance as number[]);
	},
};

function generateBinPackingInstances(count: number): number[][] {
	const instances: number[][] = [];
	const rng = mulberry32(42);
	for (let i = 0; i < count; i++) {
		const items: number[] = [];
		const n = 20 + Math.floor(rng() * 30); // 20–50 items
		for (let j = 0; j < n; j++) {
			items.push(0.1 + rng() * 0.8); // sizes 0.1–0.9
		}
		instances.push(items);
	}
	return instances;
}

function mulberry32(seed: number): () => number {
	return function () {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function evalBinPackingHeuristic(code: string, items: number[]): number {
	// Build a JS-side interpreter for Python-like bin packing.
	// We don't execute Python; instead we check if the LLM produced valid logic
	// by translating the select_bin function heuristic into a JS equivalent.
	// For demo purposes: parse common patterns (First Fit, Best Fit, Worst Fit, etc.)
	// and simulate them. Full execution would require a Python subprocess.

	const bins: number[] = []; // remaining capacity per bin
	const lowerBound = Math.ceil(items.reduce((a, b) => a + b, 0));

	const selectFn = buildSelectFn(code);
	for (const item of items) {
		const binIdx = selectFn(item, bins);
		if (binIdx < 0 || binIdx >= bins.length) {
			// Open new bin
			bins.push(1.0 - item);
		} else if (bins[binIdx] >= item) {
			bins[binIdx] -= item;
		} else {
			bins.push(1.0 - item); // can't fit, open new
		}
	}

	return lowerBound / bins.length; // 1.0 = optimal
}

function buildSelectFn(code: string): (itemSize: number, bins: number[]) => number {
	// Detect common bin packing strategies from code patterns
	const lower = code.toLowerCase();

	if (lower.includes("best fit") || (lower.includes("min") && lower.includes("remain"))) {
		// Best Fit: choose bin with least remaining capacity that still fits
		return (itemSize, bins) => {
			let best = -1;
			let bestRemaining = Infinity;
			for (let i = 0; i < bins.length; i++) {
				if (bins[i] >= itemSize && bins[i] < bestRemaining) {
					best = i;
					bestRemaining = bins[i];
				}
			}
			return best;
		};
	}

	if (lower.includes("worst fit") || (lower.includes("max") && lower.includes("remain"))) {
		// Worst Fit: choose bin with most remaining capacity
		return (itemSize, bins) => {
			let best = -1;
			let bestRemaining = -Infinity;
			for (let i = 0; i < bins.length; i++) {
				if (bins[i] >= itemSize && bins[i] > bestRemaining) {
					best = i;
					bestRemaining = bins[i];
				}
			}
			return best;
		};
	}

	if (lower.includes("almost full") || lower.includes("threshold")) {
		// Almost Full Fit: prefer bins that would be ≥ 0.8 full after placement
		return (itemSize, bins) => {
			for (let i = 0; i < bins.length; i++) {
				if (bins[i] >= itemSize && bins[i] - itemSize <= 0.2) return i;
			}
			// Fallback: best fit
			let best = -1;
			let bestRemaining = Infinity;
			for (let i = 0; i < bins.length; i++) {
				if (bins[i] >= itemSize && bins[i] < bestRemaining) {
					best = i;
					bestRemaining = bins[i];
				}
			}
			return best;
		};
	}

	// Default: First Fit
	return (itemSize, bins) => {
		for (let i = 0; i < bins.length; i++) {
			if (bins[i] >= itemSize) return i;
		}
		return -1;
	};
}

// ── Extension registration ────────────────────────────────────────────────────

export default function register(api: ExtensionAPI): void {
	const engine = new EohEngine();
	let evolutionTask: Promise<void> | null = null;
	const log: string[] = [];

	function logLine(msg: string): void {
		const ts = new Date().toISOString().slice(11, 19);
		log.push(`[${ts}] ${msg}`);
		if (log.length > 200) log.shift();
	}

	// Wire progress events to log
	engine.setProgressHandler((event) => {
		switch (event.type) {
			case "generation_start":
				logLine(`Generation ${event.generation} started`);
				break;
			case "generation_end": {
				const s = event.stats;
				logLine(`Generation ${event.generation} done — best: ${s.best.toFixed(4)} mean: ${s.mean.toFixed(4)} size: ${s.size}`);
				break;
			}
			case "heuristic_evaluated":
				logLine(`New heuristic (${event.operator}) fitness=${event.heuristic.fitness.toFixed(4)}`);
				break;
			case "heuristic_failed":
				logLine(`Failed (${event.operator}): ${event.reason}`);
				break;
			case "init_complete":
				logLine(`Init complete — ${event.population.length} heuristics`);
				break;
			case "stopped":
				logLine("Evolution stopped");
				break;
		}
	});

	// Set default demo problem
	engine.setProblem(BIN_PACKING_PROBLEM);

	// ── LLM-callable tool ─────────────────────────────────────────────────────

	api.registerTool({
		name: "evolve_heuristics",
		label: "EoH: Evolve Heuristics",
		description: `Evolve algorithmic heuristics using the Evolution of Heuristics (EoH) framework. Actions: start, stop, status, best, run_generation.`,
		parameters: {
			type: "object",
			properties: {
				action: {
					type: "string",
					description: `One of: "start" (begin evolution), "stop" (halt), "status" (get stats), "best" (get best heuristic), "run_generation" (run single generation)`,
					required: true,
				},
				baseUrl: {
					type: "string",
					description: "LLM API base URL (required for start/run_generation)",
				},
				model: {
					type: "string",
					description: "LLM model name (required for start/run_generation)",
				},
				generations: {
					type: "number",
					description: "Max generations to run (for 'start', default 5)",
				},
			},
		},
		execute: async (_id, params, _ctx) => {
			const action = String(params.action ?? "status");
			const baseUrl = String(params.baseUrl ?? process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com/v1");
			const model = String(params.model ?? process.env.EOH_MODEL ?? "claude-haiku-4-5-20251001");

			switch (action) {
				case "start": {
					if (engine.getState().running) {
						return { content: "Evolution already running", isError: true };
					}
					const gens = Number(params.generations ?? 5);
					engine.setMaxGenerations(gens);
					evolutionTask = engine.run(baseUrl, model).catch((e) => {
						logLine(`Error: ${e instanceof Error ? e.message : String(e)}`);
					});
					return { content: `Evolution started (${gens} generations, model=${model})` };
				}

				case "stop": {
					engine.stop();
					return { content: "Stop signal sent" };
				}

				case "status": {
					const state = engine.getState();
					const stats = populationStats(state.population);
					return {
						content: JSON.stringify({
							running: state.running,
							generation: state.generation,
							totalLLMCalls: state.totalLLMCalls,
							populationSize: stats.size,
							bestFitness: stats.best,
							meanFitness: stats.mean,
							recentLog: log.slice(-10),
						}, null, 2),
					};
				}

				case "best": {
					const best = engine.getBestHeuristic();
					if (!best) return { content: "No heuristics in population yet" };
					return {
						content: `Best heuristic (fitness=${best.fitness.toFixed(4)}, gen=${best.generation}):

Thought:
${best.thought}

Code:
\`\`\`python
${best.code}
\`\`\``,
					};
				}

				case "run_generation": {
					if (engine.getState().running) {
						return { content: "Evolution already running — stop it first", isError: true };
					}
					if (engine.getState().population.length === 0) {
						await engine.initialize(baseUrl, model);
					}
					const candidates = await engine.runGeneration();
					const stats = populationStats(engine.getState().population);
					return {
						content: `Generation ${engine.getState().generation} complete — ${candidates.length} new candidates evaluated. Best fitness: ${stats.best.toFixed(4)}`,
					};
				}

				default:
					return { content: `Unknown action: ${action}`, isError: true };
			}
		},
	});

	// ── Slash commands ────────────────────────────────────────────────────────

	api.registerCommand({
		name: "eoh-start",
		description: "Start EoH evolution",
		usage: "/eoh-start [generations]",
		acceptsArgs: true,
		handler: async (args, ctx) => {
			const gens = parseInt(args.trim()) || 5;
			const state = engine.getState();
			if (state.running) return "Evolution already running";

			const baseUrl = process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com/v1";
			const model = process.env.EOH_MODEL ?? "claude-haiku-4-5-20251001";
			engine.setMaxGenerations(gens);
			engine.run(baseUrl, model).catch((e) => logLine(`Error: ${e instanceof Error ? e.message : String(e)}`));
			return `EoH evolution started: ${gens} generations, model=${model}`;
		},
	});

	api.registerCommand({
		name: "eoh-stop",
		description: "Stop EoH evolution",
		usage: "/eoh-stop",
		acceptsArgs: false,
		handler: () => {
			engine.stop();
			return "Stop signal sent";
		},
	});

	api.registerCommand({
		name: "eoh-status",
		description: "Show EoH evolution status",
		usage: "/eoh-status",
		acceptsArgs: false,
		handler: () => {
			const state = engine.getState();
			const stats = populationStats(state.population);
			const lines = [
				`Running: ${state.running}`,
				`Generation: ${state.generation}`,
				`LLM calls: ${state.totalLLMCalls}`,
				`Population: ${stats.size}`,
				`Best fitness: ${stats.best.toFixed(4)}`,
				`Mean fitness: ${stats.mean.toFixed(4)}`,
				"",
				"Recent log:",
				...log.slice(-8),
			];
			return lines.join("\n");
		},
	});

	api.registerCommand({
		name: "eoh-best",
		description: "Show best heuristic found so far",
		usage: "/eoh-best",
		acceptsArgs: false,
		handler: () => {
			const best = engine.getBestHeuristic();
			if (!best) return "No heuristics yet — run /eoh-start first";
			return `Best (fitness=${best.fitness.toFixed(4)}, gen=${best.generation}, by=${best.createdBy}):\n\n${best.thought}\n\n\`\`\`python\n${best.code}\n\`\`\``;
		},
	});

	api.registerCommand({
		name: "eoh-set-problem",
		description: "Set EoH problem definition from JSON",
		usage: "/eoh-set-problem <json>",
		acceptsArgs: true,
		handler: async (args, ctx) => {
			try {
				const def = JSON.parse(args.trim()) as {
					name: string;
					description: string;
					functionSignature: string;
				};
				if (!def.name || !def.description || !def.functionSignature) {
					return "Required fields: name, description, functionSignature";
				}
				// Use a simple passthrough evaluator — caller provides no instances
				const problem: EohProblem = {
					...def,
					instances: [{}],
					evaluateInstance: async () => 0.5,
				};
				engine.setProblem(problem);
				return `Problem set: ${def.name}`;
			} catch (e) {
				return `Parse error: ${e instanceof Error ? e.message : String(e)}`;
			}
		},
	});
}
