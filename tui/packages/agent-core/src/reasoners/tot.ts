// ── ToT: Tree of Thoughts ──────────────────────────────────────────────────────
// Adapted from Python src/reasoners/tot.py.
//
// Beam search over reasoning paths, scoring each candidate.

import { BaseReasoner, type ReasoningTrace } from "./base.js";

interface ToTConfig {
	beamWidth?: number;
	maxDepth?: number;
	branchFactor?: number;
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class ToTReasoner extends BaseReasoner {
	config: ToTConfig;

	constructor(
		llm: import("../core/backend.ts").LLMBackend,
		config: ToTConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const beam = this.config.beamWidth ?? 6;
		const maxDepth = this.config.maxDepth ?? 10;
		const branch = this.config.branchFactor ?? 3;

		type State = [reasoning: string, score: number];
		let frontier: State[] = initialSolution
			? [[initialSolution, await this._score(query, initialSolution)]]
			: [["", 0.0]];

		let bestReasoning = "";
		let bestAnswer = "";
		let bestScore = -Infinity;
		let evaluated = 0;

		for (let depth = 0; depth < maxDepth; depth++) {
			const next: State[] = [];
			for (const [reasoning, scorePrev] of frontier) {
				if (/final answer/i.test(reasoning)) {
					if (scorePrev > bestScore) {
						bestScore = scorePrev;
						bestReasoning = reasoning;
						bestAnswer = BaseReasoner._extractAnswer(reasoning);
					}
					continue;
				}
				const prompt = `${query}\n\nReasoning so far:\n${reasoning || "(empty)"}\n\nContinue reasoning. If done, end with 'Final answer: ...'.`;
				for (let b = 0; b < branch; b++) {
					const cont = await this._chat([{ role: "user", content: prompt }], {
						temperature: 0.9,
						maxTokens: 512,
					});
					const full = `${reasoning}\n${cont}`.trim();
					const score = await this._score(query, full);
					evaluated++;
					if (/final answer/i.test(full) && score > bestScore) {
						bestScore = score;
						bestReasoning = full;
						bestAnswer = BaseReasoner._extractAnswer(full);
					}
					next.push([full, score]);
				}
			}
			if (next.length === 0) break;
			frontier = next.sort((a, b) => b[1] - a[1]).slice(0, beam);
		}

		if (bestScore === -Infinity && frontier.length > 0) {
			const [best] = frontier.sort((a, b) => b[1] - a[1]);
			bestReasoning = best[0];
			bestAnswer = BaseReasoner._extractAnswer(best[0]);
		}

		return {
			reasoning: bestReasoning,
			answer: bestAnswer,
			metadata: { method: "tot", states: evaluated },
		};
	}

	private async _score(query: string, reasoning: string): Promise<number> {
		const prompt = `[Problem]\n${query}\n\n[Partial solution]\n${reasoning}\n\nRate promise (0-1). Output only a number.`;
		const raw = await this._chat([{ role: "user", content: prompt }], {
			temperature: 0.0,
			maxTokens: 16,
		});
		const match = raw.trim().match(/[0-1](?:\.\d+)?/);
		if (match) {
			const v = parseFloat(match[0]);
			if (!Number.isNaN(v)) return v;
		}
		const length = Math.min(reasoning.length / 1000, 1.0);
		const bonus = /final answer/i.test(reasoning) ? 0.2 : 0.0;
		return length + bonus;
	}
}
