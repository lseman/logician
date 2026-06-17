// ── Best-of-N ──────────────────────────────────────────────────────────────────
// Adapted from Python src/reasoners/best_of_n.py.
//
// Generate N samples, score each, pick the best.

import { BaseReasoner, type ReasoningTrace } from "./base.js";

interface BestOfNConfig {
	n?: number;
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class BestOfNReasoner extends BaseReasoner {
	config: BestOfNConfig;

	constructor(
		llm: import("../agent-core/core/backend.js").LLMBackend,
		config: BestOfNConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		_initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const n = this.config.n ?? 8;
		const temp = this.config.temperature ?? 0.8;

		const samples = await Promise.all(
			Array.from({ length: n }, () =>
				this._chat(
					[
						{
							role: "user",
							content: `${query}\nThink step by step. End with 'Final answer: ...'.`,
						},
					],
					{ temperature: temp },
				),
			),
		);
		const scored = await Promise.all(
			samples.map((sample) =>
				this._score(query, sample).then((score) => ({ sample, score })),
			),
		);
		const best = scored.reduce((a, b) => (b.score > a.score ? b : a));
		const [reasoning, answer] = this._split(best.sample);
		return {
			reasoning,
			answer,
			metadata: { method: "best_of_n", score: best.score },
		};
	}

	private _score(query: string, reasoning: string): Promise<number> {
		const prompt = `[Problem]\n${query}\n\n[Candidate]\n${reasoning}\n\nRate quality (0-1). Output only the number.`;
		return this._chat([{ role: "user", content: prompt }], {
			temperature: 0.0,
			maxTokens: 16,
		}).then((raw) => {
			const match = raw.trim().match(/[0-1](?:\.\d+)?/);
			return match ? parseFloat(match[0]) : 0.0;
		});
	}
}
