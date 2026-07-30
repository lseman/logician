// ── Self-Consistency ───────────────────────────────────────────────────────────
// Adapted from Python src/reasoners/self_consistency.py.
//
// Generate N independent samples, pick the most common answer (voting).

import { BaseReasoner, type ReasoningTrace } from "./base.ts";

interface SelfConsistencyConfig {
	nRollouts?: number;
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class SelfConsistencyReasoner extends BaseReasoner {
	config: SelfConsistencyConfig;

	constructor(
		llm: import("@logician/agent-core/agent/backend.ts").LLMBackend,
		config: SelfConsistencyConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		_initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const n = this.config.nRollouts ?? 32;
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
		const answers = samples.map((s) => BaseReasoner._extractAnswer(s));
		const counts = new Map<string, number>();
		for (const a of answers) counts.set(a, (counts.get(a) ?? 0) + 1);
		const [bestAnswer, count] = [...counts.entries()].sort(
			(a, b) => b[1] - a[1],
		)[0];
		const best =
			samples.find((s) => BaseReasoner._extractAnswer(s) === bestAnswer) ??
			samples[0];
		const [reasoning, answer] = this._split(best);
		return {
			reasoning,
			answer,
			metadata: { method: "self_consistency", votes: count, n },
		};
	}
}
