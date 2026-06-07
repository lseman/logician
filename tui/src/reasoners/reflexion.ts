// ── Reflexion ──────────────────────────────────────────────────────────────────
// Adapted from Python src/reasoners/reflexion.py.
//
// Iterate: attempt → critique → rewrite based on reflections.

import { BaseReasoner, type ReasoningTrace } from "./base.js";

interface ReflexionConfig {
	maxTrials?: number;
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class ReflexionReasoner extends BaseReasoner {
	config: ReflexionConfig;

	constructor(
		llm: import("../agent-core/backend.js").LLMBackend,
		config: ReflexionConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const maxTrials = this.config.maxTrials ?? 3;
		let attempt = initialSolution ?? "";
		const reflections: string[] = [];

		for (let _ = 0; _ < maxTrials; _++) {
			if (!attempt) {
				attempt = await this._chat([
					{
						role: "user",
						content: `${query}\nThink step by step. End with 'Final answer: ...'.`,
					},
				]);
			}

			const critique = await this._chat([
				{
					role: "user",
					content: `[Problem]\n${query}\n\n[Attempt]\n${attempt}\n\nCritique weaknesses or errors.`,
				},
			]);
			reflections.push(critique);

			attempt = await this._chat([
				{
					role: "user",
					content: [
						"[Problem]",
						query,
						"",
						"Reflections:",
						...reflections,
						"",
						"Rewrite solution based on reflections.",
						"End with 'Final answer: ...'.",
					].join("\n"),
				},
			]);
		}

		const [reasoning, answer] = this._split(attempt);
		return {
			reasoning,
			answer,
			metadata: { method: "reflexion", reflections },
		};
	}
}
