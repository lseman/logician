// ── In-Context CoT ─────────────────────────────────────────────────────────────
// Adapted from Python src/reasoners/in_context_cot.py.
//
// Requires user-provided exemplars in config.

import { BaseReasoner, type ReasoningTrace } from "./base.js";

export class InContextCoTReasoner extends BaseReasoner {
	config: Record<string, unknown>;

	constructor(
		llm: import("../core/backend.ts").LLMBackend,
		config: Record<string, unknown> = {},
	) {
		super(llm, config as import("./base.js").ReasonerConfig);
		this.config = config;
	}

	async solve(
		query: string,
		_initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const exemplars = this.config.exemplars as string | undefined;
		if (!exemplars) {
			throw new Error("InContextCoT requires config['exemplars'].");
		}

		const prompt = `${exemplars.trim()}\n\nQ: ${query}\nA: (step by step)\nEnd with 'Final answer: ...'.`;

		const out = await this._chat([{ role: "user", content: prompt }], {
			temperature: 0.3,
		});
		const [reasoning, answer] = this._split(out);
		return { reasoning, answer, metadata: { method: "in_context_cot" } };
	}
}
