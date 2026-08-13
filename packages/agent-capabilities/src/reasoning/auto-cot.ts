// ── AutoCoT ────────────────────────────────────────────────────────────────────
// Adapted from Python src/reasoners/auto_cot.py.
//
// Generates domain-relevant exemplars, then uses them as in-context examples.

import { BaseReasoner, type ReasoningTrace } from "./base.ts";

interface AutoCoTConfig {
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

const EXEMPLAR_PROMPT = `You are an AI agent specializing in time series forecasting, data analysis, and software development.

Generate exactly 3 diverse example questions and answers that demonstrate clear, structured reasoning. Cover different areas: one about time series / forecasting concepts, one about code / software architecture, one about data processing or statistics.

Format each example as:

Q: <concise technical question>
REASONING: <2-4 sentences showing step-by-step analysis>
Final answer: <direct, well-structured answer>

---

Return ONLY the 3 examples, no preamble.`;

export class AutoCoTReasoner extends BaseReasoner {
	config: AutoCoTConfig;

	constructor(
		llm: import("@logician/agent-core/agent/backend.ts").LLMBackend,
		config: AutoCoTConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const maxTokens = this.config.maxTokens ?? 1024;

		const exemplars = await this._chat(
			[{ role: "user", content: EXEMPLAR_PROMPT }],
			{ temperature: 0.6, maxTokens },
		);
		const solvePrompt = `${exemplars.trim()}\n\n---\n\nNow answer the following using the same structured reasoning:\n\nQ: ${query}\nREASONING: <explain briefly>\nFinal answer: <your final answer>`;

		const out = await this._chat([{ role: "user", content: solvePrompt }], {
			temperature: this.config.temperature ?? 0.3,
		});

		if (!out) {
			const fallback = (initialSolution ?? "").trim();
			return {
				reasoning: "",
				answer: fallback,
				metadata: {
					method: "auto_cot",
					degraded: true,
					reason: "empty_generation",
					exemplar_preview: exemplars.slice(0, 200),
				},
			};
		}

		let full = out;
		if (!out.trimStart().toLowerCase().startsWith("reasoning:")) {
			full = `REASONING: ${out}`;
		}
		const [reasoning, answer] = this._split(full);

		const badTokens = new Set([
			"reasoning:",
			"reasoning",
			"final answer:",
			"final answer",
		]);
		let finalAnswer = (answer ?? "").trim();
		if (!finalAnswer || badTokens.has(finalAnswer.toLowerCase())) {
			const extracted = (BaseReasoner._extractAnswer(out) ?? "").trim();
			if (extracted && !badTokens.has(extracted.toLowerCase())) {
				finalAnswer = extracted;
			} else {
				finalAnswer = (initialSolution ?? out).trim();
			}
		}

		return {
			reasoning,
			answer: finalAnswer,
			metadata: {
				method: "auto_cot",
				exemplar_preview: exemplars.slice(0, 200),
			},
		};
	}
}
