// ── SSR: Socratic Self-Refinement ──────────────────────────────────────────────
// Adapted from Python src/reasoners/ssr.py.
//
// Flow: initial solution → decompose into steps → verify each step via sampling
// → refine weakest step → repeat.

import { BaseReasoner, type ReasoningTrace } from "./base.js";

export interface SocraticStep {
	index: number;
	question: string;
	answer: string;
	confidence: number | null;
	samples: string[] | null;
}

interface SSRConfig {
	maxIterations?: number;
	mSamples?: number;
	mode?: "plan" | "direct";
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class SSRReasoner extends BaseReasoner {
	config: SSRConfig;

	constructor(
		llm: import("@logician/agent-core/agent/backend.ts").LLMBackend,
		config: SSRConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const maxIter = this.config.maxIterations ?? 3;
		const mSamples = this.config.mSamples ?? 8;
		const mode = this.config.mode ?? "plan";

		let [reasoning, answer] = await this._initial(query, initialSolution);

		if (mode === "plan") {
			[reasoning, answer] = await this._refinePlan(query, reasoning, answer);
		}

		const history: {
			iteration: number;
			steps: { index: number; conf: number | null }[];
		}[] = [];

		for (let it = 0; it < maxIter; it++) {
			const steps = await this._decompose(query, reasoning, answer);
			const verified = await this._verify(query, steps, mSamples);
			const updated = await this._refine(query, reasoning, answer, verified);
			if (updated === null) break;
			[reasoning, answer] = updated;
			history.push({
				iteration: it + 1,
				steps: verified.map((s) => ({ index: s.index, conf: s.confidence })),
			});
		}

		return { reasoning, answer, metadata: { method: "ssr", history } };
	}

	private async _initial(
		query: string,
		init: string | undefined | null,
	): Promise<[string, string]> {
		if (init) return this._split(init);
		const resp = await this._chat([
			{
				role: "system",
				content: "Think step by step like a careful mathematician.",
			},
			{ role: "user", content: `${query}\n\nEnd with 'Final answer: ...'.` },
		]);
		return this._split(resp);
	}

	private async _refinePlan(
		query: string,
		reasoning: string,
		answer: string,
	): Promise<[string, string]> {
		const prompt = [
			"[Problem]",
			query,
			"[Draft]",
			reasoning,
			"Final answer:",
			answer,
			"",
			"1. Summarize a plan.",
			"2. Improve the plan.",
			"3. Rewrite cleanly.",
			"End with 'Final answer: ...'.",
		].join("\n");
		const resp = await this._chat([{ role: "user", content: prompt }]);
		return this._split(resp);
	}

	private async _decompose(
		query: string,
		reasoning: string,
		answer: string,
	): Promise<SocraticStep[]> {
		const prompt = [
			"[Problem]",
			query,
			"[Solution]",
			reasoning,
			"Final answer:",
			answer,
			"",
			"Break this into <= 8 steps. Return JSON array only.",
		].join("\n");
		const raw = await this._chat([{ role: "user", content: prompt }], {
			temperature: 0.0,
			maxTokens: 512,
		});
		try {
			const cleaned = raw
				.replace(/^`+\s*/, "")
				.replace(/\s*`+$/, "")
				.trim();
			const data = JSON.parse(cleaned) as Array<Record<string, unknown>>;
			return data.map(
				(step, i) =>
					({
						index: Number(step.index ?? i + 1),
						question: String(step.question ?? ""),
						answer: String(step.answer ?? ""),
						confidence: null,
						samples: null,
					}) as SocraticStep,
			);
		} catch (_e: unknown) {
			return [
				{
					index: 1,
					question: "Whole reasoning",
					answer: `${reasoning}\nFinal answer: ${answer}`,
					confidence: null,
					samples: null,
				},
			] as SocraticStep[];
		}
	}

	private async _verify(
		query: string,
		steps: SocraticStep[],
		m: number,
	): Promise<SocraticStep[]> {
		const results: SocraticStep[] = [];
		for (let idx = 0; idx < steps.length; idx++) {
			const step = steps[idx];
			const prev = steps
				.slice(0, idx)
				.map((s) => `Step ${s.index}: ${s.answer}`)
				.join("\n");
			const samples: string[] = [];
			for (let s = 0; s < m; s++) {
				const sample = await this._chat(
					[
						{
							role: "user",
							content: `Problem: ${query}\nPrevious steps:\n${prev || "(none)"}\nRe-solve the sub-question:\n${step.question}\nShort answer only.`,
						},
					],
					{ temperature: 0.8, maxTokens: 64 },
				);
				samples.push(sample);
			}
			const norm = SSRReasoner._norm(step.answer);
			const matches = samples.filter(
				(s) => SSRReasoner._norm(s) === norm,
			).length;
			step.samples = samples;
			step.confidence = matches / m;
			results.push(step);
		}
		return results;
	}

	private async _refine(
		query: string,
		reasoning: string,
		answer: string,
		steps: SocraticStep[],
	): Promise<[string, string] | null> {
		const bad = steps.reduce((worst, s) =>
			(s.confidence ?? 1) < (worst.confidence ?? 1) ? s : worst,
		);
		if (bad.confidence === null || bad.confidence > 0.8) return null;

		const counts = new Map<string, number>();
		for (const sample of bad.samples ?? []) {
			const n = SSRReasoner._norm(sample);
			counts.set(n, (counts.get(n) ?? 0) + 1);
		}
		const [bestNorm] = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
		if (bestNorm === SSRReasoner._norm(bad.answer)) return null;

		const bestRaw =
			bad.samples?.find((s) => SSRReasoner._norm(s) === bestNorm) ?? bad.answer;

		const prompt = [
			"[Problem]",
			query,
			"[Old solution]",
			reasoning,
			"Final answer:",
			answer,
			"",
			"Bad step:",
			`Q: ${bad.question}`,
			`Old A: ${bad.answer}`,
			`Better A: ${bestRaw}`,
			"",
			"Rewrite full solution consistently.",
			"End with 'Final answer: ...'.",
		].join("\n");
		const out = await this._chat([{ role: "user", content: prompt }], {
			temperature: 0.7,
		});
		return this._split(out);
	}

	private static _norm(s: string): string {
		return s
			.trim()
			.toLowerCase()
			.replace(/[ \t\n\r]+/g, " ")
			.replace(/[.,;:!?]$/, "");
	}
}
