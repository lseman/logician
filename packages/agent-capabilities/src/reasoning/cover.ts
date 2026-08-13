// ── CoVe: Chain of Verification ──────────────────────────────────────────────
// Adapted from "Chain of Verification: Reducing Hallucination in Context" (Yao et al., 2023).
//
// CoVe involves:
// 1. Initial response generation
// 2. Planning verification steps
// 3. Executing verification steps
// 4. Generating final verified response

import { BaseReasoner, type ReasoningTrace } from "./base.ts";

interface CoVeConfig {
	maxVerificationSteps?: number;
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class CoVeReasoner extends BaseReasoner {
	config: CoVeConfig;

	constructor(
		llm: import("@logician/agent-core/agent/backend.ts").LLMBackend,
		config: CoVeConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		_initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const maxSteps = this.config.maxVerificationSteps ?? 3;
		const temp = this.config.temperature ?? 0.3;

		// Step 1: Generate initial response
		const initialPrompt = `${query}\n\nThink step by step. End with 'Final answer: ...'.`;
		const initialResponse = await this._chat(
			[{ role: "user", content: initialPrompt }],
			{
				temperature: temp,
				maxTokens: this.config.maxTokens ?? 1024,
			},
		);

		this._split(initialResponse);

		// Step 2: Plan verification steps
		const planPrompt = [
			"[Problem]",
			query,
			"",
			"[Initial Response]",
			initialResponse,
			"",
			"Plan verification steps to check the accuracy and completeness of the initial response.",
			"Return a JSON array of verification steps, each with a 'step' description.",
			'Example: [{"step": "Verify factual claims about X"}, {"step": "Check logical consistency of Y"}]',
		].join("\n");

		const planResponse = await this._chat(
			[{ role: "user", content: planPrompt }],
			{
				temperature: 0.1,
				maxTokens: 512,
			},
		);

		// Parse verification steps
		let verificationSteps: string[] = [];
		try {
			const cleaned = planResponse
				.replace(/^`+\s*/, "")
				.replace(/\s*`+$/, "")
				.trim();
			const data = JSON.parse(cleaned);
			if (Array.isArray(data)) {
				verificationSteps = data.map(
					(item: any) => item.step || JSON.stringify(item),
				);
			}
		} catch (_e: unknown) {
			// Fallback to default verification steps if parsing fails
			verificationSteps = [
				"Verify factual accuracy of claims",
				"Check logical consistency",
				"Ensure completeness of the response",
			];
		}

		// Step 3: Execute verification steps
		let verifiedResponse = initialResponse;
		const verificationHistory: string[] = [];

		for (
			let step = 0;
			step < Math.min(maxSteps, verificationSteps.length);
			step++
		) {
			const verificationStep = verificationSteps[step];

			const verifyPrompt = [
				"[Problem]",
				query,
				"",
				"[Current Response]",
				verifiedResponse,
				"",
				"[Verification Step]",
				verificationStep,
				"",
				"Perform the verification step and identify any issues, inaccuracies, or improvements needed.",
				"Return your findings and an improved version of the response if needed.",
			].join("\n");

			const verificationResult = await this._chat(
				[{ role: "user", content: verifyPrompt }],
				{
					temperature: 0.1,
					maxTokens: 1024,
				},
			);

			verificationHistory.push(
				`Step ${step + 1}: ${verificationStep}\nFindings: ${verificationResult}`,
			);

			// Extract improved response from verification result
			const improvedResponsePrompt = [
				"[Problem]",
				query,
				"",
				"[Initial Response]",
				initialResponse,
				"",
				"[Verification History]",
				verificationHistory.join("\n\n"),
				"",
				"Generate a final verified response incorporating all verification findings.",
				"End with 'Final answer: ...'.",
			].join("\n");

			verifiedResponse = await this._chat(
				[{ role: "user", content: improvedResponsePrompt }],
				{
					temperature: temp,
					maxTokens: this.config.maxTokens ?? 1024,
				},
			);
		}

		const [finalReasoning, finalAnswer] = this._split(verifiedResponse);

		return {
			reasoning: finalReasoning,
			answer: finalAnswer,
			metadata: {
				method: "cover",
				verificationSteps: verificationSteps.length,
				verificationHistory,
			},
		};
	}
}
