// ── EoH prompt templates ──────────────────────────────────────────────────────
// Five operator prompts from EoH paper (arxiv 2401.02051).
// Each returns messages array for the LLM.

import type { Heuristic, EohProblem } from "./types.ts";

function heuristicBlock(h: Heuristic, index: number): string {
	return `Heuristic ${index + 1}:
Thought: ${h.thought}
Code:
\`\`\`python
${h.code}
\`\`\``;
}

function baseSystemPrompt(problem: EohProblem): string {
	return `You are an expert algorithm designer evolving heuristics for: ${problem.name}

Problem description:
${problem.description}

Required function signature:
${problem.functionSignature}

You must output exactly two sections:
<thought>
A concise description (2-4 sentences) of the key algorithmic idea.
</thought>
<code>
\`\`\`python
${problem.functionSignature}
    # implementation
    pass
\`\`\`
</code>

Output only these two XML sections. No explanation outside them.`;
}

/** E1: Generate a heuristic maximally different from all parents. */
export function promptE1Diversity(problem: EohProblem, parents: Heuristic[]): Array<{ role: string; content: string }> {
	const parentBlocks = parents.map(heuristicBlock).join("\n\n");
	return [
		{ role: "system", content: baseSystemPrompt(problem) },
		{
			role: "user",
			content: `Here are ${parents.length} existing heuristics:

${parentBlocks}

Design a NEW heuristic with a completely DIFFERENT algorithmic strategy from all of the above. The thought and implementation must use a distinct approach — not a variation of what exists.`,
		},
	];
}

/** E2: Identify common core ideas, then build a new heuristic using those ideas but differently. */
export function promptE2Convergence(problem: EohProblem, parents: Heuristic[]): Array<{ role: string; content: string }> {
	const parentBlocks = parents.map(heuristicBlock).join("\n\n");
	return [
		{ role: "system", content: baseSystemPrompt(problem) },
		{
			role: "user",
			content: `Here are ${parents.length} existing heuristics:

${parentBlocks}

First, identify the common algorithmic principles shared across these heuristics. Then design a NEW heuristic that:
1. Builds on those shared principles
2. Combines or extends them in a novel way
3. Is clearly distinct from any individual parent`,
		},
	];
}

/** M1: Improve a single heuristic's performance. */
export function promptM1Improve(problem: EohProblem, parent: Heuristic): Array<{ role: string; content: string }> {
	return [
		{ role: "system", content: baseSystemPrompt(problem) },
		{
			role: "user",
			content: `Here is an existing heuristic:

${heuristicBlock(parent, 0)}

Analyze this heuristic's weaknesses and design an IMPROVED version that achieves better performance. You may restructure the algorithm significantly.`,
		},
	];
}

/** M2: Tune parameters of a single heuristic. */
export function promptM2Tune(problem: EohProblem, parent: Heuristic): Array<{ role: string; content: string }> {
	return [
		{ role: "system", content: baseSystemPrompt(problem) },
		{
			role: "user",
			content: `Here is an existing heuristic:

${heuristicBlock(parent, 0)}

Keep the same overall algorithmic structure but TUNE the numerical parameters, thresholds, weights, or constants to improve performance. Do not change the core logic.`,
		},
	];
}

/** M3: Simplify a heuristic by removing redundant components. */
export function promptM3Simplify(problem: EohProblem, parent: Heuristic): Array<{ role: string; content: string }> {
	return [
		{ role: "system", content: baseSystemPrompt(problem) },
		{
			role: "user",
			content: `Here is an existing heuristic:

${heuristicBlock(parent, 0)}

Analyze this heuristic for redundant, unnecessary, or overly complex components. Design a SIMPLIFIED version that:
1. Removes unnecessary complexity
2. Preserves or improves performance
3. Is easier to understand and execute`,
		},
	];
}

/** Init: Generate a fresh heuristic from scratch. */
export function promptInit(problem: EohProblem, existingThoughts: string[]): Array<{ role: string; content: string }> {
	const avoidSection = existingThoughts.length > 0
		? `\n\nExisting approaches to avoid duplicating:\n${existingThoughts.map((t, i) => `${i + 1}. ${t}`).join("\n")}`
		: "";
	return [
		{ role: "system", content: baseSystemPrompt(problem) },
		{
			role: "user",
			content: `Design a novel heuristic for this problem. Be creative and use a distinctive algorithmic approach.${avoidSection}`,
		},
	];
}
