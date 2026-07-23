// ── EoH heuristic evaluator ───────────────────────────────────────────────────
// Parses LLM output, validates code syntax, evaluates on instances.

import type { EohGenerateResult, EohProblem } from "./types.ts";

const THOUGHT_RE = /<thought>([\s\S]*?)<\/thought>/;
const CODE_FENCE_RE = /```(?:python)?\n([\s\S]*?)```/;
const CODE_RE = /<code>([\s\S]*?)<\/code>/;

/** Parse <thought>...</thought> and <code>```python...```</code> from LLM output. */
export function parseHeuristicOutput(raw: string): EohGenerateResult | null {
	const thoughtMatch = THOUGHT_RE.exec(raw);
	if (!thoughtMatch) return null;
	const thought = thoughtMatch[1].trim();
	if (!thought) return null;

	// Try <code> block first, then bare code fence
	const codeBlockRaw = CODE_RE.exec(raw)?.[1] ?? raw;
	const codeMatch = CODE_FENCE_RE.exec(codeBlockRaw);
	if (!codeMatch) return null;
	const code = codeMatch[1].trim();
	if (!code) return null;

	return { thought, code };
}

/** Basic Python syntax check via Function constructor heuristic (JS-side). */
export function validateCode(code: string, expectedFnName: string): string | null {
	// Must define a function
	if (!code.includes("def ")) return "No function definition found";
	// Must contain the expected function name
	if (!code.includes(`def ${expectedFnName}`)) {
		// Try to extract any def name for a softer check
		const defMatch = /def\s+(\w+)\s*\(/.exec(code);
		if (!defMatch) return "Cannot find function definition";
		// Accept any single function definition
	}
	// Reject obvious syntax errors: unbalanced parens
	let depth = 0;
	for (const ch of code) {
		if (ch === "(") depth++;
		if (ch === ")") depth--;
		if (depth < 0) return "Unbalanced parentheses";
	}
	if (depth !== 0) return "Unbalanced parentheses";
	return null; // valid
}

/** Extract the primary function name from code. */
export function extractFunctionName(code: string): string | null {
	const match = /def\s+(\w+)\s*\(/.exec(code);
	return match ? match[1] : null;
}

/**
 * Evaluate a heuristic on all problem instances.
 * Returns mean fitness, or -Infinity if any instance throws.
 */
export async function evaluateHeuristic(
	code: string,
	problem: EohProblem,
	timeoutMs: number,
): Promise<number> {
	const scores: number[] = [];
	for (const instance of problem.instances) {
		try {
			const score = await Promise.race([
				problem.evaluateInstance(code, instance),
				new Promise<number>((_, reject) =>
					setTimeout(() => reject(new Error("eval timeout")), timeoutMs),
				),
			]);
			scores.push(score);
		} catch (e: unknown) {
			return -Infinity;
		}
	}
	if (scores.length === 0) return -Infinity;
	return scores.reduce((a, b) => a + b, 0) / scores.length;
}
