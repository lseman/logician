// ── Shared response patterns ────────────────────────────────────────────────
// Centralized regex patterns for detecting non-committal model responses.
// Consumers: output-guard, conclusion-policy, thinking-loop-detector.

// Patterns that indicate the model is hedging or deferring action rather than
// taking a concrete step. Used to detect "thinking without acting" loops.
export const NON_COMMITTAL_PATTERNS: ReadonlyArray<RegExp> = [
	/\b(i\s+(need|should|have|might|could|will)\s+(to\s+)?(?:check|look|think|consider|analyze|investigate|examine|review|verify))\b/i,
	/\b(let\s+me\s+(think|see|check|try|consider))\b/i,
	/\b(i'm\s+(going\s+to|thinking\s+about|not\s+sure|still\s+considering))\b/i,
	/\b(i'll\s+(try|check|look|see|think))\b/i,
	/\b(need\s+to\s+(check|think|verify|confirm))\b/i,
	/\b(however|but|although)\s+(i\s+(need|should|have|might))\b/i,
	/\b(this\s+(requires|needs|demands|warrants)\s+(further|more|additional))\b/i,
	/\b(i\s+(don't|do\s+not)\s+(know|think\s+|certain))\b/i,
	/\blet(?:'s|\s+me)\s+(?:step\s+back|circle\s+back|reconsider)\b/i,
	/\b(at\s+this\s+point|so\s+far)\s+(i\s+(have|can|see)|we\s+(need|should))\b/i,
];

// Patterns that indicate the model has declared completion.
export const COMPLETE_PATTERNS: ReadonlyArray<RegExp> = [
	/\b(task\s+complete|all\s+done|finished|completed\s+successfully|nothing\s+(else|more)\s+to\s+do|no\s+(further|more)\s+(steps?|action|work)|that('s|\s+is)\s+(all|done|complete))\b/i,
	/^done\s*$/i,
];

/**
 * Check if a text response is non-committal (hedging without action).
 * Returns true when the text contains hedging patterns and is long enough to
 * plausibly contain a decision.
 */
export function looksNonCommittal(text: string): boolean {
	return (
		text.trim().length >= 10 &&
		NON_COMMITTAL_PATTERNS.some((pattern) => pattern.test(text))
	);
}

/**
 * Check if a text response declares task completion.
 */
export function looksComplete(text: string): boolean {
	return Boolean(text) && COMPLETE_PATTERNS.some((pattern) => pattern.test(text));
}
