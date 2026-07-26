// ── Shared response patterns ────────────────────────────────────────────────
// Centralized regex patterns for detecting non-committal, complete, and
// circling model responses. All consumers import from here — never define
// patterns inline.

// ── Non-committal patterns ──────────────────────────────────────────────────
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

// ── Completion patterns ─────────────────────────────────────────────────────
// Patterns that indicate the model has declared task completion.

export const COMPLETE_PATTERNS: ReadonlyArray<RegExp> = [
	/\b(task\s+complete|all\s+done|finished|completed\s+successfully|nothing\s+(else|more)\s+to\s+do|no\s+(further|more)\s+(steps?|action|work)|that('s|\s+is)\s+(all|done|complete))\b/i,
	/^done\s*$/i,
];

// ── Meta-reasoning patterns ────────────────────────────────────────────────
// Patterns indicating the model is reasoning about its own reasoning rather
// than taking action. These capture the "meta-loop" signature.

export const META_REASONING_PATTERNS: ReadonlyArray<RegExp> = [
	// "Let me think about how to approach..." — planning about planning.
	/\blet\s+me\s+(think\s+(about|on|over)|consider)\s+(how\s+)?(?:to\s+)?(?:approach|handle|solve| tackle)\b/i,
	// "I need to think about..." — pure meta-reasoning, no action.
	/\bi\s+(need\s+to|should\s+to|must\s+to)\s+(think\s+about|consider|reflect\s+on|ponder)\b/i,
	// "Before I do X, I need to think about Y..." — procrastination pattern.
	/\b(?:before|while|when)\s+(?:i\s+)?(?:do|proceed|start|begin|move)\s+.*?\b,\s+(?:i\s+)?(?:need\s+to|should|have\s+to)\s+(think|consider|reflect)\b/i,
	// "I'm not sure about X, let me think..." — hesitation loop.
	/\b(?:i'm\s+not\s+sure|not\s+sure\s+about|i\s+don't\s+know\s+how)\b.*?(?:let\s+me\s+think|let\s+me\s+consider)\b/i,
	// "I should first understand... then I can..." — endless analysis.
	/\b(?:i\s+should\s+first|first\s+i\s+need\s+to)\s+(understand|comprehend|grasp|analyze)\b.*?\b(?:then|after\s+that|once\s+i.*)\b/i,
	// "Upon further reflection..." — meta-reasoning escalation.
	/\b(?:upon\s+further\s+reflection|after\s+considering|on\s+second\s+thought)\b/i,
	// "This requires me to think..." — meta-reasoning declaration.
	/\b(?:this\s+(?:requires|demands|needs)\s+(?:me\s+)?to\s+think)\b/i,
	// "I need to step back and think..." — retreat into thinking.
	/\b(?:step\s+back|pause)\s+(?:to\s+)?(?:think|consider|reflect)\b/i,
	// "I realize I need to think..." — meta-realization loop.
	/\b(?:i\s+realize|i\s+see\s+that|i\s+understand\s+now)\s+(?:that\s+)?(?:i\s+need\s+to|i\s+should|i\s+must)\s+(think|consider|rethink)\b/i,
	// "Thinking through this... okay..." — self-talk spiral.
	/\b(?:thinking\s+through\s+this|let\s+me\s+walk\s+through\s+this|working\s+through\s+this)\b.*?\b(?:okay|alright|right)\b/i,
];

// ── Circling patterns ───────────────────────────────────────────────────────
// Patterns that suggest the model is circling — retrying the same approach
// without success. Broader than stop declarations to escalate nudge tone.
// Stricter to avoid false positives on legitimate multi-step work.

export const CIRCLING_PATTERNS: ReadonlyArray<RegExp> = [
	// Future retry intent without evidence of a changed strategy.
	/\b(?:i\s+will|i'll)\s+(?:try|attempt)(?:\s+to)?\b/i,
	/\b(?:let\s+me|i(?:'m|\s+am)\s+going\s+to)\s+(?:try|attempt)\b/i,
	// A failed attempt followed by an explicit failure clause.
	/\bi\s+(?:tried|attempted)\b.*\b(?:but|however)\b.*\b(?:did(?:n't| not)\s+work|failed|unable)\b/i,
	/\bi\s+(?:tried|attempted)\b.*\b(?:again|next|yet)\b/i,
	// "I'll try again" (no X) — bare retry intent without specifying a new approach.
	/\bi(?:\s+will|'ll|ll)\s+(?:try again|attempt again)\b/i,
	// "I tried X again" — explicit past retry with "again".
	/\bi\s+(?:tried|attempted)\s+.*\b(again|yet)\b/i,
	// "Let me try again" (bare) — retry without new approach.
	/\blet\s+me\s+(?:try again|attempt again)\b/i,
	// "I've tried X again" — past retry with "again".
	/\b(i'|ve|I've)\s+(?:tried|attempted)\s+.*\b(again|yet)\b/i,
	// "cannot/can't X but try/attempt" — failed then retrying.
	/\b(cannot|can't|unable)\s+.*\b(but\s+|however\s+|instead\s+)\b.*\b(try|attempt|do|make|go)\b/i,
];

// ── Query helpers ───────────────────────────────────────────────────────────

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

/**
 * Check if a text response indicates circling — retrying the same approach
 * without success. Inspects the full text since circling patterns appear
 * anywhere in the response.
 */
export function detectsCircling(assistantText: string): boolean {
	if (!assistantText || assistantText.trim().length < 10) return false;
	const lower = assistantText.toLowerCase();
	return CIRCLING_PATTERNS.some((re) => re.test(lower));
}
