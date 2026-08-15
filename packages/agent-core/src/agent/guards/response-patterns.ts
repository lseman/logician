// ── Shared response patterns ────────────────────────────────────────────────
// Centralized regex patterns for detecting non-committal, complete, and
// circling model responses. All consumers import from here — never define
// patterns inline.

// ── Non-committal patterns ──────────────────────────────────────────────────
// Patterns that indicate the model is hedging or deferring action rather than
// taking a concrete step. Used to detect "thinking without acting" loops.
//
// CRITICAL: These patterns must be SPECIFIC to actual hedging behavior.
// They must NOT match normal planning language like "let me think about the
// approach" or "I'll try reading the file" which are legitimate agent actions.
//
// A non-committal response is one that signals uncertainty or intention to
// act but contains NO concrete next step, no tool call intent, and no
// actionable content. The key differentiator: does the response end without
// committing to a specific action?

export const NON_COMMITTAL_PATTERNS: ReadonlyArray<RegExp> = [
	// "I'm not sure what to do next" — genuine uncertainty without a plan
	/\b(i'm|i am)\s+(not\s+sure|unsure|stuck)\b.*?\b(i'll|let\s+me|i\s+should)\s+(think|consider|see)\b/i,
	// "I need to figure out" — open-ended investigation with no concrete step
	/\b(i\s+need\s+to|i\s+should)\s+(figure\s+out|work\s+out|sort\s+out)\b.*?\b(let\s+me|first|then)\b/i,
	// "I'm going to think about this more" — pure deliberation with no action
	/\b(i'm\s+going\s+to|i\s+will)\s+(think\s+about|consider|ponder)\s+(this|it|more)\b/i,
	// "I don't know how to proceed" — uncertainty without alternative
	/\b(i\s+don't\s+know\s+how|i\s+don't\s+know\s+what\s+to\s+do)\b.*?\b(let\s+me|i'll|i\s+should)\b/i,
	// "This requires more thought" — deferring action indefinitely
	/\b(this\s+(requires|needs|calls\s+for)\s+(more\s+thought|further\s+consideration|more\s+analysis))\b/i,
	// "I'm still not sure" — persistent uncertainty
	/\b(i'm\s+still\s+(not\s+sure|unsure|confused)|i\s+still\s+(don't|do\s+not)\s+know)\b/i,
	// "Let me reconsider" — abandoning current approach without stating new one
	/\blet\s+me\s+(reconsider|rethink|step\s+back\s+and\s+think)\b/i,
	// "At this point I'm not sure" — giving up without direction
	/\b(at\s+this\s+point)\s+(i\s+(am|'m)\s+(not\s+sure|unsure|confused)|we\s+(need|should)\s+(to\s+)?(?:think|consider))\b/i,
	// "I still need to check X" / "I have not finished yet" / "more work to do" —
	// bare unfinished-work statements with no concrete action taken this turn.
	// Distinct from the hedging patterns above: no "let me think" clause needed,
	// this is the plain "work remains" signal the continuation nudge relies on.
	/\b(i\s+(?:still\s+)?(?:need|have)\s+(?:to|not)\b.*?\b(?:check|verify|finish|complete|fix|test|review|investigate|update|implement)\b|(?:not|isn.?t)\s+(?:yet\s+)?(?:done|finished|complete)\b|more\s+work\s+(?:to\s+do|remains|left))\b/i,
];

// ── Completion patterns ─────────────────────────────────────────────────────
// Patterns that indicate the model has declared task completion.

const COMPLETE_PATTERNS: ReadonlyArray<RegExp> = [
	/\b(task\s+complete|all\s+done|finished|completed\s+successfully|nothing\s+(else|more)\s+to\s+do|no\s+(further|more)\s+(steps?|action|work)|that('s|\s+is)\s+(all|done|complete))\b/i,
	/^done\s*$/i,
];

// ── Meta-reasoning patterns ────────────────────────────────────────────────
// Patterns indicating the model is reasoning about its own reasoning rather
// than taking action. These capture the "meta-loop" signature.
//
// CRITICAL: These patterns must target ACTUAL meta-reasoning loops — where the
// model is stuck in "thinking about thinking" without progressing to action.
// They must NOT match normal planning language like "let me think about the
// approach before I implement" or "I should first understand the codebase".
//
// A meta-reasoning loop is characterized by:
// - Repeated cycles of "let me think" without subsequent action
// - Escalating length of reasoning without tool calls
// - Self-referential reasoning that doesn't produce concrete next steps
// - Abandoning current approach without stating a new one

export const META_REASONING_PATTERNS: ReadonlyArray<RegExp> = [
	// "Let me think about how to approach this" — planning about planning with no action
	/\blet\s+me\s+(think|consider)\s+(about\s+)?(?:how\s+)?(?:to\s+)?(?:approach|handle|solve)\s+(this|it)\b.*?\b(let\s+me|first|then)\s+(think|consider|see)\b/i,
	// "I need to think about X before I do Y" — procrastination disguised as planning
	/\b(i\s+need\s+to|i\s+should)\s+(think\s+about|consider)\s+.*?\bbefore\s+(i\s+)?(?:do|proceed|start|implement|write)\b/i,
	// "I'm not sure about X, let me think about it" — hesitation loop
	/\b(i'm\s+not\s+sure|not\s+sure\s+about)\b.*?\b(let\s+me\s+think|let\s+me\s+consider)\b/i,
	// "I should first understand... then I can..." — endless analysis without action
	/\b(i\s+should\s+first|first\s+i\s+need\s+to)\s+(understand|comprehend|grasp|analyze)\s+.*?\bthen\s+(i\s+)?(?:can|will|should)\s+(think|consider|decide)\b/i,
	// "Upon further reflection" — meta-reasoning escalation without action
	/\b(?:upon\s+further\s+reflection|after\s+considering)\b.*?\b(let\s+me\s+think|let\s+me\s+reconsider)\b/i,
	// "This requires me to think more" — deferring action indefinitely
	/\b(this\s+(?:requires|needs|demands)\s+(?:me\s+)?to\s+think\s+(more|about\s+it|further))\b/i,
	// "I realize I need to rethink" — meta-realization loop without new direction
	/\b(i\s+realize|i\s+see\s+that|i\s+understand)\s+(?:that\s+)?(i\s+need\s+to|i\s+should|i\s+must)\s+(rethink|reconsider|step\s+back)\b/i,
	// "Thinking through this... okay..." — self-talk spiral without resolution
	/\b(?:thinking\s+through\s+this|let\s+me\s+walk\s+through\s+this|working\s+through\s+this)\b.*?\b(?:okay|alright|right)\b/i,
	// "Let me step back and think about my approach" — abandoning current work without new plan
	/\blet\s+me\s+step\s+back\s+(?:to\s+)?(?:think|consider)\s+(about\s+(?:my|this)\s+)?(?:approach|strategy|plan)\b/i,
	// "I need to reflect on..." — pure meta-reasoning without action
	/\b(i\s+need\s+to|i\s+should)\s+(reflect\s+on|ponder\s+on)\s+.*?\b(let\s+me|first|then)\b/i,
];

// ── Circling patterns ───────────────────────────────────────────────────────
// Patterns that suggest the model is circling — retrying the same approach
// without success. Broader than stop declarations to escalate nudge tone.
// Stricter to avoid false positives on legitimate multi-step work.
//
// CRITICAL: These patterns must target ACTUAL circling — repeated failure
// followed by bare retry intent WITHOUT a changed strategy. They must NOT
// match normal multi-step work like "I'll try reading the file first" or
// "Let me attempt the bash command" which are legitimate first attempts.
//
// A circling pattern is: failure + retry of SAME approach without specifying
// what's different about the next attempt.

const CIRCLING_PATTERNS: ReadonlyArray<RegExp> = [
	// "I'll try again" — bare retry with NO mention of what's different
	/\bi(?:\s+will|'ll|ll)\s+(?:try|attempt)\s+again\b/i,
	// "Let me try again" — bare retry with NO mention of what's different
	/\blet\s+me\s+(?:try|attempt)\s+again\b/i,
	// "I've tried X again" — explicit past retry with "again" (no new strategy)
	/\b(?:i'|ve|I've)\s+(?:tried|attempted)\s+(?:to\s+)?(?:the\s+)?(?:same\s+)?(?:approach|way|method)?\s*(?:again|yet)\b/i,
	// "I tried X but it failed, let me try again" — failure + bare retry
	/\bi\s+(?:tried|attempted)\b.*?\b(?:but|however|unfortunately)\b.*?\b(?:failed|didn't\s+work|unable)\b.*?\b(?:let\s+me\s+)?(?:try|attempt)\s+again\b/i,
	// "I tried X again" — explicit past retry with "again" (no new strategy)
	/\bi\s+(?:tried|attempted)\s+.*?\b(?:again|yet)\b.*?\b(?:but|however)\b.*?\b(?:try|attempt)\b/i,
	// "I'll attempt the same thing" — retrying without change
	/\b(?:i'll|i will)\s+(?:attempt|try)\s+(?:the\s+)?(?:same|identical|similar)\b/i,
	// "cannot/can't X but try/attempt" — failed then retrying without explaining what's different
	/\b(cannot|can't|unable)\s+.*?\b(?:but|however|instead)\b.*?\b(?:try|attempt|do|make|go)\b/i,
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
		NON_COMMITTAL_PATTERNS.some(pattern => pattern.test(text))
	);
}

/**
 * Check if a text response declares task completion.
 */
export function looksComplete(text: string): boolean {
	return Boolean(text) && COMPLETE_PATTERNS.some(pattern => pattern.test(text));
}

/**
 * Check whether the assistant ended by handing control back to the user.
 *
 * Keep this deliberately focused on the end of the response. Questions used
 * while explaining or reasoning should not suppress continuation, but a final
 * question or an explicit request for information must always pause the loop.
 */
export function awaitsUserInput(text: string): boolean {
	const trimmed = text.trim();
	if (!trimmed) return false;

	// A question at the end may be followed by markdown emphasis or a closing
	// quote/bracket. Choice lists often follow the actual question, so inspect
	// the final short block as well as the final character.
	if (/\?[\s*_`"'”’)\]]*$/.test(trimmed)) return true;
	const lastQuestion = trimmed.lastIndexOf("?");
	if (lastQuestion >= 0) {
		const trailingLines = trimmed
			.slice(lastQuestion + 1)
			.split(/\r?\n/)
			.map(line => line.trim())
			.filter(Boolean);
		if (
			trailingLines.length > 0 &&
			trailingLines.every(line =>
				/^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+\S/.test(line),
			)
		) {
			return true;
		}
	}

	const tail = trimmed.slice(-800);
	return (
		/\b(?:please|kindly)\s+(?:answer|choose|confirm|clarify|provide|select|share|tell me)\b[^.!?]*[.!:]?\s*$/i.test(
			tail,
		) ||
		/\b(?:let me know|tell me)\s+(?:which|whether|what|when|where|who|how|if)\b[^.!?]*[.!:]?\s*$/i.test(
			tail,
		) ||
		/\b(?:i need|we need)\s+(?:your|the user's)\s+(?:answer|choice|confirmation|decision|input|permission)\b[^.!?]*[.!:]?\s*$/i.test(
			tail,
		)
	);
}

/**
 * Check if a text response indicates circling — retrying the same approach
 * without success. Inspects the full text since circling patterns appear
 * anywhere in the response.
 */
export function detectsCircling(assistantText: string): boolean {
	if (!assistantText || assistantText.trim().length < 10) return false;
	const lower = assistantText.toLowerCase();
	return CIRCLING_PATTERNS.some(re => re.test(lower));
}
