// ── Shared response patterns ────────────────────────────────────────────────
// Centralized patterns for unambiguous response signals. All consumers import
// from here — never define patterns inline.

// ── Completion patterns ─────────────────────────────────────────────────────
// Patterns that indicate the model has declared task completion.

const COMPLETE_PATTERNS: ReadonlyArray<RegExp> = [
	/\b(task\s+complete|all\s+done|finished|completed\s+successfully|nothing\s+(else|more)\s+to\s+do|no\s+(further|more)\s+(steps?|action|work)|that('s|\s+is)\s+(all|done|complete))\b/i,
	/^done\s*$/i,
];

// ── Query helpers ───────────────────────────────────────────────────────────

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

	// A trailing question mark alone is not sufficient to declare user-wait:
	// models reason aloud with long, self-directed questions
	// ("Let me check if this works, or should I try another approach?").
	// Short, direct questions aimed at the user are genuine handoffs.
	if (/\?[\s*_`"'"]*\s*$/.test(trimmed)) {
		const lastSentence = trimmed.split(/(?:[.!?:]|\n{2,})/).pop()?.trim() ?? "";
		// Self-directed reasoning: long (>= 35 chars), contains hedging/
		// conditional markers, and does not contain a direct user request.
		if (
			lastSentence.length >= 35 &&
			/\b(?:check|verify|try|approach|think|consider|maybe|perhaps)/i.test(
				lastSentence,
			) &&
			!/\b(?:what should|how can|i need your|please |tell me which|let me know|can you help|should i proceed|choose |confirm )/i.test(lastSentence)
		) {
			return false; // reasoning, not user handoff
		}
		// Not a rejected reasoning question — it's a bare trailing question,
		// which is a genuine user handoff. Return true before checking for
		// option lists (already matched above by the ?-at-end check).
		return true;
	}
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
