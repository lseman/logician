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

	// A question at the end may be followed by markdown emphasis or a closing
	// quote/bracket. Choice lists often follow the actual question, so inspect
	// the final short block as well as the final character.
	if (/\?[\s*_`"'"]*\s*$/.test(trimmed)) return true;
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
