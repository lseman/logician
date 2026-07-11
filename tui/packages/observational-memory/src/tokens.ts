// ── Token estimation ─────────────────────────────────────────────────────
// Lightweight character-to-token heuristic. No external deps.

const AVG_CHARS_PER_TOKEN = 4;

export function estimateTokens(text: string): number {
	if (!text) return 0;
	return Math.ceil(text.length / AVG_CHARS_PER_TOKEN);
}

export function estimateObservationTokens(obs: { content: string }): number {
	return estimateTokens(obs.content);
}

export function estimateReflectionTokens(ref: { content: string }): number {
	return estimateTokens(ref.content);
}
