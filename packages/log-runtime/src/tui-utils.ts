/** Display formatting shared by agent-core consumers. */

/** Format a token count for display (1234 → "1.2k"). */
function formatTokenCount(tokens: number): string {
	if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}m`;
	if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
	return String(tokens);
}

/** Format context size with optional max (12k/32k or just "12k"). */
export function formatContextSize(tokens: number, maxTokens?: number): string {
	const current = formatTokenCount(Math.max(0, Math.round(tokens || 0)));
	if (!maxTokens || maxTokens <= 0) return current;
	return `${current}/${formatTokenCount(Math.round(maxTokens))}`;
}
