/** Display formatting shared by agent-core consumers. */

/** Format a token count for display (1234 → "1.2k"). */
export function formatTokenCount(tokens: number): string {
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

/** Format a delay in ms to human-readable (15000 → "15s"). */
export function formatDelay(ms: number): string {
	if (ms < 1000) return `${ms}ms`;
	if (ms < 60_000) return `${Math.round(ms / 1000)}s`;
	return `${Math.round(ms / 60_000)}m`;
}

/** Parse an environment variable to a number, returning undefined on failure. */
export function envNumber(name: string): number | undefined {
	const raw = process.env[name];
	if (!raw) return undefined;
	const value = Number(raw);
	return Number.isFinite(value) ? value : undefined;
}

/** Escape a value for use in a markdown table cell. */
export function escapeTable(value: string): string {
	return value
		.replace(/\\/g, "\\\\")
		.replace(/\|/g, "\\|")
		.replace(/\n/g, "\\n");
}

/** Format a table row from an array of cell values. */
export function tableRow(values: string[]): string {
	return `| ${values.join(" | ")} |`;
}

/** Parse an interval token like "5m", "30s", "1h", "2d" into ms. Returns null if not valid. */
export function parseInterval(arg: string): number | null {
	const m = arg.match(/^(\d+)(s|m|h|d)$/);
	if (!m) return null;
	const [, value, unit] = m;
	const n = parseInt(value, 10);
	switch (unit) {
		case "s":
			return n * 1000;
		case "m":
			return n * 60_000;
		case "h":
			return n * 3_600_000;
		case "d":
			return n * 86_400_000;
	}
	return null;
}
