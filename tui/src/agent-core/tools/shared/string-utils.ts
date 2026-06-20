// ── String utilities ──────────────────────────────────────────────────────────

// Matches all ANSI escape sequences (CSI, OSC, simple escapes, etc.)
const ANSI_RE =
	// eslint-disable-next-line no-control-regex
	/[](?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])|[PX^_].*?\\|\][^]*(?:|\\)/g;

/**
 * Strip ANSI escape codes from a string.
 * Ported from pi packages/coding-agent/src/utils/ansi.ts.
 */
export function stripAnsi(input: string): string {
	return input.replace(ANSI_RE, "");
}

/**
 * Decode common HTML entities.
 * Ported from pi packages/coding-agent/src/utils/html.ts.
 */
export function decodeHtmlEntities(input: string): string {
	return input
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;/g, "'")
		.replace(/&nbsp;/g, " ");
}
