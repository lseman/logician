// ── Truncation config ────────────────────────────────────────────────────────
// Single source of truth for every "cap this text/output at N" limit in the
// harness. Previously each of these was a separate hardcoded constant
// scattered across tool registry, compaction, subagents, and the TUI
// transcript view.

export interface TruncationConfig {
	/** Cap on tool result content appended to context. 0 disables. */
	toolResultMaxChars?: number;
	/** Cap on lines read by file/search tools (read, find, list-files, grep). */
	maxLines?: number;
	/** Max chars per matched line in grep-style output. */
	grepLineMaxChars?: number;
	/** Cap on subagent report text bubbled up to the parent context. */
	subagentResultMaxChars?: number;
	/** Cap on tool-result text folded into a compaction summary. */
	compactionSummaryMaxChars?: number;
	/** Per-role cap used by micro-compaction when trimming oversized message bodies. */
	microCompactMaxChars?: {
		tool?: number;
		assistant?: number;
		default?: number;
	};
	/** Cap on a single rendered message in the TUI transcript view. */
	transcriptMessageMaxChars?: number;
}

export const DEFAULT_TRUNCATION: Required<
	Omit<TruncationConfig, "microCompactMaxChars">
> & { microCompactMaxChars: Required<NonNullable<TruncationConfig["microCompactMaxChars"]>> } = {
	toolResultMaxChars: 100_000,
	maxLines: 2000,
	grepLineMaxChars: 500,
	subagentResultMaxChars: 16_000,
	compactionSummaryMaxChars: 2000,
	microCompactMaxChars: {
		tool: 4000,
		assistant: 10_000,
		default: 14_000,
	},
	transcriptMessageMaxChars: 4000,
};

/** Merge a partial override on top of the defaults, one level deep. */
export function resolveTruncationConfig(
	overrides?: TruncationConfig,
): typeof DEFAULT_TRUNCATION {
	if (!overrides) return DEFAULT_TRUNCATION;
	return {
		...DEFAULT_TRUNCATION,
		...overrides,
		microCompactMaxChars: {
			...DEFAULT_TRUNCATION.microCompactMaxChars,
			...overrides.microCompactMaxChars,
		},
	};
}
