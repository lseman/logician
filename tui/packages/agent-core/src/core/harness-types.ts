// ── Shared harness types ───────────────────────────────────────────────────
// Extracted so branching/compaction/model helper modules can share these
// shapes with harness.ts without importing the harness class itself.

/** Progress tracking for branch summaries. */
export interface BranchProgress {
	done: string[];
	inProgress: string[];
	blocked: string[];
}

/** Structured branch summary data. */
export interface BranchSummaryData {
	/** Goal of the branch. */
	goal: string;
	/** Constraints and preferences. */
	constraints: string[];
	/** Progress tracking. */
	progress: BranchProgress;
	/** Key decisions with rationale. */
	keyDecisions: Array<{ decision: string; rationale: string }>;
	/** Next steps to continue work. */
	nextSteps: string[];
	/** Full human-readable summary. */
	full: string;
}

/** Public branch info for UI / callers. */
export interface BranchInfo {
	id: string;
	depth: number;
	/** Summary of this branch's work (null until branchSummary is called). */
	summary: BranchSummaryData | null;
	/** Message index where this branch forked. */
	forkedAt: number;
}
