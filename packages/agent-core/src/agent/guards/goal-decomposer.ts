// ── GoalDecomposer ───────────────────────────────────────────────────────────
// Breaks a high-level objective into subgoals with milestones, then tracks
// which subgoals are complete, in-progress, or blocked.
//
// SOTA motivation: DeepResearch and OpenDevin both decompose goals into
// subgoals before acting. Logician's current system has one objective string
// with no decomposition — the agent doesn't know "I've done step 2 of 5"
// and can't self-correct when a subgoal fails.
//
// How it works:
// 1. On first call, the decomposer asks the model (via a steering message)
//    to produce a subgoal breakdown in a structured format.
// 2. The breakdown is stored as a list of subgoals, each with:
//    - id, description, status (pending/in-progress/completed/blocked)
//    - verification command (how to check it's done)
//    - estimated effort (low/medium/high)
// 3. After each turn, the decomposer checks whether any subgoal has been
//    completed (by analyzing tool results, file changes, etc.).
// 4. If a subgoal is blocked, it records the blocker and nudges the model
//    to try a different approach.
// 5. When all subgoals are complete, the decomposer signals that the task
//    is done.

export type SubgoalStatus = "pending" | "in-progress" | "completed" | "blocked";
export type EffortEstimate = "low" | "medium" | "high";

export interface Subgoal {
	id: string;
	description: string;
	status: SubgoalStatus;
	verificationCommand?: string;
	effort: EffortEstimate;
	blocker?: string;
	completedAt?: number;
}

export interface GoalBreakdown {
	objective: string;
	subgoals: Subgoal[];
	completedCount: number;
	totalCount: number;
	completionPercentage: number;
}

export interface GoalDecomposerConfig {
	/** Whether to auto-decompose on first call (default true). */
	autoDecompose?: boolean;
	/** Max subgoals to allow (default 10). */
	maxSubgoals?: number;
	/** Min subgoals before auto-completing (default 3). */
	minSubgoalsForAutoComplete?: number;
	/** Whether to track subgoal-level progress (default true). */
	trackSubgoals?: boolean;
}

const DEFAULT_CONFIG: Required<GoalDecomposerConfig> = {
	autoDecompose: true,
	maxSubgoals: 10,
	minSubgoalsForAutoComplete: 3,
	trackSubgoals: true,
};

// ── Parsing ──────────────────────────────────────────────────────────────────

/**
 * Parse a subgoal breakdown from model output.
 * Expects a markdown list with subgoals, optionally with verification commands.
 */
export function parseSubgoalBreakdown(
	text: string,
	objective: string,
): GoalBreakdown | null {
	const lines = text.split("\n");
	const subgoals: Subgoal[] = [];
	let currentSubgoal: Partial<Subgoal> | null = null;
	let idCounter = 0;

	for (const line of lines) {
		// Match numbered or bulleted subgoal lines
		const match = line.match(/^\s*(?:\d+\.|\-|\*)\s+(.+)$/);
		if (match) {
			if (currentSubgoal) {
				subgoals.push(currentSubgoal as Subgoal);
			}
			const desc = match[1].trim();
			idCounter++;
			currentSubgoal = {
				id: `sg-${idCounter}`,
				description: desc,
				status: "pending",
				effort: estimateEffort(desc),
			};
		} else if (currentSubgoal && line.match(/^\s+verification:/i)) {
			const cmd = line.replace(/^\s+verification:\s*/i, "").trim();
			if (cmd) {
				currentSubgoal.verificationCommand = cmd;
			}
		} else if (currentSubgoal && line.match(/^\s+effort:/i)) {
			const effort = line.replace(/^\s+effort:\s*/i, "").trim().toLowerCase();
			if (["low", "medium", "high"].includes(effort)) {
				currentSubgoal.effort = effort as EffortEstimate;
			}
		}
	}

	if (currentSubgoal) {
		subgoals.push(currentSubgoal as Subgoal);
	}

	if (subgoals.length === 0) return null;

	return {
		objective,
		subgoals,
		completedCount: 0,
		totalCount: subgoals.length,
		completionPercentage: 0,
	};
}

/**
 * Estimate effort from subgoal description.
 */
function estimateEffort(description: string): EffortEstimate {
	const lower = description.toLowerCase();
	if (
		/\b(read|grep|search|list|find|check|look)\b/.test(lower)
	) {
		return "low";
	}
	if (
		/\b(edit|write|create|implement|add|fix|change|refactor)\b/.test(lower)
	) {
		return "medium";
	}
	if (
		/\b(test|verify|build|deploy|integrate|design|architect)\b/.test(lower)
	) {
		return "high";
	}
	return "medium";
}

// ── Progress checking ────────────────────────────────────────────────────────

/**
 * Check if a subgoal has been completed based on recent observations.
 * Simple heuristic: if the subgoal mentions a file and that file has been
 * changed, mark it as completed.
 */
export function checkSubgoalProgress(
	subgoal: Subgoal,
	recentChanges: Array<{ file?: string; tool?: string; content?: string }>,
): { completed: boolean; reason?: string } {
	if (subgoal.status !== "in-progress") return { completed: false };

	const desc = subgoal.description.toLowerCase();

	// Check if any recent change matches the subgoal description
	for (const change of recentChanges) {
		const file = change.file ?? "";
		const content = change.content ?? "";
		const tool = change.tool ?? "";

		// File-based matching
		if (file && desc.includes(file.toLowerCase().split("/").pop() ?? "")) {
			return { completed: true, reason: `File ${file} was modified` };
		}

		// Tool-based matching
		if (tool === "bash" && subgoal.verificationCommand) {
			// If the verification command matches the bash call
			if (content.includes("passed") || content.includes("success")) {
				return { completed: true, reason: "Verification passed" };
			}
		}

		// Content-based matching
		if (
			content.includes("created") ||
			content.includes("completed") ||
			content.includes("done")
		) {
			// If the subgoal is about creating something and we see completion
			if (/\b(create|write|add|implement)\b/.test(desc)) {
				return { completed: true, reason: "Completion signal detected" };
			}
		}
	}

	return { completed: false };
}

// ── GoalDecomposer class ─────────────────────────────────────────────────────

export class GoalDecomposer {
	private config: Required<GoalDecomposerConfig>;
	private breakdown: GoalBreakdown | null = null;
	private decomposed = false;

	constructor(config: GoalDecomposerConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	/**
	 * Set the goal breakdown (from model-generated or manual input).
	 */
	setBreakdown(breakdown: GoalBreakdown): void {
		this.breakdown = breakdown;
		this.decomposed = true;
		this.advanceToNextPending();
	}

	/**
	 * Parse and set a breakdown from model output.
	 */
	parseFromText(text: string, objective: string): boolean {
		const parsed = parseSubgoalBreakdown(text, objective);
		if (!parsed) return false;
		this.breakdown = parsed;
		this.decomposed = true;
		this.advanceToNextPending();
		return true;
	}

	/**
	 * Auto-advance the first pending subgoal to in-progress. Nothing calls
	 * markInProgress() externally (there's no model-facing tool for it), so
	 * without this, updateProgress()'s completion check — which only looks at
	 * in-progress subgoals — would never have anything to act on.
	 */
	private advanceToNextPending(): void {
		if (!this.breakdown) return;
		if (this.breakdown.subgoals.some(s => s.status === "in-progress")) return;
		const next = this.breakdown.subgoals.find(s => s.status === "pending");
		if (next) next.status = "in-progress";
	}

	/**
	 * Get the current breakdown.
	 */
	getBreakdown(): GoalBreakdown | null {
		return this.breakdown;
	}

	/**
	 * Check if decomposition is needed (hasn't been done yet).
	 */
	needsDecomposition(): boolean {
		return this.config.autoDecompose && !this.decomposed;
	}

	/**
	 * Build a steering message asking the model to decompose the goal.
	 */
	buildDecompositionPrompt(objective: string): string {
		return (
			"[goal-decomposition] Break this objective into 3-7 subgoals. " +
			"For each subgoal, provide: (1) a clear description, " +
			"(2) a verification command if applicable, (3) effort estimate (low/medium/high). " +
			"Format as a numbered list. Example:\n" +
			"1. Read the existing test file to understand the test structure\n" +
			"   effort: low\n" +
			"2. Add a new test case for the edge case\n" +
			"   verification: npm test -- edge-case.spec.ts\n" +
			"   effort: medium\n" +
			"3. Run the full test suite to verify no regressions\n" +
			"   verification: npm test\n" +
			"   effort: high"
		);
	}

	/**
	 * Update subgoal statuses based on recent observations.
	 * Returns updated breakdown or null if no decomposition exists.
	 */
	updateProgress(
		observations: Array<{
			file?: string;
			tool?: string;
			content?: string;
			kind?: "change" | "verification" | "failure";
		}>,
	): GoalBreakdown | null {
		if (!this.breakdown) return null;

		const subgoals = this.breakdown.subgoals;
		let completedCount = 0;

		for (const subgoal of subgoals) {
			if (subgoal.status === "completed") {
				completedCount++;
				continue;
			}

			if (subgoal.status === "in-progress") {
				const result = checkSubgoalProgress(subgoal, observations);
				if (result.completed) {
					subgoal.status = "completed";
					subgoal.completedAt = Date.now();
					completedCount++;
				}
			}
		}

		this.advanceToNextPending();

		this.breakdown = {
			...this.breakdown,
			completedCount,
			totalCount: subgoals.length,
			completionPercentage: Math.round(
				(completedCount / subgoals.length) * 100,
			),
		};

		return this.breakdown;
	}

	/**
	 * Mark a subgoal as in-progress.
	 */
	markInProgress(subgoalId: string): boolean {
		if (!this.breakdown) return false;
		const subgoal = this.breakdown.subgoals.find((s) => s.id === subgoalId);
		if (!subgoal || subgoal.status !== "pending") return false;
		subgoal.status = "in-progress";
		return true;
	}

	/**
	 * Mark a subgoal as blocked.
	 */
	markBlocked(subgoalId: string, reason: string): boolean {
		if (!this.breakdown) return false;
		const subgoal = this.breakdown.subgoals.find((s) => s.id === subgoalId);
		if (!subgoal) return false;
		subgoal.status = "blocked";
		subgoal.blocker = reason;
		return true;
	}

	/**
	 * Check if all subgoals are complete.
	 */
	isComplete(): boolean {
		if (!this.breakdown) return false;
		return (
			this.breakdown.completedCount === this.breakdown.totalCount &&
			this.breakdown.totalCount > 0
		);
	}

	/**
	 * Get a status summary for the model.
	 */
	getStatusSummary(): string | null {
		if (!this.breakdown) return null;
		const { completedCount, totalCount, subgoals } = this.breakdown;

		const lines = [`[goal-progress] ${completedCount}/${totalCount} subgoals complete (${this.breakdown.completionPercentage}%)`];

		for (const sg of subgoals) {
			const statusIcon =
				sg.status === "completed" ? "✓" :
				sg.status === "in-progress" ? "→" :
				sg.status === "blocked" ? "✗" : "○";
			lines.push(`  ${statusIcon} ${sg.id}: ${sg.description}${sg.blocker ? ` [BLOCKED: ${sg.blocker}]` : ""}`);
		}

		return lines.join("\n");
	}

	/**
	 * Reset the decomposer.
	 */
	reset(): void {
		this.breakdown = null;
		this.decomposed = false;
	}
}
