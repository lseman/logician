// ── ProgressSignal ───────────────────────────────────────────────────────────
// Measures whether the agent is making real progress toward its objective,
// not just producing new tool-call shapes.
//
// Current stagnation detection (LoopDetector) uses "no new tool shapes" which
// is weak — the agent can call 5 different tools on the same wrong file and
// never make progress. ProgressSignal tracks:
//
// 1. **File-level progress** — distinct files changed, not just edits to the
//    same file. A file that changes across 10 turns but never advances toward
//    the objective is not progress.
//
// 2. **Verification progress** — test results, lint results, build results.
//    If tests were failing and now pass, that's progress. If they were passing
//    and now fail, that's regression.
//
// 3. **Evidence quality** — not all evidence is equal. An observation that
//    eliminates a hypothesis is more valuable than one that generates a new
//    question.
//
// 4. **Phase advancement** — the agent should move through orient → investigate
//    → implement → verify. Staying in one phase for too long is a signal.
//
// The signal is a score (0–100) with breakdown by category, plus a list of
// "stuck reasons" that explain why progress is low.

// ── Types ───────────────────────────────────────────────────────────────────

/** A single progress observation from a tool call or turn. */
export interface ProgressObservation {
	/** What happened — e.g. "edit_file on src/auth.ts", "bash: npm test passed". */
	kind: "change" | "observation" | "verification" | "failure" | "regression";
	/** Brief description of the observation. */
	summary: string;
	/** Which tool produced it. */
	tool?: string;
	/** Which file it affected (if any). */
	file?: string;
	/** Whether this advances the task phase. */
	advancesPhase?: boolean;
	/** Whether this is a regression (tests failing, etc.). */
	isRegression?: boolean;
	/** The iteration this happened at. */
	iteration: number;
}

/** Progress score breakdown by category. */
export interface ProgressBreakdown {
	/** File-level progress: distinct files changed with meaningful edits. */
	files: { score: number; total: number; meaningful: number };
	/** Verification progress: tests passing, lint clean, etc. */
	verification: { score: number; total: number; passing: number };
	/** Phase advancement: moved through orient → investigate → implement → verify. */
	phase: { score: number; current: string; expected: string };
	/** Evidence quality: observations that eliminate hypotheses vs generate questions. */
	evidence: { score: number; total: number; highQuality: number };
}

/** Overall progress signal for a turn. */
export interface ProgressSignal {
	/** Overall score 0–100. */
	score: number;
	/** Breakdown by category. */
	breakdown: ProgressBreakdown;
	/** Why progress is low, if any. */
	stuckReasons: string[];
	/** Whether the agent should be nudged. */
	shouldNudge: boolean;
	/** The nudge message if shouldNudge is true. */
	nudgeMessage?: string;
}

/** Configuration for progress tracking. */
export interface ProgressSignalConfig {
	/** Minimum score before nudging (default 30). */
	minScoreBeforeNudge?: number;
	/** Minimum turns with low score before nudging (default 5). */
	minLowScoreTurns?: number;
	/** Whether to track file-level progress (default true). */
	trackFiles?: boolean;
	/** Whether to track verification progress (default true). */
	trackVerification?: boolean;
	/** Whether to track phase advancement (default true). */
	trackPhase?: boolean;
	/** Whether to track evidence quality (default true). */
	trackEvidence?: boolean;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const DEFAULT_CONFIG: Required<ProgressSignalConfig> = {
	minScoreBeforeNudge: 30,
	minLowScoreTurns: 5,
	trackFiles: true,
	trackVerification: true,
	trackPhase: true,
	trackEvidence: true,
};

// Patterns that indicate meaningful file changes (not just formatting).
const MEANINGFUL_CHANGE_PATTERNS = [
	/\b(add|remove|import|export|function|class|const|let|var|interface|type|def|func|struct|enum)\b/,
	/\b(if|else|return|throw|catch|try|finally|switch|case|break|continue)\b/,
	/\b(test|spec|describe|it|assert|expect|should|verify|check)\b/,
	/\b(TODO|FIXME|HACK|XXX|BUG|NOTE|WARNING)\b/,
];

// Patterns that indicate verification results.
const VERIFY_PASS_PATTERN =
	/\b(pass|success|ok|clean|no errors?|all tests?\s+(pass|ok|green)|\d+\/\d+\s+(pass|ok|green))\b/i;
const VERIFY_FAIL_PATTERN =
	/\b(fail|error|exception|traceback|not ok|exit(?:ed)? (?:code )?[1-9]|test\s+failed|\d+\/\d+\s+(fail|error|red))\b/i;

// Phase ordering for scoring advancement. "blocked" is intentionally excluded —
// it isn't a forward-progress phase, and is handled as a special case in
// scorePhase() rather than participating in the ordinal ranking below.
const PHASE_ORDER = ["orient", "investigate", "implement", "verify", "handoff"];

// ── ProgressSignal class ─────────────────────────────────────────────────────

export class ProgressSignalTracker {
	private config: Required<ProgressSignalConfig>;
	private observations: ProgressObservation[] = [];
	private lowScoreTurns = 0;
	private lastScore = 100;
	private currentPhase = "orient";

	constructor(config: ProgressSignalConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	/**
	 * Record a batch of tool call results and compute progress.
	 * Returns the progress signal for this turn.
	 */
	record(
		calls: Array<{ name: string; args: string }>,
		results: Array<{ content: string; file?: string }>,
		iteration: number,
		currentPhase: string,
	): ProgressSignal {
		this.currentPhase = currentPhase;
		const newObservations: ProgressObservation[] = [];

		for (let i = 0; i < calls.length; i++) {
			const call = calls[i];
			const result = results[i];
			const resultContent = String(result?.content ?? "");

			// Detect file changes
			if (this.config.trackFiles) {
				const file = result?.file ?? this.extractFile(call);
				if (file) {
					const isMeaningful = this.isMeaningfulChange(resultContent);
					newObservations.push({
						kind: isMeaningful ? "change" : "observation",
						summary: `${call.name} on ${file.slice(0, 120)}`,
						tool: call.name,
						file,
						advancesPhase: isMeaningful,
						iteration,
					});
				}
			}

			// Detect verification results
			if (this.config.trackVerification) {
				if (VERIFY_PASS_PATTERN.test(resultContent)) {
					newObservations.push({
						kind: "verification",
						summary: `Verification passed: ${resultContent.slice(0, 200)}`,
						tool: call.name,
						advancesPhase: true,
						iteration,
					});
				} else if (VERIFY_FAIL_PATTERN.test(resultContent)) {
					newObservations.push({
						kind: resultContent.includes("regression") || resultContent.includes("new fail") ? "regression" : "failure",
						summary: `Verification failed: ${resultContent.slice(0, 200)}`,
						tool: call.name,
						isRegression: resultContent.includes("regression") || resultContent.includes("new fail"),
						iteration,
					});
				}
			}

			// Detect failures
			if (resultContent.includes("Error:") || resultContent.includes("error:")) {
				newObservations.push({
					kind: "failure",
					summary: resultContent.slice(0, 300),
					tool: call.name,
					iteration,
				});
			}
		}

		// Merge observations
		this.observations.push(...newObservations);

		// Track low-score turns *before* computing the signal, so shouldNudge
		// (computed inside computeSignal() from this.lowScoreTurns) reflects
		// this turn's score rather than lagging a turn behind.
		const provisionalScore = this.computeSignal().score;
		if (provisionalScore < this.config.minScoreBeforeNudge) {
			this.lowScoreTurns++;
		} else {
			this.lowScoreTurns = 0;
		}

		const signal = this.computeSignal();
		this.lastScore = signal.score;
		return signal;
	}

	/**
	 * Check if the agent should be nudged based on accumulated low progress.
	 */
	shouldNudge(): boolean {
		return (
			this.lowScoreTurns >= this.config.minLowScoreTurns &&
			this.lastScore < this.config.minScoreBeforeNudge
		);
	}

	/**
	 * Get the current stuck reasons for diagnostics.
	 */
	getStuckReasons(): string[] {
		return this.computeSignal().stuckReasons;
	}

	/**
	 * Reset the tracker (e.g. after a successful subgoal).
	 */
	reset(): void {
		this.observations = [];
		this.lowScoreTurns = 0;
		this.lastScore = 100;
	}

	/**
	 * Get all observations for diagnostics.
	 */
	getObservations(): ProgressObservation[] {
		return [...this.observations];
	}

	// ── Internals ───────────────────────────────────────────────────────────

	private computeSignal(): ProgressSignal {
		const breakdown: ProgressBreakdown = {
			files: this.scoreFiles(),
			verification: this.scoreVerification(),
			phase: this.scorePhase(),
			evidence: this.scoreEvidence(),
		};

		const score = Math.round(
			(breakdown.files.score * 0.3 +
				breakdown.verification.score * 0.3 +
				breakdown.phase.score * 0.2 +
				breakdown.evidence.score * 0.2),
		);

		const stuckReasons: string[] = [];
		if (breakdown.files.score < 30) {
			stuckReasons.push(
				`Only ${breakdown.files.meaningful}/${breakdown.files.total} file changes are meaningful. Consider whether you're editing the right files.`,
			);
		}
		if (breakdown.verification.score < 30) {
			stuckReasons.push(
				`Verification is not improving: ${breakdown.verification.passing}/${breakdown.verification.total} passed. Check if tests are actually running.`,
			);
		}
		if (breakdown.phase.score < 30) {
			stuckReasons.push(
				`Stuck in "${breakdown.phase.current}" phase. Expected to be in "${breakdown.phase.expected}". Consider advancing to the next phase.`,
			);
		}
		if (breakdown.evidence.score < 30) {
			stuckReasons.push(
				`Most observations are low-quality (not eliminating hypotheses). Consider testing specific assumptions.`,
			);
		}

		const shouldNudge =
			score < this.config.minScoreBeforeNudge &&
			this.lowScoreTurns >= this.config.minLowScoreTurns;

		const nudgeMessage = shouldNudge
			? this.buildNudgeMessage(stuckReasons)
			: undefined;

		return {
			score,
			breakdown,
			stuckReasons,
			shouldNudge,
			nudgeMessage,
		};
	}

	private scoreFiles(): ProgressBreakdown["files"] {
		const fileChanges = this.observations.filter(
			(o) => o.kind === "change" || o.kind === "observation",
		);
		const total = fileChanges.length;
		const meaningful = fileChanges.filter((o) => o.advancesPhase).length;

		if (total === 0) return { score: 0, total: 0, meaningful: 0 };

		// Count distinct files
		const distinctFiles = new Set(fileChanges.map((o) => o.file).filter(Boolean));
		const distinctFileRatio = distinctFiles.size / Math.max(total, 1);

		// Score: meaningful changes weighted higher, distinct files bonus
		const score = Math.min(
			100,
			Math.round(
				(meaningful / Math.max(total, 1)) * 60 +
					distinctFileRatio * 40,
			),
		);

		return { score, total, meaningful };
	}

	private scoreVerification(): ProgressBreakdown["verification"] {
		const verifications = this.observations.filter(
			(o) => o.kind === "verification" || o.kind === "failure" || o.kind === "regression",
		);
		const total = verifications.length;
		const passing = verifications.filter((o) => o.kind === "verification").length;

		if (total === 0) return { score: 50, total: 0, passing: 0 }; // neutral if no verification

		const passRatio = passing / total;
		const score = Math.round(passRatio * 100);

		return { score, total, passing };
	}

	private scorePhase(): ProgressBreakdown["phase"] {
		// "blocked" is a stall, not a forward-progress phase — score it at the
		// floor and expect a return to "investigate" rather than ranking it
		// against PHASE_ORDER (where an absent phase would otherwise look like
		// index -1, silently scoring 0 for the wrong reason).
		if (this.currentPhase === "blocked") {
			return { score: 0, current: this.currentPhase, expected: "investigate" };
		}

		const currentIndex = PHASE_ORDER.indexOf(this.currentPhase);
		const expectedIndex = Math.min(
			currentIndex + 1,
			PHASE_ORDER.length - 1,
		);
		const expectedPhase = PHASE_ORDER[expectedIndex];

		// Score based on whether we've advanced from orient
		const score =
			currentIndex > 0
				? Math.min(100, (currentIndex / (PHASE_ORDER.length - 1)) * 100)
				: 0;

		return { score, current: this.currentPhase, expected: expectedPhase };
	}

	private scoreEvidence(): ProgressBreakdown["evidence"] {
		const observations = this.observations.filter(
			(o) => o.kind === "observation" || o.kind === "change",
		);
		const total = observations.length;
		const highQuality = observations.filter((o) => o.advancesPhase).length;

		if (total === 0) return { score: 50, total: 0, highQuality: 0 }; // neutral

		const score = Math.round((highQuality / total) * 100);
		return { score, total, highQuality };
	}

	private isMeaningfulChange(content: string): boolean {
		return MEANINGFUL_CHANGE_PATTERNS.some((p) => p.test(content));
	}

	private extractFile(call: { name: string; args: string }): string | undefined {
		try {
			const args = JSON.parse(call.args) as Record<string, unknown>;
			const file =
				(args.path as string) ??
				(args.file as string) ??
				(args.filePath as string) ??
				(args.filename as string);
			return file ? String(file) : undefined;
		} catch {
			return undefined;
		}
	}

	private buildNudgeMessage(stuckReasons: string[]): string {
		const reason = stuckReasons[0] ?? "Low progress detected.";
		return `[progress-signal:low-progress] Progress score is low (${this.lastScore}/100). ${reason} Consider: (1) Are you working on the right files? (2) Are tests actually running? (3) Have you advanced past the investigation phase?`;
	}
}
