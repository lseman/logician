// ── HypothesisTracker ────────────────────────────────────────────────────────
// The agent maintains and falsifies hypotheses about why it's stuck.
//
// Current problem: when the model is stuck, it either (a) tries the same thing
// again, or (b) randomly guesses a new approach. Neither is systematic.
//
// HypothesisTracker solves this by:
// 1. When stuck, asking the model to generate hypotheses about WHY it's stuck.
// 2. Each hypothesis has: a statement, a test, and a status (active/falsified/verified).
// 3. After each turn, checking if any hypothesis was falsified (e.g. "the file
//    exists" is falsified if grep finds nothing).
// 4. When all hypotheses are falsified, forcing the model to generate new ones.
// 5. When one is verified, nudging the model to act on it.
//
// This turns the agent from a random guesser into a systematic investigator.

export type HypothesisStatus =
	| "active"
	| "falsified"
	| "verified"
	| "testing";

export interface Hypothesis {
	/** Unique ID. */
	id: string;
	/** The hypothesis statement — e.g. "The file is in a different directory than expected." */
	statement: string;
	/** How to test it — e.g. "grep for the file in the project root." */
	test: string;
	/** Current status. */
	status: HypothesisStatus;
	/** When created. */
	createdAt: number;
	/** When last tested. */
	lastTestedAt?: number;
	/** Test result (if tested). */
	testResult?: string;
	/** Evidence that supports or refutes this hypothesis. */
	evidence: string[];
	/** Confidence 0-100. */
	confidence: number;
}

export interface HypothesisTrackerConfig {
	/** Max hypotheses to track (default 10). */
	maxHypotheses?: number;
	/** Min hypotheses before generating new ones (default 3). */
	minHypothesesForGeneration?: number;
	/** Whether to auto-generate hypotheses when stuck (default true). */
	autoGenerate?: boolean;
}

const DEFAULT_CONFIG: Required<HypothesisTrackerConfig> = {
	maxHypotheses: 10,
	minHypothesesForGeneration: 3,
	autoGenerate: true,
};

// ── HypothesisTracker class ──────────────────────────────────────────────────

export class HypothesisTracker {
	private config: Required<HypothesisTrackerConfig>;
	private hypotheses: Hypothesis[] = [];
	private idCounter = 0;

	constructor(config: HypothesisTrackerConfig = {}) {
		this.config = { ...DEFAULT_CONFIG, ...config };
	}

	/**
	 * Add a new hypothesis.
	 */
	add(statement: string, test: string, confidence = 50): Hypothesis {
		const hypothesis: Hypothesis = {
			id: `hyp-${++this.idCounter}`,
			statement,
			test,
			status: "active",
			createdAt: Date.now(),
			confidence,
			evidence: [],
		};

		this.hypotheses.push(hypothesis);
		this.trimToLimit();

		return hypothesis;
	}

	/** Trim to maxHypotheses, preferring to drop the oldest falsified entries first. */
	private trimToLimit(): void {
		while (this.hypotheses.length > this.config.maxHypotheses) {
			const falsified = this.hypotheses
				.map((h, i) => ({ h, i }))
				.filter(({ h }) => h.status === "falsified")
				.sort((a, b) => a.h.createdAt - b.h.createdAt);

			if (falsified.length > 0) {
				this.hypotheses.splice(falsified[0].i, 1);
			} else {
				this.hypotheses = this.hypotheses.slice(-this.config.maxHypotheses);
				break;
			}
		}
	}

	/**
	 * Falsify a hypothesis (it was proven wrong).
	 */
	falsify(hypothesisId: string, testResult: string): boolean {
		const hypothesis = this.hypotheses.find((h) => h.id === hypothesisId);
		if (!hypothesis) return false;

		hypothesis.status = "falsified";
		hypothesis.testResult = testResult;
		hypothesis.lastTestedAt = Date.now();
		hypothesis.confidence = 0;
		hypothesis.evidence.push(testResult);
		return true;
	}

	/**
	 * Verify a hypothesis (it was proven right).
	 */
	verify(hypothesisId: string, testResult: string): boolean {
		const hypothesis = this.hypotheses.find((h) => h.id === hypothesisId);
		if (!hypothesis) return false;

		hypothesis.status = "verified";
		hypothesis.testResult = testResult;
		hypothesis.lastTestedAt = Date.now();
		hypothesis.confidence = 100;
		hypothesis.evidence.push(testResult);
		return true;
	}

	/**
	 * Mark a hypothesis as being tested.
	 */
	startTesting(hypothesisId: string): boolean {
		const hypothesis = this.hypotheses.find((h) => h.id === hypothesisId);
		if (!hypothesis) return false;
		hypothesis.status = "testing";
		return true;
	}

	/**
	 * Check if any hypothesis was falsified by recent evidence.
	 * Returns falsified hypothesis IDs.
	 */
	checkAgainstEvidence(
		evidence: Array<{ content: string; type?: string }>,
	): string[] {
		const falsified: string[] = [];

		for (const hyp of this.hypotheses) {
			if (hyp.status !== "active") continue;

			for (const ev of evidence) {
				const content = ev.content.toLowerCase();

				// If the hypothesis says "the file exists" and evidence says "file not found"
				if (
					/\b(file|path|directory|location)\b/.test(hyp.statement.toLowerCase()) &&
					/\b(not found|does not exist|missing|no such|cannot find)\b/.test(content)
				) {
					this.falsify(hyp.id, ev.content.slice(0, 200));
					falsified.push(hyp.id);
					break;
				}

				// If the hypothesis says "the test passes" and evidence says "test failed"
				if (
					/\b(test|verification)\b/.test(hyp.statement.toLowerCase()) &&
					/\b(fail|error|exception|not ok|exit code [1-9])\b/.test(content)
				) {
					this.falsify(hyp.id, ev.content.slice(0, 200));
					falsified.push(hyp.id);
					break;
				}

				// If the hypothesis says "the command works" and evidence shows an error
				if (
					/\b(command|tool|script)\b/.test(hyp.statement.toLowerCase()) &&
					/\b(error|failed|exception|traceback)\b/.test(content)
				) {
					this.falsify(hyp.id, ev.content.slice(0, 200));
					falsified.push(hyp.id);
					break;
				}
			}
		}

		return falsified;
	}

	/**
	 * Get hypotheses that need testing.
	 */
	getActiveHypotheses(): Hypothesis[] {
		return this.hypotheses.filter((h) => h.status === "active");
	}

	/**
	 * Get verified hypotheses (actionable).
	 */
	getVerifiedHypotheses(): Hypothesis[] {
		return this.hypotheses.filter((h) => h.status === "verified");
	}

	/**
	 * Check if all hypotheses are falsified (need new ones).
	 */
	areAllFalsified(): boolean {
		const active = this.getActiveHypotheses();
		return active.length === 0;
	}

	/**
	 * Build a prompt asking the model to generate hypotheses.
	 */
	buildHypothesisPrompt(stuckReasons: string[]): string {
		const existing = this.getActiveHypotheses();
		const existingText =
			existing.length > 0
				? `\n\nExisting hypotheses (still active):\n${existing
					.map((h) => `- ${h.statement} [confidence: ${h.confidence}%]`)
					.join("\n")}`
				: "";

		return (
			`[hypothesis-generation] You are stuck. Generate 3-5 hypotheses about WHY you're stuck. ` +
			`For each hypothesis, provide: (1) the statement, (2) a concrete test to verify/falsify it. ` +
			`Reasoning: ${stuckReasons.join("; ")}.${existingText}`
		);
	}

	/**
	 * Parse hypotheses from model output.
	 */
	parseFromText(text: string): Hypothesis[] {
		const lines = text.split("\n");
		const parsed: Hypothesis[] = [];
		let currentHypothesis: Partial<Hypothesis> | null = null;

		for (const line of lines) {
			// Match numbered hypothesis lines
			const match = line.match(/^\s*(?:\d+\.|\-|\*)\s+(.+)$/);
			if (match) {
				if (currentHypothesis) {
					parsed.push(currentHypothesis as Hypothesis);
				}
				const desc = match[1].trim();
				// Try to extract statement and test from the line
				const statementMatch = desc.match(
					/^(.+?)\b(?:because|since|as)\b(.+)$/i,
				);
				if (statementMatch) {
					currentHypothesis = {
						id: `hyp-${++this.idCounter}`,
						statement: statementMatch[1].trim(),
						test: statementMatch[2].trim(),
						status: "active",
						createdAt: Date.now(),
						confidence: 50,
						evidence: [],
					};
				} else {
					currentHypothesis = {
						id: `hyp-${++this.idCounter}`,
						statement: desc,
						test: "",
						status: "active",
						createdAt: Date.now(),
						confidence: 50,
						evidence: [],
					};
				}
			} else if (currentHypothesis && line.match(/^\s+test:/i)) {
				const test = line.replace(/^\s+test:\s*/i, "").trim();
				if (test) {
					currentHypothesis.test = test;
				}
			} else if (currentHypothesis && line.match(/^\s+confidence:/i)) {
				const conf = parseInt(
					line.replace(/^\s+confidence:\s*/i, "").trim(),
					10,
				);
				if (!isNaN(conf)) {
					currentHypothesis.confidence = Math.min(100, Math.max(0, conf));
				}
			}
		}

		if (currentHypothesis) {
			parsed.push(currentHypothesis as Hypothesis);
		}

		this.hypotheses.push(...parsed);
		this.trimToLimit();

		return parsed;
	}

	/**
	 * Get a status summary for the model.
	 */
	getStatusSummary(): string | null {
		if (this.hypotheses.length === 0) return null;

		const active = this.getActiveHypotheses();
		const verified = this.getVerifiedHypotheses();
		const falsified = this.hypotheses.filter(
			(h) => h.status === "falsified",
		);

		const lines = [`[hypotheses] ${active.length} active, ${verified.length} verified, ${falsified.length} falsified`];

		for (const hyp of active) {
			lines.push(`  ? ${hyp.statement} [${hyp.confidence}%]`);
		}
		for (const hyp of verified) {
			lines.push(`  ✓ ${hyp.statement} → ${hyp.testResult?.slice(0, 100)}`);
		}
		for (const hyp of falsified.slice(-3)) {
			lines.push(`  ✗ ${hyp.statement} → ${hyp.testResult?.slice(0, 100)}`);
		}

		return lines.join("\n");
	}

	/**
	 * Reset the tracker.
	 */
	reset(): void {
		this.hypotheses = [];
	}
}
