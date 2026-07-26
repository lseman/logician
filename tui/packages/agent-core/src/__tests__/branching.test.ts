// ── Branching enhancement tests ──────────────────────────────────────────────

import { describe, it } from "node:test";
import { strict as assert } from "node:assert";

describe("BranchInfo type", () => {
	it("has all required fields", () => {
		const branch: { id: string; depth: number; summary: unknown; forkedAt: number } = {
			id: "branch_1",
			depth: 1,
			summary: null,
			forkedAt: 5,
		};
		assert.strictEqual(branch.id, "branch_1");
		assert.strictEqual(branch.depth, 1);
		assert.strictEqual(branch.summary, null);
		assert.strictEqual(branch.forkedAt, 5);
	});
});

describe("BranchSummaryData structure", () => {
	it("has all sections", () => {
		const summary = {
			goal: "Fix login bug",
			constraints: ["Must support OAuth", "No external deps"],
			progress: {
				done: ["Set up project"],
				inProgress: ["Add tests"],
				blocked: ["Waiting on API key"],
			},
			keyDecisions: [
				{ decision: "Use SQLite", rationale: "Faster queries" },
			],
			nextSteps: ["Run tests", "Deploy"],
			full: "Full summary text",
		};

		assert.strictEqual(summary.goal, "Fix login bug");
		assert.strictEqual(summary.constraints.length, 2);
		assert.strictEqual(summary.progress.done.length, 1);
		assert.strictEqual(summary.progress.inProgress.length, 1);
		assert.strictEqual(summary.progress.blocked.length, 1);
		assert.strictEqual(summary.keyDecisions.length, 1);
		assert.strictEqual(summary.nextSteps.length, 2);
		assert.strictEqual(summary.full, "Full summary text");
	});

	it("handles empty sections", () => {
		const summary = {
			goal: "Simple branch",
			constraints: [],
			progress: { done: [], inProgress: [], blocked: [] },
			keyDecisions: [],
			nextSteps: [],
			full: "No detailed work",
		};

		assert.strictEqual(summary.constraints.length, 0);
		assert.strictEqual(summary.progress.done.length, 0);
		assert.strictEqual(summary.progress.inProgress.length, 0);
		assert.strictEqual(summary.progress.blocked.length, 0);
		assert.strictEqual(summary.keyDecisions.length, 0);
		assert.strictEqual(summary.nextSteps.length, 0);
	});
});

describe("BranchProgress", () => {
	it("tracks done items", () => {
		const progress = {
			done: ["Item 1", "Item 2", "Item 3"],
			inProgress: [],
			blocked: [],
		};
		assert.strictEqual(progress.done.length, 3);
	});

	it("tracks in-progress items", () => {
		const progress = {
			done: [],
			inProgress: ["Partially complete"],
			blocked: [],
		};
		assert.strictEqual(progress.inProgress.length, 1);
	});

	it("tracks blocked items", () => {
		const progress = {
			done: [],
			inProgress: [],
			blocked: ["Depends on external API"],
		};
		assert.strictEqual(progress.blocked.length, 1);
	});
});

describe("BranchSummaryData with file ops", () => {
	it("includes file info in full summary", () => {
		const full = "Branch conversation\n\nRead files: src/main.ts, config.json\nModified files: src/main.ts";
		const summary = {
			goal: "Refactor main module",
			constraints: [],
			progress: { done: [], inProgress: [], blocked: [] },
			keyDecisions: [],
			nextSteps: [],
			full,
		};

		assert.ok(summary.full.includes("Read files:"));
		assert.ok(summary.full.includes("Modified files:"));
		assert.ok(summary.full.includes("src/main.ts"));
	});
});
