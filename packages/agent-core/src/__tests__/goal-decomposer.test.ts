import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import {
	GoalDecomposer,
	parseSubgoalBreakdown,
} from "../agent/guards/goal-decomposer.ts";

describe("parseSubgoalBreakdown", () => {
	it("parses a numbered list into subgoals with defaults", () => {
		const text = "1. Read the test file\n2. Add a test case\n3. Run the suite";
		const breakdown = parseSubgoalBreakdown(text, "Add a test");
		assert.ok(breakdown);
		assert.strictEqual(breakdown?.subgoals.length, 3);
		assert.strictEqual(breakdown?.totalCount, 3);
		assert.strictEqual(breakdown?.completedCount, 0);
	});

	it("attaches verification and effort lines to the preceding subgoal", () => {
		const text = [
			"1. Add a test case",
			"   verification: npm test -- edge-case.spec.ts",
			"   effort: medium",
		].join("\n");
		const breakdown = parseSubgoalBreakdown(text, "obj");
		assert.strictEqual(breakdown?.subgoals[0].verificationCommand, "npm test -- edge-case.spec.ts");
		assert.strictEqual(breakdown?.subgoals[0].effort, "medium");
	});

	it("returns null when no subgoals are found", () => {
		const breakdown = parseSubgoalBreakdown("no list here", "obj");
		assert.strictEqual(breakdown, null);
	});

	it("estimates effort heuristically when not explicit", () => {
		const breakdown = parseSubgoalBreakdown("1. grep for the usage", "obj");
		assert.strictEqual(breakdown?.subgoals[0].effort, "low");
	});
});

describe("GoalDecomposer", () => {
	const makeDecomposer = (overrides = {}) =>
		new GoalDecomposer({ maxSubgoals: 10, ...overrides });

	// ── Lifecycle ────────────────────────────────────────────────────────────

	it("needsDecomposition() is true until a breakdown is set", () => {
		const decomposer = makeDecomposer();
		assert.strictEqual(decomposer.needsDecomposition(), true);
		decomposer.parseFromText("1. step one", "obj");
		assert.strictEqual(decomposer.needsDecomposition(), false);
	});

	it("parseFromText() auto-advances the first subgoal to in-progress", () => {
		// Regression: nothing calls markInProgress() externally, so without an
		// auto-transition, updateProgress() (which only checks in-progress
		// subgoals) could never mark anything complete.
		const decomposer = makeDecomposer();
		decomposer.parseFromText("1. Read the file\n2. Edit the file", "obj");
		const breakdown = decomposer.getBreakdown();
		assert.strictEqual(breakdown?.subgoals[0].status, "in-progress");
		assert.strictEqual(breakdown?.subgoals[1].status, "pending");
	});

	it("setBreakdown() also auto-advances the first pending subgoal", () => {
		const decomposer = makeDecomposer();
		decomposer.setBreakdown({
			objective: "obj",
			subgoals: [
				{ id: "sg-1", description: "first", status: "pending", effort: "low" },
				{ id: "sg-2", description: "second", status: "pending", effort: "low" },
			],
			completedCount: 0,
			totalCount: 2,
			completionPercentage: 0,
		});
		assert.strictEqual(decomposer.getBreakdown()?.subgoals[0].status, "in-progress");
	});

	// ── Progress updates ─────────────────────────────────────────────────────

	it("updateProgress() completes the in-progress subgoal and advances to the next", () => {
		const decomposer = makeDecomposer();
		decomposer.parseFromText("1. Edit src/foo.ts\n2. Run tests", "obj");
		const updated = decomposer.updateProgress([
			{ file: "src/foo.ts", tool: "edit_file", content: "edited" },
		]);
		assert.strictEqual(updated?.subgoals[0].status, "completed");
		assert.strictEqual(updated?.completedCount, 1);
		// The next pending subgoal should now be in-progress, ready for its own
		// completion check on a future call.
		assert.strictEqual(updated?.subgoals[1].status, "in-progress");
	});

	it("updateProgress() returns null when no breakdown exists", () => {
		const decomposer = makeDecomposer();
		assert.strictEqual(decomposer.updateProgress([]), null);
	});

	it("isComplete() is true only once every subgoal is completed", () => {
		const decomposer = makeDecomposer();
		decomposer.parseFromText("1. Add the new test case", "obj");
		assert.strictEqual(decomposer.isComplete(), false);
		decomposer.updateProgress([
			{ content: "task created and completed" },
		]);
		assert.strictEqual(decomposer.isComplete(), true);
	});

	// ── Blocked subgoals ─────────────────────────────────────────────────────

	it("markBlocked() records a blocker without completing the subgoal", () => {
		const decomposer = makeDecomposer();
		decomposer.parseFromText("1. Do the thing", "obj");
		const id = decomposer.getBreakdown()?.subgoals[0].id ?? "";
		const ok = decomposer.markBlocked(id, "missing permissions");
		assert.strictEqual(ok, true);
		assert.strictEqual(decomposer.getBreakdown()?.subgoals[0].status, "blocked");
		assert.strictEqual(decomposer.getBreakdown()?.subgoals[0].blocker, "missing permissions");
	});

	// ── Prompt / summary builders ────────────────────────────────────────────

	it("buildDecompositionPrompt() references the objective-independent template", () => {
		const decomposer = makeDecomposer();
		const prompt = decomposer.buildDecompositionPrompt("Add auth to the app");
		assert.match(prompt, /subgoals/i);
		assert.match(prompt, /effort/i);
	});

	it("getStatusSummary() returns null before decomposition", () => {
		const decomposer = makeDecomposer();
		assert.strictEqual(decomposer.getStatusSummary(), null);
	});

	it("getStatusSummary() reports completion counts and blockers", () => {
		const decomposer = makeDecomposer();
		decomposer.parseFromText("1. Step one\n2. Step two", "obj");
		const summary = decomposer.getStatusSummary();
		assert.match(summary ?? "", /0\/2 subgoals complete/);
	});

	// ── Reset ────────────────────────────────────────────────────────────────

	it("reset() clears the breakdown", () => {
		const decomposer = makeDecomposer();
		decomposer.parseFromText("1. Step", "obj");
		decomposer.reset();
		assert.strictEqual(decomposer.getBreakdown(), null);
		assert.strictEqual(decomposer.needsDecomposition(), true);
	});
});
