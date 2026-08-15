import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import { ProgressSignalTracker } from "../agent/guards/progress-signal.ts";

describe("ProgressSignalTracker", () => {
	const makeTracker = (overrides = {}) =>
		new ProgressSignalTracker({
			minScoreBeforeNudge: 30,
			minLowScoreTurns: 3,
			...overrides,
		});

	// ── Phase scoring ────────────────────────────────────────────────────────

	it("scores 'orient' phase at 0", () => {
		const tracker = makeTracker();
		const signal = tracker.record([], [], 1, "orient");
		assert.strictEqual(signal.breakdown.phase.score, 0);
		assert.strictEqual(signal.breakdown.phase.expected, "investigate");
	});

	it("scores 'handoff' phase at 100", () => {
		const tracker = makeTracker();
		const signal = tracker.record([], [], 1, "handoff");
		assert.strictEqual(signal.breakdown.phase.score, 100);
	});

	it("scores 'blocked' phase at the floor, expecting a return to investigate", () => {
		const tracker = makeTracker();
		const signal = tracker.record([], [], 1, "blocked");
		assert.strictEqual(signal.breakdown.phase.score, 0);
		assert.strictEqual(signal.breakdown.phase.current, "blocked");
		assert.strictEqual(signal.breakdown.phase.expected, "investigate");
	});

	it("does not misreport 'blocked' as expecting 'orient'", () => {
		// Regression: PHASE_ORDER.indexOf("blocked") used to be -1, which the
		// old code silently treated as index 0 ("orient"), producing a
		// nonsensical "expected: orient" message for a blocked run.
		const tracker = makeTracker();
		const signal = tracker.record([], [], 1, "blocked");
		assert.notStrictEqual(signal.breakdown.phase.expected, "orient");
	});

	// ── File-level progress ──────────────────────────────────────────────────

	it("scores meaningful file changes higher than plain reads", () => {
		const tracker = makeTracker();
		const signal = tracker.record(
			[{ name: "edit_file", args: JSON.stringify({ path: "src/foo.ts" }) }],
			[{ content: "function foo() { return 1; }", file: "src/foo.ts" }],
			1,
			"implement",
		);
		assert.ok(signal.breakdown.files.meaningful >= 1);
		assert.ok(signal.breakdown.files.score > 0);
	});

	// ── Verification progress ────────────────────────────────────────────────

	it("scores passing verification higher than failing verification", () => {
		const tracker = makeTracker();
		const signal = tracker.record(
			[{ name: "bash", args: "{}" }],
			[{ content: "5/5 tests pass, all clean" }],
			1,
			"verify",
		);
		assert.strictEqual(signal.breakdown.verification.passing, 1);
		assert.strictEqual(signal.breakdown.verification.score, 100);
	});

	it("scores failing verification at 0", () => {
		const tracker = makeTracker();
		const signal = tracker.record(
			[{ name: "bash", args: "{}" }],
			[{ content: "1 failed, Error: assertion failed" }],
			1,
			"verify",
		);
		assert.strictEqual(signal.breakdown.verification.score, 0);
	});

	// ── Nudge triggering ──────────────────────────────────────────────────────

	it("does not nudge before minLowScoreTurns is reached", () => {
		const tracker = makeTracker({ minLowScoreTurns: 3 });
		for (let i = 0; i < 2; i++) {
			const signal = tracker.record([], [], i, "orient");
			assert.strictEqual(signal.shouldNudge, false);
		}
	});

	it("nudges once minLowScoreTurns of low scores accumulate", () => {
		const tracker = makeTracker({ minLowScoreTurns: 3, minScoreBeforeNudge: 30 });
		let lastSignal;
		for (let i = 0; i < 3; i++) {
			lastSignal = tracker.record([], [], i, "orient");
		}
		assert.strictEqual(lastSignal?.shouldNudge, true);
		assert.ok(lastSignal?.nudgeMessage);
	});

	it("resets the low-score streak after a high-scoring turn", () => {
		const tracker = makeTracker({ minLowScoreTurns: 3, minScoreBeforeNudge: 30 });
		tracker.record([], [], 0, "orient");
		tracker.record([], [], 1, "orient");
		// A turn that scores well (handoff phase) should reset the streak.
		tracker.record([], [], 2, "handoff");
		const signal = tracker.record([], [], 3, "orient");
		assert.strictEqual(signal.shouldNudge, false);
	});

	it("shouldNudge() reflects the tracker's internal state after record()", () => {
		const tracker = makeTracker({ minLowScoreTurns: 2, minScoreBeforeNudge: 30 });
		tracker.record([], [], 0, "orient");
		tracker.record([], [], 1, "orient");
		assert.strictEqual(tracker.shouldNudge(), true);
	});

	// ── Reset ────────────────────────────────────────────────────────────────

	it("reset() clears observations and low-score streak", () => {
		const tracker = makeTracker({ minLowScoreTurns: 2 });
		tracker.record([], [], 0, "orient");
		tracker.record([], [], 1, "orient");
		tracker.reset();
		assert.strictEqual(tracker.shouldNudge(), false);
		assert.deepStrictEqual(tracker.getObservations(), []);
	});
});
