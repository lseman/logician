import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import { HypothesisTracker } from "../agent/guards/hypothesis-tracker.ts";

describe("HypothesisTracker", () => {
	const makeTracker = (overrides = {}) =>
		new HypothesisTracker({ maxHypotheses: 10, ...overrides });

	// ── add / falsify / verify ───────────────────────────────────────────────

	it("adds a hypothesis as active with the given confidence", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The file is in a different directory", "grep for it", 60);
		assert.strictEqual(hyp.status, "active");
		assert.strictEqual(hyp.confidence, 60);
		assert.strictEqual(tracker.getActiveHypotheses().length, 1);
	});

	it("falsify() marks a hypothesis falsified with 0 confidence", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The test passes", "run npm test");
		const ok = tracker.falsify(hyp.id, "npm test failed with 3 errors");
		assert.strictEqual(ok, true);
		assert.strictEqual(tracker.getActiveHypotheses().length, 0);
		assert.strictEqual(tracker.getVerifiedHypotheses().length, 0);
	});

	it("verify() marks a hypothesis verified with 100 confidence", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The bug is in auth.ts", "read auth.ts");
		const ok = tracker.verify(hyp.id, "confirmed: null check missing");
		assert.strictEqual(ok, true);
		assert.strictEqual(tracker.getVerifiedHypotheses().length, 1);
		assert.strictEqual(tracker.getVerifiedHypotheses()[0].confidence, 100);
	});

	it("falsify()/verify() return false for an unknown id", () => {
		const tracker = makeTracker();
		assert.strictEqual(tracker.falsify("does-not-exist", "n/a"), false);
		assert.strictEqual(tracker.verify("does-not-exist", "n/a"), false);
	});

	it("areAllFalsified() is true only when every hypothesis is falsified", () => {
		const tracker = makeTracker();
		const h1 = tracker.add("A", "test A");
		const h2 = tracker.add("B", "test B");
		assert.strictEqual(tracker.areAllFalsified(), false);
		tracker.falsify(h1.id, "wrong");
		assert.strictEqual(tracker.areAllFalsified(), false);
		tracker.falsify(h2.id, "wrong too");
		assert.strictEqual(tracker.areAllFalsified(), true);
	});

	// ── checkAgainstEvidence ─────────────────────────────────────────────────

	it("falsifies a file-existence hypothesis when evidence says the file is missing", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The config file exists in the project root", "ls the root");
		const falsified = tracker.checkAgainstEvidence([
			{ content: "Error: file does not exist at that path" },
		]);
		assert.deepStrictEqual(falsified, [hyp.id]);
		assert.strictEqual(tracker.getActiveHypotheses().length, 0);
	});

	it("falsifies a test-passing hypothesis when evidence shows a failure", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The test suite passes", "run the test command");
		const falsified = tracker.checkAgainstEvidence([
			{ content: "3 tests failed with exit code 1" },
		]);
		assert.deepStrictEqual(falsified, [hyp.id]);
	});

	it("leaves unrelated hypotheses untouched", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The user prefers dark mode", "check settings");
		const falsified = tracker.checkAgainstEvidence([
			{ content: "Error: file not found" },
		]);
		assert.deepStrictEqual(falsified, []);
		assert.strictEqual(tracker.getActiveHypotheses()[0].id, hyp.id);
	});

	it("does not re-check already-resolved hypotheses", () => {
		const tracker = makeTracker();
		const hyp = tracker.add("The file exists", "ls it");
		tracker.verify(hyp.id, "confirmed present");
		const falsified = tracker.checkAgainstEvidence([
			{ content: "file not found" },
		]);
		assert.deepStrictEqual(falsified, []);
		assert.strictEqual(tracker.getVerifiedHypotheses().length, 1);
	});

	// ── parseFromText ────────────────────────────────────────────────────────

	it("parseFromText() stores parsed hypotheses, not just returns them", () => {
		// Regression: parseFromText used to build and return a list without
		// ever pushing it into the tracker's internal state.
		const tracker = makeTracker();
		const text = "1. The path is wrong because the config was moved\n   test: grep for the new path";
		const parsed = tracker.parseFromText(text);
		assert.ok(parsed.length >= 1);
		assert.strictEqual(tracker.getActiveHypotheses().length, parsed.length);
	});

	it("parseFromText() extracts statement/test split on 'because'", () => {
		const tracker = makeTracker();
		const parsed = tracker.parseFromText(
			"1. Tests fail because the mock is stale",
		);
		assert.strictEqual(parsed.length, 1);
		assert.match(parsed[0].statement, /tests fail/i);
		assert.match(parsed[0].test, /mock is stale/i);
	});

	it("parseFromText() picks up explicit test: and confidence: lines", () => {
		const tracker = makeTracker();
		const text = [
			"1. The API key is missing",
			"   test: check the .env file",
			"   confidence: 70",
		].join("\n");
		const parsed = tracker.parseFromText(text);
		assert.strictEqual(parsed.length, 1);
		assert.strictEqual(parsed[0].test, "check the .env file");
		assert.strictEqual(parsed[0].confidence, 70);
	});

	it("parseFromText() respects maxHypotheses trimming", () => {
		const tracker = makeTracker({ maxHypotheses: 2 });
		tracker.add("existing 1", "t1");
		tracker.add("existing 2", "t2");
		const text = "1. new hypothesis\n2. another new hypothesis";
		tracker.parseFromText(text);
		assert.ok(tracker["hypotheses"].length <= 2);
	});

	// ── Prompt / summary builders ────────────────────────────────────────────

	it("buildHypothesisPrompt() includes stuck reasons and existing hypotheses", () => {
		const tracker = makeTracker();
		tracker.add("The file moved", "grep for it", 40);
		const prompt = tracker.buildHypothesisPrompt(["progress is low"]);
		assert.match(prompt, /progress is low/);
		assert.match(prompt, /The file moved/);
	});

	it("getStatusSummary() returns null when no hypotheses exist", () => {
		const tracker = makeTracker();
		assert.strictEqual(tracker.getStatusSummary(), null);
	});

	it("getStatusSummary() reports active/verified/falsified counts", () => {
		const tracker = makeTracker();
		const h1 = tracker.add("A", "test A");
		const h2 = tracker.add("B", "test B");
		tracker.verify(h1.id, "confirmed");
		tracker.falsify(h2.id, "denied");
		const summary = tracker.getStatusSummary();
		assert.match(summary ?? "", /0 active, 1 verified, 1 falsified/);
	});

	// ── Reset ────────────────────────────────────────────────────────────────

	it("reset() clears all hypotheses", () => {
		const tracker = makeTracker();
		tracker.add("A", "test A");
		tracker.reset();
		assert.strictEqual(tracker.getActiveHypotheses().length, 0);
		assert.strictEqual(tracker.getStatusSummary(), null);
	});
});
