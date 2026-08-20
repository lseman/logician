import { test } from "bun:test";
import assert from "node:assert/strict";
import { LoopDetector } from "../infrastructure/guards/loop-detector.ts";

// ── Guard: duplicate call blocking ──────────────────────────────────────

void test("guard blocks on duplicate tool calls", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// 1st call — no block.
	assert.equal(d.checkToolCall("read_file", '{"path":"a.txt"}').block, false);
	// 2nd call — no block.
	assert.equal(d.checkToolCall("read_file", '{"path":"a.txt"}').block, false);
	// 3rd call — guard blocks.
	const decision = d.checkToolCall("read_file", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.ok(decision.message?.includes("3 times"));
});

void test("guard allows different args (interleaved resets counter)", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// 1st call — no block.
	assert.equal(d.checkToolCall("read_file", '{"path":"a.txt"}').block, false);
	// Different tool resets the counter.
	assert.equal(d.checkToolCall("list_files", '{"path":"/"}').block, false);
	// Same args as first — counter reset, starts at 1.
	assert.equal(d.checkToolCall("read_file", '{"path":"a.txt"}').block, false);
	// Different args resets too.
	assert.equal(d.checkToolCall("read_file", '{"path":"b.txt"}').block, false);
	// Same as previous — counter at 1 again.
	assert.equal(d.checkToolCall("read_file", '{"path":"b.txt"}').block, false);
	// A third consecutive call of the same tool+args blocks.
	assert.equal(
		d.checkToolCall("read_file", '{"path":"b.txt"}').block,
		true,
		"consecutive 3rd call should block",
	);
});

// ── Guard: failure loop blocking ────────────────────────────────────────

void test("guard blocks on repeated failures same call", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
	});
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	// 4th call → guard blocks after 3 failures.
	assert.equal(
		d.checkToolCall("read_file", '{"path":"a.txt"}').block,
		true,
		"3 failures should trigger guard block",
	);
});

void test("guard blocks on repeated failures same path", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
	});
	const paths = ['{"path":"a.txt"}', '{"path":"b.txt"}', '{"path":"c.txt"}'];
	for (const p of paths) {
		d.recordFailure("read_file", p, "Error: not found");
	}
	// Same path failed 3 times → guard blocks on next call to that path.
	assert.equal(
		d.checkToolCall("read_file", paths[0]).block,
		true,
		"3 path failures should trigger guard block",
	);
});

void test("guard blocks on repeated failures same category", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
	});
	// Same error message across different paths → same failure category.
	const results = [
		"Error: file not found",
		"Error: file not found",
		"Error: file not found",
	];
	const paths = ['{"path":"a.txt"}', '{"path":"b.txt"}', '{"path":"c.txt"}'];
	for (let i = 0; i < 3; i++) {
		d.recordFailure("read_file", paths[i], results[i]);
	}
	// Same category (file not found) failed 3 times → guard blocks.
	assert.equal(
		d.checkToolCall("read_file", '{"path":"d.txt"}').block,
		true,
		"3 category failures should trigger guard block",
	);
});

void test("guard does not block on single failure", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
	});
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	// Should not be blocked with only 1 failure.
	assert.equal(
		d.checkToolCall("read_file", '{"path":"a.txt"}').block,
		false,
		"1 failure should not trigger guard block",
	);
});

void test("successful work clears stale failure state", () => {
	const d = new LoopDetector({ failureThreshold: 3 });
	const args = '{"path":"a.txt"}';
	for (let i = 0; i < 2; i++) {
		d.recordFailure("read_file", args, "Error: not found");
	}
	d.recordSuccess("read_file", args);
	d.recordFailure("read_file", args, "Error: not found");
	assert.equal(d.checkToolCall("read_file", args).block, false);
});

// ── Guard: disabled ───────────────────────────────────────────────────

void test("guard disabled when thresholds are 0", () => {
	const d = new LoopDetector({
		duplicateThreshold: 0,
		failureThreshold: 0,
	});
	// Even 10 identical calls should not be blocked (guard disabled).
	for (let i = 0; i < 10; i++) {
		assert.equal(
			d.checkToolCall("read_file", '{"path":"a.txt"}').block,
			false,
		);
	}
});

// ── Guard: checkToolCall standalone ─────────────────────────────────────

void test("checkToolCall returns block with message", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// First call — no block.
	assert.equal(d.checkToolCall("read_file", '{"path":"a.txt"}').block, false);
	// Second call — no block.
	assert.equal(d.checkToolCall("read_file", '{"path":"a.txt"}').block, false);
	// Third call — blocked.
	const decision = d.checkToolCall("read_file", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.ok(decision.message?.includes("3 times"));
});

void test("recordFailure increments failure counts", () => {
	const d = new LoopDetector({ failureThreshold: 2 });
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read_file", '{"path":"b.txt"}', "Error: not found");
	// After 2 failures in same category, next check should block.
	assert.equal(
		d.checkToolCall("read_file", '{"path":"c.txt"}').block,
		true,
		"category failures should block",
	);
});

void test("guard returns correct guard type", () => {
	const d = new LoopDetector({ duplicateThreshold: 2 });
	d.checkToolCall("read_file", '{"path":"a.txt"}');
	const decision = d.checkToolCall("read_file", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.equal(decision.guard, "duplicate");
});

void test("failure guard returns correct guard type", () => {
	const d = new LoopDetector({ failureThreshold: 2 });
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	const decision = d.checkToolCall("read_file", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.equal(decision.guard, "failure");
});

// ── Reset ──────────────────────────────────────────────────────────────

void test("reset clears all state", () => {
	const d = new LoopDetector({
		duplicateThreshold: 2,
		failureThreshold: 2,
	});
	// Build up failure state.
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read_file", '{"path":"a.txt"}', "Error: not found");
	// After reset, failure state is cleared.
	d.reset();
	assert.equal(
		d.checkToolCall("read_file", '{"path":"a.txt"}').block,
		false,
		"reset should clear failure state",
	);
	// Also resets duplicate counter.
	d.checkToolCall("read_file", '{"path":"a.txt"}');
	d.reset();
	assert.equal(
		d.checkToolCall("read_file", '{"path":"a.txt"}').block,
		false,
		"reset should clear duplicate counter",
	);
});
