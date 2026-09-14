import { test } from "bun:test";
import assert from "node:assert/strict";
import { LoopDetector } from "../../control/guards/loop-detector.ts";

// ── Guard: duplicate call blocking ──────────────────────────────────────

void test("guard blocks on duplicate tool calls", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// 1st call — no block.
	assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	// 2nd call — no block.
	assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	// 3rd call — guard blocks.
	const decision = d.checkToolCall("read", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.ok(decision.message?.includes("3 times"));
});

void test("guard allows different args (interleaved resets counter)", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// 1st call — no block.
	assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	// Different tool resets the counter.
	assert.equal(d.checkToolCall("glob", '{"path":"/"}').block, false);
	// Same args as first — counter reset, starts at 1.
	assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	// Different args resets too.
	assert.equal(d.checkToolCall("read", '{"path":"b.txt"}').block, false);
	// Same as previous — counter at 1 again.
	assert.equal(d.checkToolCall("read", '{"path":"b.txt"}').block, false);
	// A third consecutive call of the same tool+args blocks.
	assert.equal(
		d.checkToolCall("read", '{"path":"b.txt"}').block,
		true,
		"consecutive 3rd call should block",
	);
});

// ── Guard: failure loop blocking ────────────────────────────────────────

void test("guard blocks on repeated failures same call", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
	});
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	// 4th call → guard blocks after 3 failures.
	assert.equal(
		d.checkToolCall("read", '{"path":"a.txt"}').block,
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
		d.recordFailure("read", p, "Error: not found");
	}
	// Same path failed 3 times → guard blocks on next call to that path.
	assert.equal(
		d.checkToolCall("read", paths[0] ?? "").block,
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
		d.recordFailure("read", paths[i] ?? "", results[i] ?? "");
	}
	// Same category (file not found) failed 3 times → guard blocks.
	assert.equal(
		d.checkToolCall("read", '{"path":"d.txt"}').block,
		true,
		"3 category failures should trigger guard block",
	);
});

void test("guard does not block on single failure", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
	});
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	// Should not be blocked with only 1 failure.
	assert.equal(
		d.checkToolCall("read", '{"path":"a.txt"}').block,
		false,
		"1 failure should not trigger guard block",
	);
});

void test("successful work clears stale failure state", () => {
	const d = new LoopDetector({ failureThreshold: 3 });
	const args = '{"path":"a.txt"}';
	for (let i = 0; i < 2; i++) {
		d.recordFailure("read", args, "Error: not found");
	}
	d.recordSuccess("read", args);
	d.recordFailure("read", args, "Error: not found");
	assert.equal(d.checkToolCall("read", args).block, false);
});

// ── Guard: disabled ───────────────────────────────────────────────────

void test("guard disabled when thresholds are 0", () => {
	const d = new LoopDetector({
		duplicateThreshold: 0,
		failureThreshold: 0,
	});
	// Even 10 identical calls should not be blocked (guard disabled).
	for (let i = 0; i < 10; i++) {
		assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	}
});

// ── Guard: checkToolCall standalone ─────────────────────────────────────

void test("checkToolCall returns block with message", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// First call — no block.
	assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	// Second call — no block.
	assert.equal(d.checkToolCall("read", '{"path":"a.txt"}').block, false);
	// Third call — blocked.
	const decision = d.checkToolCall("read", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.ok(decision.message?.includes("3 times"));
});

void test("recordFailure increments failure counts", () => {
	const d = new LoopDetector({ failureThreshold: 2 });
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read", '{"path":"b.txt"}', "Error: not found");
	// After 2 failures in same category, next check should block.
	assert.equal(
		d.checkToolCall("read", '{"path":"c.txt"}').block,
		true,
		"category failures should block",
	);
});

void test("guard returns correct guard type", () => {
	const d = new LoopDetector({ duplicateThreshold: 2 });
	d.checkToolCall("read", '{"path":"a.txt"}');
	const decision = d.checkToolCall("read", '{"path":"a.txt"}');
	assert.equal(decision.block, true);
	assert.equal(decision.guard, "duplicate");
});

void test("failure guard returns correct guard type", () => {
	const d = new LoopDetector({ failureThreshold: 2 });
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	const decision = d.checkToolCall("read", '{"path":"a.txt"}');
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
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	d.recordFailure("read", '{"path":"a.txt"}', "Error: not found");
	// After reset, failure state is cleared.
	d.reset();
	assert.equal(
		d.checkToolCall("read", '{"path":"a.txt"}').block,
		false,
		"reset should clear failure state",
	);
	// Also resets duplicate counter.
	d.checkToolCall("read", '{"path":"a.txt"}');
	d.reset();
	assert.equal(
		d.checkToolCall("read", '{"path":"a.txt"}').block,
		false,
		"reset should clear duplicate counter",
	);
});
// ── Batch loop detection (OMP-style, post-turn) ──────────────────────

void test("recordTurn detects identical batches across turns", () => {
	const d = new LoopDetector({ batchThreshold: 3 });
	const calls = [{ id: "1", name: "read", arguments: '{"path":"a.txt"}' }];
	const results = [{ id: "1", content: "file contents" }];

	// First two turns — no detection.
	assert.equal(d.recordTurn(calls, results), null);
	assert.equal(d.recordTurn(calls, results), null);

	// Third turn — detection!
	const detection = d.recordTurn(calls, results);
	assert.ok(detection);
	assert.equal(detection.kind, "repeated_tool_call");
	assert.equal(detection.toolName, "read");
	assert.equal(detection.count, 3);
	assert.ok(detection.resultSummary.includes("file contents"));
});

void test("recordTurn resets on different batch", () => {
	const d = new LoopDetector({ batchThreshold: 3 });
	const a = [{ id: "1", name: "read", arguments: '{"path":"a.txt"}' }];
	const b = [{ id: "1", name: "write", arguments: '{"path":"b.txt"}' }];
	const results = [{ id: "1", content: "ok" }];

	assert.equal(d.recordTurn(a, results), null);
	// Different batch resets counter.
	assert.equal(d.recordTurn(b, results), null);
	assert.equal(d.recordTurn(b, results), null);
	// Third identical to b — detection.
	const detection = d.recordTurn(b, results);
	assert.ok(detection);
	assert.equal(detection.toolName, "write");
});

void test("recordTurn resets on turn with no tool calls", () => {
	const d = new LoopDetector({ batchThreshold: 2 });
	const calls = [{ id: "1", name: "read", arguments: '{"path":"a.txt"}' }];
	const results = [{ id: "1", content: "ok" }];

	assert.equal(d.recordTurn(calls, results), null);
});
void test("recordTurn handles canonicalized key order", () => {
	const d = new LoopDetector({ batchThreshold: 3 });
	const calls1 = [
		{ id: "1", name: "bash", arguments: '{"cmd":"ls","path":"."}' },
	];
	const calls2 = [
		{ id: "1", name: "bash", arguments: '{"path":".","cmd":"ls"}' },
	];
	const results = [{ id: "1", content: "output" }];

	// Key order differs but content is same — should not detect.
	assert.equal(d.recordTurn(calls1, results), null);
	assert.equal(d.recordTurn(calls2, results), null);
	// Third identical → detection.
	assert.ok(d.recordTurn(calls2, results));
});

void test("recordTurn uses exemptTools", () => {
	const d = new LoopDetector({ batchThreshold: 2, exemptTools: ["hub"] });
	const calls = [{ id: "1", name: "hub", arguments: '{"action":"poll"}' }];
	const results = [{ id: "1", content: "no change" }];

	// Exempt tool — should reset every turn.
	assert.equal(d.recordTurn(calls, results), null);
	assert.equal(d.recordTurn(calls, results), null);
});

void test("recordTurn resets when every call is exempt", () => {
	const d = new LoopDetector({
		batchThreshold: 2,
		exemptTools: ["hub", "compaction"],
	});
	const calls = [
		{ id: "1", name: "hub", arguments: "{}" },
		{ id: "2", name: "compaction", arguments: "{}" },
	];
	const results = [
		{ id: "1", content: "ok" },
		{ id: "2", content: "ok" },
	];

	// All exempt — reset every time.
	assert.equal(d.recordTurn(calls, results), null);
	assert.equal(d.recordTurn(calls, results), null);
});

void test("recordTurn detects multi-call batches", () => {
	const d = new LoopDetector({ batchThreshold: 2 });
	const batch = [
		{ id: "1", name: "read", arguments: '{"path":"a.ts"}' },
		{ id: "2", name: "bash", arguments: '{"command":"echo hi"}' },
	];
	const results = [
		{ id: "1", content: "file contents" },
		{ id: "2", content: "hi" },
	];

	assert.equal(d.recordTurn(batch, results), null);
	const detection = d.recordTurn(batch, results);
	assert.ok(detection);
	assert.equal(detection.count, 2);
	// Reports first non-exempt call in original order.
	assert.equal(detection.toolName, "read");
});

void test("batch-loop guard blocks on checkToolCall after recordTurn", () => {
	const d = new LoopDetector({ batchThreshold: 2 });
	const calls = [{ id: "1", name: "read", arguments: '{"path":"a.txt"}' }];
	const results = [{ id: "1", content: "ok" }];

	// Two identical turns trigger detection.
	d.recordTurn(calls, results);
	d.recordTurn(calls, results);

	// Next checkToolCall for the same tool blocks with batch-loop.
	const decision = d.checkToolCall("read", '{"path":"a.txt"}');
	assert.ok(decision.block);
	assert.equal(decision.guard, "batch-loop");
});

void test("recordTurn resets on reset()", () => {
	const d = new LoopDetector({ batchThreshold: 2 });
	const calls = [{ id: "1", name: "read", arguments: '{"path":"a.txt"}' }];
	const results = [{ id: "1", content: "ok" }];

	d.recordTurn(calls, results);
	d.recordTurn(calls, results);

	d.reset();

	// After reset, same batch does not detect.
	assert.equal(d.recordTurn(calls, results), null);
});
