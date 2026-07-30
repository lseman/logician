import assert from "node:assert/strict";
import { test } from "node:test";
import { LoopDetector } from "../agent/guards/loop-detector.ts";

// Helper to create a signature the way the loop does.
function record(
	detector: LoopDetector,
	turns: Array<{
		content: string;
		toolCalls: Array<{ name: string; args: string; result: string }>;
	}>,
): boolean {
	for (const turn of turns) {
		if (detector.recordAndDetect(turn.content, turn.toolCalls)) {
			return true;
		}
	}
	return false;
}

function tool(
	name: string,
	args = "{}",
	result = "ok",
): { name: string; args: string; result: string } {
	return { name, args, result };
}

// ── Guard: duplicate call blocking ──────────────────────────────────────

void test("guard blocks on duplicate tool calls", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// Guard lives in the beforeToolCall hook — test it directly.
	// 1st call — no block.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"a.txt\"}").block, false);
	// 2nd call — no block.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"a.txt\"}").block, false);
	// 3rd call — guard blocks.
	const decision = d.checkToolCall("read_file", "{\"path\":\"a.txt\"}");
	assert.equal(decision.block, true);
	assert.ok(decision.message?.includes("3 times"));
});

void test("guard allows different args (interleaved resets counter)", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
		maxHistory: 10,
	});
	// 1st call — no block.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"a.txt\"}").block, false);
	// Different tool resets the counter.
	assert.equal(d.checkToolCall("list_files", "{\"path\":\"/\"}").block, false);
	// Same args as first — counter reset, starts at 1.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"a.txt\"}").block, false);
	// Different args resets too.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"b.txt\"}").block, false);
	// Same as previous — counter at 1 again.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"b.txt\"}").block, false);
	// A third consecutive call of the same tool+args blocks.
	assert.equal(
		d.checkToolCall("read_file", "{\"path\":\"b.txt\"}").block,
		true,
		"consecutive 3rd call should block",
	);
});

// ── Guard: failure loop blocking ────────────────────────────────────────

void test("guard blocks on repeated failures same call", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
		maxHistory: 10,
	});
	d.recordAndDetect("first", [tool("read_file", "{\"path\":\"a.txt\"}")]);
	d.recordFailure("read_file", "{\"path\":\"a.txt\"}", "Error: not found");
	d.recordAndDetect("second", [tool("read_file", "{\"path\":\"a.txt\"}")]);
	d.recordFailure("read_file", "{\"path\":\"a.txt\"}", "Error: not found");
	d.recordAndDetect("third", [tool("read_file", "{\"path\":\"a.txt\"}")]);
	d.recordFailure("read_file", "{\"path\":\"a.txt\"}", "Error: not found");
	// 4th call → guard blocks after 3 failures.
	d.recordAndDetect("fourth", [tool("read_file", "{\"path\":\"a.txt\"}")]);
	assert.equal(
		d.recordAndDetect("fifth", [tool("read_file", "{\"path\":\"a.txt\"}")]),
		true,
		"3 failures should trigger guard block",
	);
});

void test("guard blocks on repeated failures same path", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
		maxHistory: 10,
	});
	const paths = ["{\"path\":\"a.txt\"}", "{\"path\":\"b.txt\"}", "{\"path\":\"c.txt\"}"];
	for (const p of paths) {
		d.recordAndDetect("call", [tool("read_file", p)]);
		d.recordFailure("read_file", p, "Error: not found");
	}
	// Same path failed 3 times → guard blocks on next call.
	assert.equal(
		d.recordAndDetect("block", [tool("read_file", paths[0])]),
		true,
		"3 path failures should trigger guard block",
	);
});

void test("guard blocks on repeated failures same category", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
		maxHistory: 10,
	});
	const results = [
		"Error: file not found",
		"Error: No such file",
		"Error: file does not exist",
	];
	const paths = ["{\"path\":\"a.txt\"}", "{\"path\":\"b.txt\"}", "{\"path\":\"c.txt\"}"];
	for (let i = 0; i < 3; i++) {
		d.recordAndDetect("call", [tool("read_file", paths[i])]);
		d.recordFailure("read_file", paths[i], results[i]);
	}
	// Same category (file not found) failed 3 times → guard blocks.
	assert.equal(
		d.recordAndDetect("block", [tool("read_file", "{\"path\":\"d.txt\"}")]),
		true,
		"3 category failures should trigger guard block",
	);
});

void test("guard does not block on single failure", () => {
	const d = new LoopDetector({
		failureThreshold: 3,
		maxHistory: 10,
	});
	d.recordAndDetect("call", [tool("read_file", "{\"path\":\"a.txt\"}")]);
	d.recordFailure("read_file", "{\"path\":\"a.txt\"}", "Error: not found");
	// Should not be blocked with only 1 failure.
	assert.equal(
		d.recordAndDetect("call", [tool("read_file", "{\"path\":\"a.txt\"}")]),
		false,
		"1 failure should not trigger guard block",
	);
});

// ── Guard: disabled ───────────────────────────────────────────────────

void test("guard disabled when thresholds are 0", () => {
	const d = new LoopDetector({
		duplicateThreshold: 0,
		failureThreshold: 0,
		exactRepeatWindow: 20,   // disable turn-level detection
		degenerateWindow: 20,     // disable turn-level detection
		stagnationWindow: 20,     // disable turn-level detection
		maxHistory: 10,
	});
	// Even 10 identical calls should not be blocked (guard disabled, all turn
	// detection windows are high).
	for (let i = 0; i < 10; i++) {
		assert.equal(
			d.recordAndDetect(`call ${i}`, [tool("read_file", "{\"path\":\"a.txt\"}")]),
			false,
		);
	}
});

// ── Turn detection: exact repeat ───────────────────────────────────────

void test("detects exact repeat turns", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 3,
		maxHistory: 5,
		duplicateThreshold: 0, // disable guard for this test
	});
	const turn = {
		content: "I will read the file",
		toolCalls: [tool("read_file")],
	};
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false); // 1st
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false); // 2nd
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), true); // 3rd = loop
});

void test("does not flag single varying turns as exact repeat", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 3,
		maxHistory: 5,
		duplicateThreshold: 0, // disable guard for this test
	});
	d.recordAndDetect("read file", [
		tool("read_file", "{\"path\":\"a\"}", "content A"),
	]);
	d.recordAndDetect("read file", [
		tool("read_file", "{\"path\":\"b\"}", "content B"),
	]);
	assert.equal(
		d.recordAndDetect("read file", [
			tool("read_file", "{\"path\":\"a\"}", "content A"),
		]),
		false,
	);
});

// ── Turn detection: degenerate ──────────────────────────────────────────

void test("detects degenerate loops (same tools, same result shape, different args)", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 3,
		degenerateWindow: 4,
		maxHistory: 10,
	});
	const baseContent = "I will try reading the file";
	// Same tool, same result prefix (same kind of failure), different args.
	// This is the degenerate pattern: agent keeps retrying the same approach.
	const turns = [
		{
			content: baseContent,
			toolCalls: [
				tool("read_file", "{\"path\":\"a.txt\"}", "Error: file not found"),
			],
		},
		{
			content: baseContent,
			toolCalls: [
				tool("read_file", "{\"path\":\"b.txt\"}", "Error: file not found"),
			],
		},
		{
			content: baseContent,
			toolCalls: [
				tool("read_file", "{\"path\":\"c.txt\"}", "Error: file not found"),
			],
		},
		{
			content: baseContent,
			toolCalls: [
				tool("read_file", "{\"path\":\"d.txt\"}", "Error: file not found"),
			],
		},
	];
	assert.equal(record(d, turns), true, "should detect degenerate loop");
});

void test("does not flag varied tool sequences as degenerate", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10, // won't trigger
		degenerateWindow: 4,
		maxHistory: 10,
	});
	const turns = [
		{
			content: "trying approach 1",
			toolCalls: [
				tool("read_file", "{\"path\":\"a\"}"),
				tool("edit_file", "{\"path\":\"b\"}"),
			],
		},
		{
			content: "trying approach 2",
			toolCalls: [
				tool("edit_file", "{\"path\":\"c\"}"),
				tool("read_file", "{\"path\":\"d\"}"),
			],
		},
		{
			content: "trying approach 3",
			toolCalls: [
				tool("bash", "{\"cmd\":\"ls\"}"),
				tool("read_file", "{\"path\":\"e\"}"),
			],
		},
		{
			content: "trying approach 4",
			toolCalls: [
				tool("read_file", "{\"path\":\"f\"}"),
				tool("bash", "{\"cmd\":\"pwd\"}"),
			],
		},
	];
	assert.equal(
		record(d, turns),
		false,
		"varied sequences should not be flagged",
	);
});

// ── Turn detection: stagnation ──────────────────────────────────────────

void test("detects stagnation (zero new result prefixes)", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10, // won't trigger
		degenerateWindow: 10, // won't trigger
		stagnationWindow: 3,
		maxHistory: 10,
	});
	// Same result prefix each time — the detector tracks distinct (name:prefix) pairs.
	const turns = [
		{
			content: "reading",
			toolCalls: [
				tool("read_file", "{\"path\":\"a\"}", "Error: file not found"),
			],
		},
		{
			content: "reading again",
			toolCalls: [
				tool("read_file", "{\"path\":\"b\"}", "Error: file not found"),
			],
		},
		{
			content: "reading once more",
			toolCalls: [
				tool("read_file", "{\"path\":\"c\"}", "Error: file not found"),
			],
		},
	];
	assert.equal(record(d, turns), true, "should detect stagnation");
});

void test("does not flag when result prefixes vary", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10,
		degenerateWindow: 10,
		stagnationWindow: 3,
		maxHistory: 10,
	});
	const turns = [
		{
			content: "first",
			toolCalls: [
				tool("read_file", "{\"path\":\"a\"}", "Error: file not found"),
			],
		},
		{
			content: "second",
			toolCalls: [
				tool("read_file", "{\"path\":\"b\"}", "Error: permission denied"),
			],
		},
		{
			content: "third",
			toolCalls: [
				tool("read_file", "{\"path\":\"c\"}", "content found here"),
			],
		},
	];
	assert.equal(
		record(d, turns),
		false,
		"varying results should not be stagnation",
	);
});

void test("reset clears all state", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 2,
		maxHistory: 5,
		duplicateThreshold: 0, // disable guard for this test
	});
	d.recordAndDetect("same", [tool("read_file", "{\"path\":\"a\"}")]);
	d.recordAndDetect("same", [tool("read_file", "{\"path\":\"a\"}")]); // exact repeat
	d.reset();
	d.recordAndDetect("same", [tool("read_file", "{\"path\":\"a\"}")]); // should NOT be a loop after reset
});

void test("stagnation requires tool calls (no silent repetition)", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10,
		degenerateWindow: 10,
		stagnationWindow: 3,
		maxHistory: 10,
	});
	// No tool calls — stagnation should NOT trigger (handled by unproductive cap).
	const turns = [
		{ content: "thinking...", toolCalls: [] },
		{ content: "still thinking...", toolCalls: [] },
		{ content: "more thinking...", toolCalls: [] },
	];
	assert.equal(
		record(d, turns),
		false,
		"no-tool turns should not trigger stagnation",
	);
});

void test("degenerate requires args to vary (exact repeat caught separately)", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 3,
		degenerateWindow: 3,
		maxHistory: 10,
		duplicateThreshold: 0, // disable guard for this test
	});
	const turn = {
		content: "read",
		toolCalls: [tool("read_file", "{}", "ok")],
	};
	// 3 exact repeats → caught by exactRepeatWindow before degenerate
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false);
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false);
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), true);
});

// ── Guard: checkToolCall standalone ─────────────────────────────────────

void test("checkToolCall returns block with message", () => {
	const d = new LoopDetector({
		duplicateThreshold: 3,
	});
	// First call — no block.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"a.txt\"}").block, false);
	// Second call — no block.
	assert.equal(d.checkToolCall("read_file", "{\"path\":\"a.txt\"}").block, false);
	// Third call — blocked.
	const decision = d.checkToolCall("read_file", "{\"path\":\"a.txt\"}");
	assert.equal(decision.block, true);
	assert.ok(decision.message?.includes("3 times"));
});

void test("recordFailure increments failure counts", () => {
	const d = new LoopDetector({ failureThreshold: 2 });
	d.recordFailure("read_file", "{\"path\":\"a.txt\"}", "Error: not found");
	d.recordFailure("read_file", "{\"path\":\"b.txt\"}", "Error: not found");
	// After 2 failures in same category, next check should block.
	assert.equal(
		d.checkToolCall("read_file", "{\"path\":\"c.txt\"}").block,
		true,
		"category failures should block",
	);
});

// ── Diagnostics ─────────────────────────────────────────────────────────

void test("getLoopDiagnostic returns message for exact repeat", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 2,
		maxHistory: 5,
		duplicateThreshold: 0,
	});
	d.recordAndDetect("same content", [tool("read_file")]);
	d.recordAndDetect("same content", [tool("read_file")]);
	const diag = d.getLoopDiagnostic();
	assert.ok(diag?.startsWith("Exact repeat"));
});

void test("getLoopDiagnostic returns message for degenerate", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10,
		degenerateWindow: 3,
		maxHistory: 10,
		duplicateThreshold: 0,
	});
	const turns = [
		{ content: "read", toolCalls: [tool("read_file", "{\"p\":\"a\"}", "Error: not found")] },
		{ content: "read again", toolCalls: [tool("read_file", "{\"p\":\"b\"}", "Error: not found")] },
		{ content: "read again", toolCalls: [tool("read_file", "{\"p\":\"c\"}", "Error: not found")] },
	];
	record(d, turns);
	const diag = d.getLoopDiagnostic();
	assert.ok(diag?.startsWith("Degenerate"));
});

void test("getLoopDiagnostic returns message for stagnation", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10,
		degenerateWindow: 10,
		stagnationWindow: 3,
		maxHistory: 10,
		duplicateThreshold: 0,
	});
	const turns = [
		{ content: "read", toolCalls: [tool("read_file", "{\"p\":\"a\"}", "Error: not found")] },
		{ content: "read", toolCalls: [tool("read_file", "{\"p\":\"b\"}", "Error: not found")] },
		{ content: "read", toolCalls: [tool("read_file", "{\"p\":\"c\"}", "Error: not found")] },
	];
	record(d, turns);
	const diag = d.getLoopDiagnostic();
	assert.ok(diag?.startsWith("Stagnation"));
});
