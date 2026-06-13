import assert from "node:assert/strict";
import { test } from "node:test";
import { LoopDetector } from "../loop-detector.ts";

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

function tool(name: string, args = "{}", result = "ok"): { name: string; args: string; result: string } {
	return { name, args, result };
}

void test("detects exact repeat turns", () => {
	const d = new LoopDetector({ exactRepeatWindow: 3, maxHistory: 5 });
	const turn = { content: "I will read the file", toolCalls: [tool("read_file")] };
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false); // 1st
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false); // 2nd
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), true); // 3rd = loop
});

void test("does not flag single varying turns as exact repeat", () => {
	const d = new LoopDetector({ exactRepeatWindow: 3, maxHistory: 5 });
	d.recordAndDetect("read file", [tool("read_file", "{}", "content A")]);
	d.recordAndDetect("read file", [tool("read_file", "{}", "content B")]); // result differs
	assert.equal(d.recordAndDetect("read file", [tool("read_file", "{}", "content A")]), false);
});

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
		{ content: baseContent, toolCalls: [tool("read_file", '{"path":"a.txt"}', "Error: file not found")] },
		{ content: baseContent, toolCalls: [tool("read_file", '{"path":"b.txt"}', "Error: file not found")] },
		{ content: baseContent, toolCalls: [tool("read_file", '{"path":"c.txt"}', "Error: file not found")] },
		{ content: baseContent, toolCalls: [tool("read_file", '{"path":"d.txt"}', "Error: file not found")] },
	];
	assert.equal(record(d, turns), true, "should detect degenerate loop");
});

void test("does not flag varied tool sequences as degenerate", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 3,
		degenerateWindow: 4,
		maxHistory: 10,
	});
	const turns = [
		{ content: "trying approach 1", toolCalls: [tool("read_file"), tool("edit_file")] },
		{ content: "trying approach 2", toolCalls: [tool("edit_file"), tool("read_file")] },
		{ content: "trying approach 3", toolCalls: [tool("bash"), tool("read_file")] },
		{ content: "trying approach 4", toolCalls: [tool("read_file"), tool("bash")] },
	];
	assert.equal(record(d, turns), false, "varied sequences should not be flagged");
});

void test("detects stagnation (zero new result prefixes)", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 10, // won't trigger
		degenerateWindow: 10, // won't trigger
		stagnationWindow: 3,
		maxHistory: 10,
	});
	// Same result prefix each time — the detector tracks distinct (name:prefix) pairs.
	const turns = [
		{ content: "reading", toolCalls: [tool("read_file", '{"path":"a"}', "Error: file not found")] },
		{ content: "reading again", toolCalls: [tool("read_file", '{"path":"b"}', "Error: file not found")] },
		{ content: "reading once more", toolCalls: [tool("read_file", '{"path":"c"}', "Error: file not found")] },
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
		{ content: "first", toolCalls: [tool("read_file", '{"path":"a"}', "Error: file not found")] },
		{ content: "second", toolCalls: [tool("read_file", '{"path":"b"}', "Error: permission denied")] },
		{ content: "third", toolCalls: [tool("read_file", '{"path":"c"}', "content found here")] },
	];
	assert.equal(record(d, turns), false, "varying results should not be stagnation");
});

void test("reset clears all state", () => {
	const d = new LoopDetector({ exactRepeatWindow: 2, maxHistory: 5 });
	d.recordAndDetect("same", [tool("read_file")]);
	d.recordAndDetect("same", [tool("read_file")]); // exact repeat
	d.reset();
	d.recordAndDetect("same", [tool("read_file")]); // should NOT be a loop after reset
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
	assert.equal(record(d, turns), false, "no-tool turns should not trigger stagnation");
});

void test("degenerate requires args to vary (exact repeat caught separately)", () => {
	const d = new LoopDetector({
		exactRepeatWindow: 3,
		degenerateWindow: 3,
		maxHistory: 10,
	});
	const turn = { content: "read", toolCalls: [tool("read_file", "{}", "ok")] };
	// 3 exact repeats → caught by exactRepeatWindow before degenerate
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false);
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), false);
	assert.equal(d.recordAndDetect(turn.content, turn.toolCalls), true);
});
