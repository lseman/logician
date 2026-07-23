import assert from "node:assert/strict";
import { test } from "node:test";
import { Transcript } from "../sessions/transcript.ts";

void test("subagent tool notices become one integrated lifecycle entry", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: '▶ call-1 read_file path=src/index.ts',
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ call-1 read_file 120 lines",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	assert.ok(assistant);
	assert.equal(
		assistant.chunks.some((chunk) => chunk.type === "notice"),
		false,
		"child activity should not duplicate into top-level notices",
	);
	const calls = assistant.chunks.find(
		(chunk) => chunk.tool?.tool_name === "spawn_agent",
	)?.tool?.details?.childToolCalls as Array<Record<string, unknown>>;
	assert.deepEqual(calls, [
		{
			agentId: "explorer-1",
			toolCallId: "call-1",
			toolName: "read_file",
			args: "path=src/index.ts",
			status: "completed",
			isError: false,
			resultPreview: "120 lines",
		},
	]);
});

void test("concurrent same-name child tool calls resolve to the correct call by id", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	// Two concurrent calls to the same tool name, interleaved arrival: start A,
	// start B, end B, end A — a name-only match would wrongly resolve "end B"
	// against the first running "read_file" (call A) instead of call B.
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: "▶ call-A read_file path=a.ts",
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "↳ explorer-1",
		text: "▶ call-B read_file path=b.ts",
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ call-B read_file 40 lines",
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ call-A read_file 120 lines",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	const calls = assistant?.chunks.find(
		(chunk) => chunk.tool?.tool_name === "spawn_agent",
	)?.tool?.details?.childToolCalls as Array<Record<string, unknown>>;

	const callA = calls.find((c) => c.toolCallId === "call-A");
	const callB = calls.find((c) => c.toolCallId === "call-B");
	assert.equal(callA?.status, "completed");
	assert.equal(callA?.resultPreview, "120 lines");
	assert.equal(callB?.status, "completed");
	assert.equal(callB?.resultPreview, "40 lines");
});

void test("subagent lifecycle notices update the parent card without duplication", () => {
	const transcript = new Transcript();
	transcript.addTurn("Delegate this task");
	transcript.handleEvent({ type: "turn_start", turn_id: "turn-1" });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "spawn_agent",
		tool_name: "spawn_agent",
		tool_call_id: "parent-tool",
		tool_args: { agent: "explorer", task: "Inspect the workspace" },
	});
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "Subagent explorer",
		text: "started: Inspect the workspace",
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "Subagent explorer",
		text: "done in 2 turns",
	});

	const assistant = transcript.getTurns()[0]?.assistantMessage;
	assert.ok(assistant);
	assert.equal(
		assistant.chunks.some((chunk) => chunk.type === "notice"),
		false,
	);
	const parent = assistant.chunks.find(
		(chunk) => chunk.tool?.tool_call_id === "parent-tool",
	)?.tool;
	assert.equal(parent?.details?.status, "completed");
	assert.equal(parent?.details?.lifecycleSummary, "done in 2 turns");
});
