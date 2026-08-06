import assert from "node:assert/strict";
import { test } from "node:test";
import { Transcript } from "../sessions/transcript.ts";

// Reproduces the reported "stuck streaming spinner" bug end to end, using the
// exact event shapes the bridge's mapAgentEvent() produces for each stage:
// a streaming placeholder start (from agent-loop-runner's onToolCallStart,
// mapped tool_call_start -> "tool_execution_start"), then the tool-batch-
// controller "length" branch's end (mapped tool_call_end -> "tool_execution_end").

void test("a length-truncated tool call closes its spinner instead of hanging", () => {
	const transcript = new Transcript();
	transcript.addTurn("write a big file");

	// Streaming placeholder: model already sent the real id (typical case).
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "call_abc123",
		tool_args: {},
	});

	// tool-batch-controller's rawStopReason === "length" branch emits
	// tool_call_end (mapped to "tool_execution_end") with the same id.
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "call_abc123",
		result:
			'Tool call "write_file" was not executed because the assistant response hit the output token limit...',
		is_error: true,
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	assert.ok(tool, "tool chunk should exist");
	assert.equal(tool.isComplete, true, "spinner must close, not hang forever");
	assert.equal(tool.isError, true);
});

void test("a length-truncated tool call closes even with a placeholder streaming id", () => {
	const transcript = new Transcript();
	transcript.addTurn("write a big file");

	// Streaming placeholder: provider hasn't sent a real id yet (backend.ts
	// falls back to `tool_${index}` — here "tool_0").
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "tool_0",
		tool_args: {},
	});

	// By the time the length branch fires, the response is fully assembled
	// and the real id is known — it no longer matches the placeholder.
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "call_real_id_9",
		result: "not executed, truncated",
		is_error: true,
	});

	const tool = transcript.getAssistantTools(transcript.getTurns()[0])[0];
	assert.ok(tool, "tool chunk should exist");
	assert.equal(
		tool.isComplete,
		true,
		"spinner must close even when the streaming placeholder id differs from the final call id",
	);
});

void test("two parallel same-name calls truncated together both close correctly", () => {
	const transcript = new Transcript();
	transcript.addTurn("write two big files");

	// Two write_file calls stream in parallel, each with a placeholder id
	// (provider hasn't assigned real ids yet for either).
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "tool_0",
		tool_args: {},
	});
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "tool_1",
		tool_args: {},
	});

	// Both truncate together; the length branch emits real ids for each.
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "call_real_a",
		result: "not executed, truncated",
		is_error: true,
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "write_file",
		tool_name: "write_file",
		tool_call_id: "call_real_b",
		result: "not executed, truncated",
		is_error: true,
	});

	const tools = transcript.getAssistantTools(transcript.getTurns()[0]);
	const stillOpen = tools.filter(tool => !tool.isComplete);
	assert.equal(stillOpen.length, 0, "no spinner should be left stuck");
});
