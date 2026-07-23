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
		text: '▶ read_file path=src/index.ts',
	});
	transcript.handleEvent({
		type: "notice",
		level: "success",
		label: "↳ explorer-1",
		text: "✓ read_file 120 lines",
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
			toolName: "read_file",
			args: "path=src/index.ts",
			status: "completed",
			isError: false,
			resultPreview: "120 lines",
		},
	]);
});
