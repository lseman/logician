import { test } from "bun:test";
import assert from "node:assert/strict";
import { Transcript } from "../../application/transcript/transcript.ts";

function start(transcript: Transcript, id: string, path: string): void {
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: id,
		args: { path },
	});
}

void test("parallel same-name tool output stays attached to its call id", () => {
	const transcript = new Transcript();
	transcript.addTurn("Read both files");
	start(transcript, "call-a", "a.ts");
	start(transcript, "call-b", "b.ts");

	transcript.handleEvent({
		type: "tool_execution_update",
		toolName: "read_file",
		toolCallId: "call-a",
		partialResult: "A progress",
	});
	transcript.handleEvent({
		type: "tool_execution_update",
		toolName: "read_file",
		toolCallId: "call-b",
		partialResult: "B progress",
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call-b",
		result: "B result",
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call-a",
		result: "A result",
	});

	const tools = transcript.getAssistantTools(transcript.getTurns()[0]);
	assert.deepEqual(
		tools.map(tool => ({
			id: tool.tool_call_id,
			path: tool.args?.path,
			result: tool.result,
			complete: tool.isComplete,
		})),
		[
			{
				id: "call-a",
				path: "a.ts",
				result: "A result",
				complete: true,
			},
			{
				id: "call-b",
				path: "b.ts",
				result: "B result",
				complete: true,
			},
		],
	);
});

void test("execution start enriches the card created during call preparation", () => {
	const transcript = new Transcript();
	transcript.addTurn("Read the file");
	transcript.handleEvent({
		type: "tool_call_start",
		toolName: "read_file",
		toolCallId: "call-a",
		args: {},
	});

	transcript.handleEvent({
		type: "tool_call_update",
		toolCallId: "call-a",
		delta: '{"path":"a.ts"}',
	});
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: "call-a",
		args: { path: "a.ts" },
	});

	const tools = transcript.getAssistantTools(transcript.getTurns()[0]);
	assert.equal(tools.length, 1);
	assert.equal(tools[0].tool_call_id, "call-a");
	assert.deepEqual(tools[0].args, { path: "a.ts" });
});
