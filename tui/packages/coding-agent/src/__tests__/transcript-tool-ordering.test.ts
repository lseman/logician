import assert from "node:assert/strict";
import { test } from "node:test";
import { Transcript } from "../sessions/transcript.ts";

function start(transcript: Transcript, id: string, path: string): void {
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: id,
		tool_args: { path },
	});
}

void test("parallel same-name tool output stays attached to its call id", () => {
	const transcript = new Transcript();
	transcript.addTurn("Read both files");
	start(transcript, "call-a", "a.ts");
	start(transcript, "call-b", "b.ts");

	transcript.handleEvent({
		type: "tool_execution_update",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call-a",
		update_kind: "output",
		partial_result: "A progress",
	});
	transcript.handleEvent({
		type: "tool_execution_update",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call-b",
		update_kind: "output",
		partial_result: "B progress",
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call-b",
		result: "B result",
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call-a",
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

void test("ambiguous id-less updates never overwrite a parallel tool", () => {
	const transcript = new Transcript();
	transcript.addTurn("Read both files");
	start(transcript, "call-a", "a.ts");
	start(transcript, "call-b", "b.ts");

	transcript.handleEvent({
		type: "tool_execution_update",
		tool: "read_file",
		tool_name: "read_file",
		partial_result: "ambiguous",
		update_kind: "output",
	});

	const tools = transcript.getAssistantTools(transcript.getTurns()[0]);
	assert.ok(tools.every(tool => tool.streamOutput === undefined));
});
