import assert from "node:assert/strict";
import { test } from "node:test";
import { Transcript } from "../sessions/transcript.ts";

void test("full message updates render non-streaming assistant responses", () => {
	const transcript = new Transcript();
	transcript.addTurn("hello");
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: { role: "assistant", content: "Hello back" },
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	assert.equal(chunks.map((chunk) => chunk.contentText ?? "").join(""), "Hello back");
});

void test("full message updates do not duplicate streamed prefixes", () => {
	const transcript = new Transcript();
	transcript.addTurn("hello");
	transcript.handleEvent({ type: "token", token: "Hello" });
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: { role: "assistant", content: "Hello back" },
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	assert.equal(chunks.map((chunk) => chunk.contentText ?? "").join(""), "Hello back");
});

void test("promoted textual tool calls replace their streamed markup", () => {
	const transcript = new Transcript();
	transcript.addTurn("inspect the file");
	transcript.handleEvent({
		type: "token",
		token: "**<tool\\_call>**\n<function=read_file>raw markup</function>\n**</tool\\_call>**",
	});
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: {
			role: "assistant",
			content: "",
			tool_calls: [{
				id: "call_1",
				name: "read_file",
				arguments: "{\"path\":\"file.ts\"}",
			}],
		},
	});
	const chunks = transcript.getTurns().at(-1)?.assistantMessage?.chunks ?? [];
	assert.equal(chunks.map((chunk) => chunk.contentText ?? "").join(""), "");
});

void test("promoted tool calls preserve text and tool chronology across iterations", () => {
	const transcript = new Transcript();
	transcript.addTurn("inspect and continue");
	transcript.handleEvent({ type: "token", token: "I will inspect the file." });
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call_1",
		tool_args: { path: "file.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call_1",
		result: "file contents",
	});
	transcript.handleEvent({
		type: "token",
		token: "I found another file.\n<tool_call>raw markup</tool_call>",
	});
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: {
			role: "assistant",
			content: "I found another file.",
			tool_calls: [{
				id: "call_2",
				name: "read_file",
				arguments: "{\"path\":\"other.ts\"}",
			}],
		},
	});
	transcript.handleEvent({
		type: "tool_execution_start",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call_2",
		tool_args: { path: "other.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		tool: "read_file",
		tool_name: "read_file",
		tool_call_id: "call_2",
		result: "other contents",
	});

	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	assert.deepEqual(
		chunks.map((chunk) => ({
			type: chunk.type,
			text: chunk.contentText,
			tool: chunk.tool?.tool_name,
		})),
		[
			{ type: "content", text: "I will inspect the file.", tool: undefined },
			{ type: "tool", text: undefined, tool: "read_file" },
			{ type: "content", text: "I found another file.", tool: undefined },
			{ type: "tool", text: undefined, tool: "read_file" },
		],
	);
});
