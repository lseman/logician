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
