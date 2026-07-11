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
