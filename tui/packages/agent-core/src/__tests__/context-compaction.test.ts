import assert from "node:assert/strict";
import { test } from "node:test";
import {
	compactMessagesForContext,
	createAssistantMessage,
	createToolResultMessage,
	createUserMessage,
} from "../core/messages.ts";
import type { Message } from "../core/types.ts";

void test("context compaction progressively tightens until it meets the target", () => {
	const messages: Message[] = Array.from({ length: 20 }, (_, index) =>
		createUserMessage(`message ${index} ${"x".repeat(2000)}`),
	);
	const result = compactMessagesForContext(messages, { targetTokens: 1200 });
	assert.equal(result.changed, true);
	assert.ok(result.tokensAfter <= 1200, `${result.tokensAfter} should fit target`);
	assert.ok(
		result.messages.some((message) =>
			String(message.content).includes("context-compaction"),
		),
	);
});

void test("context compaction never leaves an orphaned tool result", () => {
	const call = { id: "call_1", name: "read_file", arguments: "{}" };
	const messages: Message[] = [
		...Array.from({ length: 8 }, (_, index) => createUserMessage(`old ${index}`)),
		createAssistantMessage("", [call]),
		createToolResultMessage(call.id, call.name, "result", false),
	];
	const result = compactMessagesForContext(messages, {
		targetTokens: 1000,
		keepRecentMessages: 1,
	});
	for (let index = 0; index < result.messages.length; index++) {
		const message = result.messages[index];
		if (message.role !== "tool") continue;
		const previous = result.messages[index - 1];
		assert.equal(previous?.role, "assistant");
		assert.ok(previous?.tool_calls?.some((toolCall) => toolCall.id === message.tool_call_id));
	}
});
