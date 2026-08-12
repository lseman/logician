import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	createAssistantMessage,
	createToolResultMessage,
	createUserMessage,
	estimateChatPayloadTokens,
} from "../agent/messages.ts";
import type { CompactableMessage } from "../agent/types.ts";
import { compactToFit } from "../compaction/compaction.ts";
import { serializeConversation } from "../compaction/utils.ts";

void test("conversation serialization tolerates null and malformed content", () => {
	assert.equal(
		serializeConversation([
			{ role: "user", content: null },
			{ role: "assistant", content: null },
			{ role: "assistant", content: [null, { type: "text", text: "kept" }] },
			{ role: "tool_result", content: undefined },
		]),
		"[Assistant]: kept",
	);
});

void test("context estimates include tool definition overhead", () => {
	const messages = [createUserMessage("hello")];
	const withoutTools = estimateChatPayloadTokens(messages);
	const withTools = estimateChatPayloadTokens(messages, [
		{
			type: "function",
			function: {
				name: "search",
				description: "Search project files",
				parameters: {
					type: "object",
					properties: { query: { type: "string" } },
				},
			},
		},
	]);

	assert.ok(withTools > withoutTools);
});

void test("context compaction progressively tightens until it meets the target", async () => {
	const messages: CompactableMessage[] = Array.from(
		{ length: 20 },
		(_, index) => createUserMessage(`message ${index} ${"x".repeat(2000)}`),
	);
	const result = await compactToFit(messages, {
		triggerTokens: 0,
		targetTokens: 1200,
	});
	assert.equal(result.changed, true);
	assert.ok(
		result.tokensAfter <= 1200,
		`${result.tokensAfter} should fit target`,
	);
	assert.ok(
		result.messages.some(message => message.role === "compactionSummary"),
	);
});

void test("context compaction never leaves an orphaned tool result", async () => {
	const call = { id: "call_1", name: "read_file", arguments: "{}" };
	const messages: CompactableMessage[] = [
		...Array.from({ length: 8 }, (_, index) =>
			createUserMessage(`old ${index}`),
		),
		createAssistantMessage("", [call]),
		createToolResultMessage(call.id, call.name, "result", false),
	];
	const result = await compactToFit(messages, {
		triggerTokens: 0,
		targetTokens: 1000,
		keepRecentMessages: 1,
	});
	for (let index = 0; index < result.messages.length; index++) {
		const message = result.messages[index] as {
			role: string;
			tool_call_id?: string;
		};
		if (message.role !== "tool") continue;
		const previous = result.messages[index - 1] as {
			role: string;
			tool_calls?: Array<{ id: string }>;
		};
		assert.equal(previous?.role, "assistant");
		assert.ok(
			previous?.tool_calls?.some(
				toolCall => toolCall.id === message.tool_call_id,
			),
		);
	}
});
