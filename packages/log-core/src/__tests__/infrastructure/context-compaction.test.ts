import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	createAssistantMessage,
	createToolResultMessage,
	createUserMessage,
	estimateChatPayloadTokens,
} from "../../capabilities/provider/messages.ts";
import {
	compactToFit,
	pruneHistoricalToolOutputs,
} from "../../runtime/compaction/engine.ts";
import { serializeConversation } from "../../runtime/compaction/serialization.ts";
import type { CompactableMessage } from "../../system/types/types-messages.ts";

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

void test("pruneHistoricalToolOutputs trims old verbose tool results while preserving recent turns", () => {
	const largeLog = Array.from(
		{ length: 50 },
		(_, i) => `log line ${i}: build output detail`,
	).join("\n");

	const messages: CompactableMessage[] = [
		createUserMessage("First turn: build project"),
		createAssistantMessage("", [
			{ id: "call_1", name: "bash", arguments: "{}" },
		]),
		createToolResultMessage("call_1", "bash", largeLog, false),
		createAssistantMessage("Build completed, now running tests."),
		createUserMessage("Second turn: fix tests"),
		createAssistantMessage("", [
			{ id: "call_2", name: "bash", arguments: "{}" },
		]),
		createToolResultMessage("call_2", "bash", "test result: passed", false),
		createAssistantMessage("All tests passing!"),
	];

	// pruneHistoricalToolOutputs with keepRecentTurns: 1
	const pruned = pruneHistoricalToolOutputs(messages, {
		keepRecentTurns: 1,
		maxHistoricalChars: 200,
		headLines: 3,
		tailLines: 3,
	});

	assert.equal(pruned.changed, true);
	assert.equal(pruned.prunedCount, 1);
	assert.ok(pruned.charactersSaved > 500);

	// The old tool result (call_1) should be pruned
	const oldResult = pruned.messages[2] as { content: string };
	assert.match(oldResult.content, /historical output trimmed/);
	assert.match(oldResult.content, /log line 0/);
	assert.match(oldResult.content, /log line 49/);

	// The recent tool result (call_2) should be completely untouched
	const recentResult = pruned.messages[6] as { content: string };
	assert.equal(recentResult.content, "test result: passed");
});
