import { test } from "bun:test";
import assert from "node:assert/strict";
import { Transcript } from "../sessions/transcript.ts";

void test("events that do not change transcript state do not notify listeners", () => {
	const transcript = new Transcript();
	let notifications = 0;
	transcript.onChange(() => notifications++);

	transcript.handleEvent({
		type: "tool_call_update",
		toolCallId: "missing",
		delta: "{}",
	});

	assert.equal(notifications, 0);
});

void test("full message updates render non-streaming assistant responses", () => {
	const transcript = new Transcript();
	transcript.addTurn("hello");
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: { role: "assistant", content: "Hello back" },
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	assert.equal(
		chunks.map(chunk => chunk.contentText ?? "").join(""),
		"Hello back",
	);
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
	assert.equal(
		chunks.map(chunk => chunk.contentText ?? "").join(""),
		"Hello back",
	);
});

void test("completed streamed response remains committed when the next turn starts", () => {
	const transcript = new Transcript();
	const firstTurn = transcript.addTurn("first question");
	transcript.handleEvent({ type: "turn_start", turnId: "turn_1" });
	transcript.handleEvent({
		type: "token",
		token: "Persistent streamed answer",
	});
	transcript.handleEvent({
		type: "turn_end",
		turnId: "turn_1",
	});

	transcript.addTurn("second question");
	transcript.handleEvent({ type: "turn_start", turnId: "turn_2" });

	assert.equal(
		transcript.getAssistantContent(firstTurn),
		"Persistent streamed answer",
	);
	assert.equal(firstTurn.assistantMessage?.isComplete, true);
	assert.ok(
		firstTurn.assistantMessage?.chunks.every(chunk => chunk.isComplete),
	);
});

void test("empty structured-tool snapshot preserves streamed assistant prose", () => {
	const transcript = new Transcript();
	const firstTurn = transcript.addTurn("inspect the project");
	transcript.handleEvent({ type: "turn_start", turnId: "turn_1" });
	transcript.handleEvent({
		type: "token",
		token: "I found the relevant implementation.",
	});
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: {
			role: "assistant",
			content: "",
			tool_calls: [
				{
					id: "call_1",
					name: "read_file",
					arguments: '{"path":"implementation.ts"}',
				},
			],
		},
	});
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: "call_1",
		args: { path: "implementation.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call_1",
		result: "contents",
	});
	transcript.handleEvent({
		type: "turn_end",
		turnId: "turn_1",
	});
	transcript.addTurn("next question");

	assert.equal(
		transcript.getAssistantContent(firstTurn),
		"I found the relevant implementation.",
	);
});

void test("terminal snapshot restores output missed after a Skills notice", () => {
	const transcript = new Transcript();
	const turn = transcript.addTurn("list tools with ctx batch");
	transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "Skills",
		text: "context-mode · relevant to this request",
	});
	transcript.handleEvent({ type: "turn_start", turnId: "turn_1" });
	transcript.handleEvent({
		type: "turn_end",
		turnId: "turn_1",
		finalMessage: {
			role: "assistant",
			content: "ctx_batch_execute — run multiple commands in one call.",
		},
	});

	assert.equal(turn.id, "turn_1");
	assert.equal(turn.isComplete, true);
	assert.equal(turn.assistantMessage?.isComplete, true);
	assert.deepEqual(
		turn.assistantMessage?.chunks.map(chunk => ({
			type: chunk.type,
			text: chunk.contentText,
			label: chunk.notice?.label,
		})),
		[
			{ type: "notice", text: undefined, label: "Skills" },
			{
				type: "content",
				text: "ctx_batch_execute — run multiple commands in one call.",
				label: undefined,
			},
		],
	);
});

void test("promoted textual tool calls replace their streamed markup", () => {
	const transcript = new Transcript();
	transcript.addTurn("inspect the file");
	transcript.handleEvent({
		type: "token",
		token:
			"**<tool\\_call>**\n<function=read_file>raw markup</function>\n**</tool\\_call>**",
	});
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: {
			role: "assistant",
			content: "",
			tool_calls: [
				{
					id: "call_1",
					name: "read_file",
					arguments: '{"path":"file.ts"}',
				},
			],
		},
	});
	const chunks = transcript.getTurns().at(-1)?.assistantMessage?.chunks ?? [];
	assert.equal(chunks.map(chunk => chunk.contentText ?? "").join(""), "");
});

void test("message_update applies streamed tool-call argument snapshots to the matching chunk", () => {
	const transcript = new Transcript();
	transcript.addTurn("read a file");
	transcript.handleEvent({
		type: "tool_call_start",
		toolCallId: "call_1",
		toolName: "read_file",
		args: {},
	});
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: {
			role: "assistant",
			content: "",
			tool_calls: [
				{ id: "call_1", name: "read_file", arguments: '{"path":"a.ts"}' },
			],
		},
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	const toolChunk = chunks.find(chunk => chunk.type === "tool");
	assert.equal(toolChunk?.tool?.partialResult, '{"path":"a.ts"}');
});

void test("message_update does not overwrite a completed tool call's partialResult", () => {
	const transcript = new Transcript();
	transcript.addTurn("read a file");
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: "call_1",
		args: { path: "a.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call_1",
		result: "contents",
	});
	transcript.handleEvent({
		type: "message_update",
		turnId: "turn_1",
		message: {
			role: "assistant",
			content: "",
			tool_calls: [
				{ id: "call_1", name: "read_file", arguments: '{"path":"a.ts"}' },
			],
		},
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	const toolChunk = chunks.find(chunk => chunk.type === "tool");
	assert.equal(toolChunk?.tool?.partialResult, undefined);
	assert.equal(toolChunk?.isComplete, true);
});

void test("message_reasoning_update renders a coherent reasoning snapshot without duplicating streamed prefixes", () => {
	const transcript = new Transcript();
	transcript.addTurn("think about it");
	transcript.handleEvent({ type: "thinking_token", token: "Let me " });
	transcript.handleEvent({
		type: "message_reasoning_update",
		turnId: "turn_1",
		reasoning: "Let me consider the options",
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	const thinkingText = chunks
		.filter(chunk => chunk.type === "thinking")
		.map(chunk => chunk.contentText ?? "")
		.join("");
	assert.equal(thinkingText, "Let me consider the options");
});

void test("message_reasoning_update replaces a divergent streamed prefix instead of dropping the update", () => {
	const transcript = new Transcript();
	transcript.addTurn("think about it");
	// The raw delta stream and the snapshot channel can diverge (a provider
	// revising its own reasoning phrasing mid-stream). The snapshot must win
	// as a wholesale replacement rather than being silently dropped because it
	// doesn't share the streamed prefix — losing reasoning text is worse than
	// a non-incremental redraw.
	transcript.handleEvent({ type: "thinking_token", token: "The" });
	transcript.handleEvent({
		type: "message_reasoning_update",
		turnId: "turn_1",
		reasoning: "user said \"hi\" - I should respond warmly.",
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	const thinkingChunks = chunks.filter(chunk => chunk.type === "thinking");
	assert.equal(thinkingChunks.length, 1);
	assert.equal(
		thinkingChunks[0]?.contentText,
		'user said "hi" - I should respond warmly.',
	);
});

void test("message_reasoning_update only diffs against the latest contiguous thinking run", () => {
	const transcript = new Transcript();
	transcript.addTurn("investigate then answer");
	// First provider call's reasoning segment, closed out by a tool call.
	transcript.handleEvent({ type: "thinking_token", token: "Checking the file." });
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: "call_1",
		args: { path: "a.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call_1",
		result: "contents",
	});
	// Second provider call starts its own reasoning from empty — must not be
	// diffed against (or concatenated with) the first segment.
	transcript.handleEvent({ type: "thinking_token", token: "Now " });
	transcript.handleEvent({
		type: "message_reasoning_update",
		turnId: "turn_1",
		reasoning: "Now I have enough context to answer.",
	});
	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	const thinkingChunks = chunks.filter(chunk => chunk.type === "thinking");
	assert.equal(thinkingChunks.length, 2);
	assert.equal(thinkingChunks[0]?.contentText, "Checking the file.");
	assert.equal(thinkingChunks[1]?.contentText, "Now I have enough context to answer.");
});

void test("promoted tool calls preserve text and tool chronology across iterations", () => {
	const transcript = new Transcript();
	transcript.addTurn("inspect and continue");
	transcript.handleEvent({ type: "token", token: "I will inspect the file." });
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: "call_1",
		args: { path: "file.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call_1",
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
			tool_calls: [
				{
					id: "call_2",
					name: "read_file",
					arguments: '{"path":"other.ts"}',
				},
			],
		},
	});
	transcript.handleEvent({
		type: "tool_execution_start",
		toolName: "read_file",
		toolCallId: "call_2",
		args: { path: "other.ts" },
	});
	transcript.handleEvent({
		type: "tool_execution_end",
		toolName: "read_file",
		toolCallId: "call_2",
		result: "other contents",
	});

	const chunks = transcript.getTurns()[0].assistantMessage?.chunks ?? [];
	assert.deepEqual(
		chunks.map(chunk => ({
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
