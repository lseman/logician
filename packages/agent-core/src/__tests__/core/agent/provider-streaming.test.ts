import { test } from "bun:test";
import assert from "node:assert/strict";
import { buildStreamingCallbacks } from "../../../core/loop/provider-streaming.ts";
import type { AgentEvent } from "../../../core/types/types-messages.ts";

void test("onSnapshot does not emit message_update for a pure-reasoning snapshot", () => {
	const events: AgentEvent[] = [];
	const callbacks = buildStreamingCallbacks("turn_1", event =>
		events.push(event),
	);

	callbacks.onSnapshot?.({
		content: "",
		reasoning: "thinking so far",
		toolCalls: [],
	});

	assert.equal(
		events.some(event => event.type === "message_update"),
		false,
	);
	assert.ok(events.some(event => event.type === "message_reasoning_update"));
});

void test("onSnapshot emits message_update once real content is present", () => {
	const events: AgentEvent[] = [];
	const callbacks = buildStreamingCallbacks("turn_1", event =>
		events.push(event),
	);

	callbacks.onSnapshot?.({ content: "Hello", reasoning: "", toolCalls: [] });

	const update = events.find(event => event.type === "message_update");
	assert.ok(update && update.type === "message_update");
	assert.equal(update.message.content, "Hello");
});

void test("onSnapshot emits message_update once a tool call is present, even with empty content", () => {
	const events: AgentEvent[] = [];
	const callbacks = buildStreamingCallbacks("turn_1", event =>
		events.push(event),
	);

	callbacks.onSnapshot?.({
		content: "",
		reasoning: "",
		toolCalls: [
			{ id: "call_1", name: "read_file", arguments: '{"path":"a.ts"}' },
		],
	});

	const update = events.find(event => event.type === "message_update");
	assert.ok(update && update.type === "message_update");
	assert.equal(update.message.tool_calls?.[0]?.name, "read_file");
});
