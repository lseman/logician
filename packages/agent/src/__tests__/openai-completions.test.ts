import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	normalizeProviderMessages,
	streamOpenAiCompletions,
} from "../ai/openai-completions.ts";
import type { AssistantMessageEvent, Context, Model } from "../ai/types.ts";

const MODEL: Model = {
	id: "test-model",
	name: "Test Model",
	api: "openai-completions",
	provider: "openai-compatible",
	baseUrl: "https://example.invalid",
	reasoning: false,
	contextWindow: 128_000,
	maxTokens: 4096,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

function sseResponse(lines: string[]): Response {
	const body = `${lines.map(line => `data: ${line}\n\n`).join("")}data: [DONE]\n\n`;
	return new Response(body, {
		status: 200,
		headers: { "Content-Type": "text/event-stream" },
	});
}

async function collect(
	stream: AsyncIterable<AssistantMessageEvent>,
): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

void test("streams plain text and emits a done event", async () => {
	const chunks = [
		JSON.stringify({ choices: [{ delta: { content: "Hel" } }] }),
		JSON.stringify({ choices: [{ delta: { content: "lo" } }] }),
		JSON.stringify({
			choices: [{ delta: {}, finish_reason: "stop" }],
			usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
		}),
	];
	const fetchImpl = async () => sseResponse(chunks);

	const context: Context = {
		messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
	};
	const events = await collect(
		streamOpenAiCompletions(MODEL, context, {
			fetch: fetchImpl as unknown as typeof fetch,
		}),
	);

	assert.equal(events[0]?.type, "start");
	assert.equal(
		events.some(e => e.type === "text_delta" && e.delta === "Hel"),
		true,
	);
	assert.equal(
		events.some(e => e.type === "text_delta" && e.delta === "lo"),
		true,
	);

	const done = events.find(e => e.type === "done");
	assert.ok(done && done.type === "done");
	assert.equal(done.reason, "stop");
	assert.equal(
		done.message.content.some(c => c.type === "text" && c.text === "Hello"),
		true,
	);
	assert.equal(done.message.usage.input, 10);
	assert.equal(done.message.usage.output, 2);
});

void test("accumulates a streamed tool call and reports toolUse", async () => {
	const chunks = [
		JSON.stringify({
			choices: [
				{
					delta: {
						tool_calls: [
							{
								index: 0,
								id: "call_1",
								function: { name: "read_file", arguments: "" },
							},
						],
					},
				},
			],
		}),
		JSON.stringify({
			choices: [
				{
					delta: {
						tool_calls: [{ index: 0, function: { arguments: '{"path":' } }],
					},
				},
			],
		}),
		JSON.stringify({
			choices: [
				{
					delta: {
						tool_calls: [{ index: 0, function: { arguments: '"a.txt"}' } }],
					},
				},
			],
		}),
		JSON.stringify({ choices: [{ delta: {}, finish_reason: "tool_calls" }] }),
	];
	const fetchImpl = async () => sseResponse(chunks);

	const context: Context = {
		messages: [{ role: "user", content: "read a.txt", timestamp: Date.now() }],
	};
	const events = await collect(
		streamOpenAiCompletions(MODEL, context, {
			fetch: fetchImpl as unknown as typeof fetch,
		}),
	);

	const toolCallEnd = events.find(e => e.type === "toolcall_end");
	assert.ok(toolCallEnd && toolCallEnd.type === "toolcall_end");
	assert.equal(toolCallEnd.toolCall.name, "read_file");
	assert.deepEqual(toolCallEnd.toolCall.arguments, { path: "a.txt" });

	const done = events.find(e => e.type === "done");
	assert.ok(done && done.type === "done");
	assert.equal(done.reason, "toolUse");
});

void test("maps a non-2xx response to an error event", async () => {
	const fetchImpl = async () =>
		new Response("context length exceeded", { status: 400 });
	const context: Context = {
		messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
	};
	const events = await collect(
		streamOpenAiCompletions(MODEL, context, {
			fetch: fetchImpl as unknown as typeof fetch,
		}),
	);

	assert.equal(events.length, 2); // start, error
	const errorEvent = events[1];
	assert.ok(errorEvent && errorEvent.type === "error");
	assert.equal(errorEvent.reason, "error");
	assert.match(errorEvent.error.errorMessage ?? "", /context length exceeded/);
});

void test("normalizeProviderMessages re-roles trailing system messages to user", () => {
	const out = normalizeProviderMessages([
		{ role: "system", content: "base" },
		{ role: "user", content: "hi" },
		{ role: "system", content: "trailing" },
	]);
	assert.equal(out[0]?.role, "system");
	assert.equal(out[2]?.role, "user");
	assert.equal(out[2]?.content, "trailing");
});
