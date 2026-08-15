import { test } from "bun:test";
import assert from "node:assert/strict";
import { normalizeProviderMessages, OpenAIBackend } from "../agent/backend.ts";

void test("leading system message is preserved", () => {
	const out = normalizeProviderMessages([
		{ role: "system", content: "base" },
		{ role: "user", content: "hi" },
		{ role: "assistant", content: "yo" },
	]);
	assert.deepEqual(out, [
		{ role: "system", content: "base" },
		{ role: "user", content: "hi" },
		{ role: "assistant", content: "yo" },
	]);
});

void test("trailing system messages are re-roled to user, leading kept", () => {
	const out = normalizeProviderMessages([
		{ role: "system", content: "base" },
		{ role: "user", content: "hi" },
		{ role: "assistant", content: "yo" },
		{ role: "system", content: "# Agent Context\n..." },
		{ role: "system", content: "<task_state>phase: verify</task_state>" },
	]);
	assert.equal(out[0].role, "system");
	assert.equal(out[0].content, "base");
	assert.equal(out[3].role, "user");
	assert.equal(out[3].content, "# Agent Context\n...");
	assert.equal(out[4].role, "user");
	assert.equal(out[4].content, "<task_state>phase: verify</task_state>");
});

void test("without a leading system, any later system becomes user", () => {
	const out = normalizeProviderMessages([
		{ role: "user", content: "hi" },
		{ role: "system", content: "late" },
	]);
	assert.equal(out[0].role, "user");
	assert.equal(out[1].role, "user");
	assert.equal(out[1].content, "late");
});

void test("messages without system roles are unchanged", () => {
	const input = [
		{ role: "user", content: "hi" },
		{ role: "assistant", content: "yo", tool_calls: [] },
		{ role: "tool", content: "result", tool_call_id: "t1" },
	];
	assert.deepEqual(normalizeProviderMessages(input), input);
});

void test("generate sends trailing system context as a user message", async () => {
	const sse = [
		'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}',
		"",
		"data: [DONE]",
		"",
	].join("\n");
	const sentBodies: Record<string, unknown>[] = [];
	const originalFetch = globalThis.fetch;
	globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
		sentBodies.push(JSON.parse(String(init?.body ?? "{}")));
		return new Response(sse, {
			status: 200,
			headers: { "Content-Type": "text/event-stream" },
		});
	}) as typeof fetch;
	try {
		const backend = new OpenAIBackend({
			baseUrl: "http://test.local",
			model: "test-model",
		});
		const response = await backend.generate(
			[
				{ role: "system", content: "base prompt" },
				{ role: "user", content: "do the thing" },
				{ role: "assistant", content: "done" },
				{ role: "system", content: "<task_state>phase: verify</task_state>" },
			],
			{ maxRetries: 0 },
		);
		assert.equal(response.content, "ok");
		assert.equal(sentBodies.length, 1);
		const sent = sentBodies[0].messages as Record<string, unknown>[];
		assert.equal(sent[0].role, "system");
		assert.equal(sent[0].content, "base prompt");
		assert.equal(sent[3].role, "user");
		assert.equal(sent[3].content, "<task_state>phase: verify</task_state>");
	} finally {
		globalThis.fetch = originalFetch;
	}
});
