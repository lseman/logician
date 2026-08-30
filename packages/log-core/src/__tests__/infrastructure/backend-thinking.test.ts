import { test } from "bun:test";
import assert from "node:assert/strict";
import { OpenAIBackend } from "../../capabilities/provider/backend.ts";

const SSE = [
	'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}',
	"",
	"data: [DONE]",
	"",
].join("\n");

/** Run one generate() call and return the JSON body sent to the server. */
async function captureRequestBody(
	backend: OpenAIBackend,
	options?: { thinkingLevel?: "off" | "low" | "medium" | "high" | "xhigh" },
): Promise<Record<string, unknown>> {
	let captured: Record<string, unknown> | undefined;
	const originalFetch = globalThis.fetch;
	globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
		captured = JSON.parse(String(init?.body ?? "{}"));
		return new Response(SSE, {
			status: 200,
			headers: { "Content-Type": "text/event-stream" },
		});
	}) as typeof fetch;
	try {
		await backend.generate([{ role: "user", content: "hi" }], options);
	} finally {
		globalThis.fetch = originalFetch;
	}
	assert.ok(captured, "expected a request to be sent");
	return captured!;
}

void test("qwen format sends explicit enable_thinking:false when thinking is off", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "qwen3",
		thinkingFormat: "qwen",
	});
	const body = await captureRequestBody(backend, { thinkingLevel: "off" });
	assert.equal(body.enable_thinking, false);
	assert.equal("reasoning_effort" in body, false);
	assert.equal("reasoning" in body, false);
});

void test("qwen format enables thinking and sets effort when level is on", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "qwen3",
		thinkingFormat: "qwen",
	});
	const body = await captureRequestBody(backend, { thinkingLevel: "high" });
	assert.equal(body.enable_thinking, true);
	assert.equal(body.reasoning_effort, "high");
});

void test("qwen-chat-template format uses chat_template_kwargs", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "qwen3",
		thinkingFormat: "qwen-chat-template",
	});
	const body = await captureRequestBody(backend, { thinkingLevel: "off" });
	assert.deepEqual(body.chat_template_kwargs, {
		enable_thinking: false,
		preserve_thinking: true,
	});
	assert.equal("enable_thinking" in body, false);
});

void test("default format sends top-level reasoning_effort when on", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "gpt5",
	});
	const body = await captureRequestBody(backend, { thinkingLevel: "medium" });
	assert.equal(body.reasoning_effort, "medium");
	assert.equal("reasoning" in body, false);
	assert.equal("enable_thinking" in body, false);
});

void test("default format omits all reasoning fields when off", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "gpt5",
	});
	const body = await captureRequestBody(backend, { thinkingLevel: "off" });
	assert.equal("reasoning_effort" in body, false);
	assert.equal("reasoning" in body, false);
	assert.equal("enable_thinking" in body, false);
});

void test("per-call thinkingLevel overrides the backend default", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "qwen3",
		thinkingFormat: "qwen",
		thinkingLevel: "high",
	});
	// No per-call level → default "high" applies.
	const on = await captureRequestBody(backend);
	assert.equal(on.enable_thinking, true);
	// Explicit off wins over the default.
	const off = await captureRequestBody(backend, { thinkingLevel: "off" });
	assert.equal(off.enable_thinking, false);
});

void test("withModel and withEndpoint preserve thinkingFormat", async () => {
	const backend = new OpenAIBackend({
		baseUrl: "http://test.local",
		model: "qwen3",
		thinkingFormat: "qwen",
	});
	for (const clone of [backend.withModel("other"), backend.withEndpoint("m2", "http://x")]) {
		const body = await captureRequestBody(
			clone as OpenAIBackend,
			{ thinkingLevel: "off" },
		);
		assert.equal(body.enable_thinking, false, "clone must keep qwen format");
	}
});
