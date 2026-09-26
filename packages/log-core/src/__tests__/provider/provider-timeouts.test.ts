// Regression tests for the provider timeout guards (P0.3): a wedged local
// server (llama.cpp OOM, vLLM deadlock) must abort the turn with a retryable
// BackendError instead of hanging until a manual abort. Uses real local HTTP
// servers because the guards live in OpenAIBackend's fetch/stream layer.
//
// Timer note (ts-no-test-timers exception): the guards under test ARE real
// timers in production code; the tests await the guard's rejection (the real
// signal) rather than sleeping, and the wedged streams below stay open
// indefinitely via a pull source that never enqueues again — no test-side
// wall-clock waits.

import { test } from "bun:test";
import assert from "node:assert/strict";
import { BackendError, OpenAIBackend } from "../../provider/backend.ts";

function sseResponse(chunks: string[]): Response {
	const encoder = new TextEncoder();
	return new Response(
		new ReadableStream({
			start(controller) {
				for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
				controller.close();
			},
		}),
		{ headers: { "Content-Type": "text/event-stream" } },
	);
}

// A stream that delivers exactly one SSE chunk, then stays open forever
// (pull is called again, enqueues nothing, resolves) — the shape of a server
// that wedged mid-generation.
function wedgedSseResponse(firstChunk: string): Response {
	const encoder = new TextEncoder();
	let delivered = false;
	return new Response(
		new ReadableStream({
			pull(controller) {
				if (!delivered) {
					delivered = true;
					controller.enqueue(encoder.encode(firstChunk));
				}
				// Never enqueue again: the reader blocks until the guard fires.
			},
		}),
		{ headers: { "Content-Type": "text/event-stream" } },
	);
}

void test("initial-response timeout aborts a server that never answers", async () => {
	const server = Bun.serve({
		port: 0,
		fetch: () => new Promise<Response>(() => {}), // accepts, never responds
	});
	try {
		const backend = new OpenAIBackend({
			baseUrl: `http://127.0.0.1:${server.port}`,
			model: "test",
			initialResponseTimeoutMs: 150,
		});
		let error: unknown;
		try {
			await backend.generate([{ role: "user", content: "hi" }]);
		} catch (e) {
			error = e;
		}
		assert.ok(
			error instanceof BackendError,
			`expected BackendError, got ${String(error)}`,
		);
		assert.equal(error.category, "transient");
		assert.match(error.message, /No response from provider/);
		assert.equal(error.retryable, true);
	} finally {
		server.stop(true);
	}
});

void test("stream idle timeout aborts a stream that goes silent mid-generation", async () => {
	const server = Bun.serve({
		port: 0,
		fetch: () =>
			wedgedSseResponse(
				'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
			),
	});
	try {
		const backend = new OpenAIBackend({
			baseUrl: `http://127.0.0.1:${server.port}`,
			model: "test",
			streamIdleTimeoutMs: 150,
		});
		let sawDelta = false;
		let error: unknown;
		try {
			await backend.generate([{ role: "user", content: "hi" }], {
				callbacks: { onDelta: () => (sawDelta = true) },
			});
		} catch (e) {
			error = e;
		}
		assert.ok(sawDelta, "first chunk should have arrived before the wedge");
		assert.ok(
			error instanceof BackendError,
			`expected BackendError, got ${String(error)}`,
		);
		assert.equal(error.category, "transient");
		assert.match(error.message, /No stream data from provider/);
	} finally {
		server.stop(true);
	}
});

void test("timeout guards do not fire on a healthy fast stream", async () => {
	const server = Bun.serve({
		port: 0,
		fetch: () =>
			sseResponse([
				'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
				'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
				'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
				"data: [DONE]\n\n",
			]),
	});
	try {
		const backend = new OpenAIBackend({
			baseUrl: `http://127.0.0.1:${server.port}`,
			model: "test",
			streamIdleTimeoutMs: 150, // guard active, must not fire
		});
		let content = "";
		const response = await backend.generate([{ role: "user", content: "hi" }], {
			callbacks: { onDelta: delta => (content += delta) },
		});
		assert.equal(content, "hello");
		assert.equal(response.usage?.totalTokens, 5);
	} finally {
		server.stop(true);
	}
});
