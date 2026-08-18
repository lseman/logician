import { test } from "bun:test";
import assert from "node:assert/strict";
import { classifyHttpError, parseProviderUsage } from "../core/backend.ts";

void test("429 with numeric Retry-After carries retryAfterMs", () => {
	const err = classifyHttpError(429, "rate limited", "2");
	assert.equal(err.category, "rate_limit");
	assert.equal(err.retryable, true);
	assert.equal(err.retryAfterMs, 2000);
});

void test("429 without Retry-After leaves retryAfterMs undefined", () => {
	const err = classifyHttpError(429, "rate limited");
	assert.equal(err.retryAfterMs, undefined);
});

void test("Retry-After is clamped to five minutes", () => {
	const err = classifyHttpError(429, "rate limited", "9000");
	assert.equal(err.retryAfterMs, 5 * 60_000);
});

void test("context-full body beats the status code", () => {
	const err = classifyHttpError(400, "prompt exceeds maximum context length");
	assert.equal(err.category, "context_full");
});

void test("5xx is transient, other 4xx is client", () => {
	assert.equal(classifyHttpError(503, "unavailable").category, "transient");
	assert.equal(classifyHttpError(404, "nope").category, "client");
});

void test("malformed assistant message is client, not context_full", () => {
	const err = classifyHttpError(
		400,
		"Assistant message must contain either 'content' or 'tool_calls'!",
	);
	assert.equal(err.category, "client");
	assert.equal(err.retryable, false);
});

void test("failed tool-call JSON parse is poisoned_history, not transient", () => {
	const err = classifyHttpError(
		500,
		'{"error":{"code":500,"message":"Failed to parse tool call arguments as JSON"}}',
	);
	assert.equal(err.category, "poisoned_history");
	assert.equal(err.retryable, false);
});

void test("llama.cpp cached prompt tokens are normalized from usage details", () => {
	assert.deepEqual(
		parseProviderUsage({
			prompt_tokens: 20_000,
			completion_tokens: 50,
			total_tokens: 20_050,
			prompt_tokens_details: { cached_tokens: 12_400 },
		}),
		{
			promptTokens: 20_000,
			completionTokens: 50,
			totalTokens: 20_050,
			cachedTokens: 12_400,
		},
	);
});

void test("missing provider cache telemetry remains unknown", () => {
	assert.deepEqual(parseProviderUsage({ prompt_tokens: 42 }), {
		promptTokens: 42,
		completionTokens: undefined,
		totalTokens: undefined,
	});
});

void test("llama.cpp timings cache_n is accepted as a legacy fallback", () => {
	assert.deepEqual(parseProviderUsage(undefined, { cache_n: 236 }), {
		promptTokens: undefined,
		completionTokens: undefined,
		totalTokens: undefined,
		cachedTokens: 236,
	});
});
