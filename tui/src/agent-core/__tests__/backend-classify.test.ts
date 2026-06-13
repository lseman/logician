import assert from "node:assert/strict";
import { test } from "node:test";
import { classifyHttpError } from "../backend.ts";

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
