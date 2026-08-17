import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import { BackendError } from "../agent/core/backend.ts";
import { OutputGuard } from "../agent/guards/output-guard.ts";

describe("OutputGuard", () => {
	const makeGuard = (overrides = {}) =>
		new OutputGuard({
			maxRetries: 3,
			retryBaseDelayMs: 500,
			maxRetryDelayMs: 15000,
			autoCompactOnContextFull: true,
			maxEmptyResponses: 3,
			...overrides,
		});

	// ── Error handling ──────────────────────────────────────────────────────

	it("handles context_full: returns compact_then_retry", () => {
		const guard = makeGuard();
		const err = new Error("context is too long for model");
		const result = guard.handleError(err);

		assert.strictEqual(result.action, "compact_then_retry");
		assert.strictEqual(result.attempt, 1);
		assert.strictEqual(result.isRetryable, true);
		assert.ok(result.message?.includes("compaction"));
	});

	it("handles context_full without auto-compact: returns abort", () => {
		const guard = makeGuard({ autoCompactOnContextFull: false });
		const err = new Error("reduce the length of input");
		const result = guard.handleError(err);

		assert.strictEqual(result.action, "abort");
		assert.strictEqual(result.isRetryable, false);
	});

	it("aborts immediately (no retries) on poisoned_history, never blind-retries as transient", () => {
		const guard = makeGuard();
		const err = new BackendError({
			category: "poisoned_history",
			message: "Failed to parse tool call arguments as JSON",
			status: 500,
		});

		const result = guard.handleError(err);

		assert.strictEqual(result.action, "abort");
		assert.strictEqual(result.isRetryable, false);
	});

	it("classifies raw 'failed to parse tool call' errors as poisoned_history via string fallback", () => {
		const guard = makeGuard();
		const err = new Error("500 Failed to parse tool call arguments as JSON");

		const result = guard.handleError(err);

		assert.strictEqual(result.action, "abort");
		assert.strictEqual(result.isRetryable, false);
	});

	it("aborts after repeated context_full cycles instead of compacting forever", () => {
		const guard = makeGuard({ maxConsecutiveCompactions: 2 });
		const err = new Error("context is too long for model");

		let result = guard.handleError(err);
		assert.strictEqual(result.action, "compact_then_retry");
		result = guard.handleError(err);
		assert.strictEqual(result.action, "compact_then_retry");
		result = guard.handleError(err);

		assert.strictEqual(result.action, "abort");
		assert.strictEqual(result.isRetryable, false);
		assert.ok(result.message?.includes("compaction attempts"));
	});

	it("resets the context_full cycle counter after a successful response", () => {
		const guard = makeGuard({ maxConsecutiveCompactions: 2 });
		const err = new Error("context is too long for model");

		guard.handleError(err);
		guard.handleError(err);
		guard.checkResponse("ok, continuing", 1);

		const result = guard.handleError(err);
		assert.strictEqual(result.action, "compact_then_retry");
	});

	it("handles rate_limit with backoff retry", () => {
		const guard = makeGuard();
		const err = new BackendError({
			category: "rate_limit",
			message: "429 Too Many Requests",
			retryAfterMs: 2000,
		});

		let result = guard.handleError(err);
		assert.strictEqual(result.action, "retry");
		assert.strictEqual(result.retryDelayMs, 2000);
		assert.strictEqual(result.attempt, 1);
		assert.strictEqual(result.isRetryable, true);

		result = guard.handleError(err);
		assert.strictEqual(result.action, "retry");
		assert.strictEqual(result.retryDelayMs, 2000);
		assert.strictEqual(result.attempt, 2);
	});

	it("aborts after max retries exhausted", () => {
		const guard = makeGuard({ maxRetries: 2 });
		const err = new BackendError({
			category: "transient",
			message: "502 Bad Gateway",
		});

		guard.handleError(err);
		guard.handleError(err);
		const result = guard.handleError(err);

		assert.strictEqual(result.action, "abort");
		assert.strictEqual(result.isRetryable, true);
	});

	it("handles unknown errors with single retry", () => {
		const guard = makeGuard();
		const err = new Error("some unknown thing broke");

		const result = guard.handleError(err);
		assert.strictEqual(result.action, "retry");
		assert.strictEqual(result.retryDelayMs, 500);
	});

	it("aborts unknown error on second attempt", () => {
		const guard = makeGuard();
		const err = new Error("unknown");

		guard.handleError(err);
		const result = guard.handleError(err);

		assert.strictEqual(result.action, "abort");
		assert.strictEqual(result.isRetryable, false);
	});

	// ── Empty response detection ───────────────────────────────────────────

	it("returns proceed for valid responses", () => {
		const guard = makeGuard();
		const result = guard.checkResponse("Hello, how can I help?", 1);
		assert.strictEqual(result.action, "proceed");
	});

	it("tracks empty responses and aborts after threshold", () => {
		const guard = makeGuard({ maxEmptyResponses: 2 });

		let result = guard.checkResponse(null, 0);
		assert.strictEqual(result.action, "proceed");

		result = guard.checkResponse("", 0);
		assert.strictEqual(result.action, "abort");
	});

	it("resets empty response counter on non-empty response", () => {
		const guard = makeGuard({ maxEmptyResponses: 2 });

		guard.checkResponse(null, 0);
		guard.checkResponse("", 0);
		guard.checkResponse("recovery", 0);

		const result = guard.checkResponse(null, 0);
		assert.strictEqual(result.action, "proceed");
	});

	// ── Context tracking ────────────────────────────────────────────────────

	it("emits context_update on processResponse", () => {
		const events: Array<{ type: string }> = [];
		const guard = makeGuard({
			onEvent: (event: { type: string }) => events.push(event),
		});

		guard.processResponse(5000, 10000);
		assert.ok(events.some(e => e.type === "context_update"));
	});

	it("emits budget_exhausted when near limit", () => {
		const events: Array<{ type: string }> = [];
		const guard = makeGuard({
			onEvent: (event: { type: string }) => events.push(event),
		});

		const result = guard.processResponse(9600, 10000);
		assert.strictEqual(result.action, "budget_exhausted");
		assert.ok(events.some(e => e.type === "budget_exhausted"));
	});

	it("returns proceed when under budget", () => {
		const events: Array<{ type: string }> = [];
		const guard = makeGuard({
			onEvent: (event: { type: string }) => events.push(event),
		});

		const result = guard.processResponse(5000, 10000);
		assert.strictEqual(result.action, "proceed");
	});

	// ── Reset and diagnostics ──────────────────────────────────────────────

	it("resets retry count on reset", () => {
		const guard = makeGuard();
		guard.handleError(new Error("err"));
		assert.strictEqual(guard.getRetryCount(), 1);
		guard.reset();
		assert.strictEqual(guard.getRetryCount(), 0);
	});

	it("extracts context_full from error message", () => {
		const guard = makeGuard();
		const err = new Error("Error: context exceeds maximum context tokens");
		const result = guard.handleError(err);
		assert.strictEqual(result.action, "compact_then_retry");
	});

	it("extracts rate_limit from error message", () => {
		const guard = makeGuard({ maxRetries: 2 });
		const err = new Error("rate limit exceeded: 429");
		const result = guard.handleError(err);
		assert.strictEqual(result.action, "retry");
		assert.strictEqual(result.retryDelayMs, 500);
	});

	it("uses exponential backoff when no retryAfter header", () => {
		const guard = makeGuard();
		const err = new BackendError({
			category: "transient",
			message: "502 Bad Gateway",
		});

		const r0 = guard.handleError(err);
		assert.strictEqual(r0.retryDelayMs, 500);

		const r1 = guard.handleError(err);
		assert.strictEqual(r1.retryDelayMs, 1000);

		const r2 = guard.handleError(err);
		assert.strictEqual(r2.retryDelayMs, 2000);
	});

	it("maxRetries zero disables the unknown-error safety retry", () => {
		const guard = makeGuard({ maxRetries: 0 });
		const result = guard.handleError(new Error("unclassified failure"));

		assert.strictEqual(result.action, "abort");
	});

	// ── Malformed assistant message recovery ───────────────────────────────

	it("recovers from malformed assistant message error via compact_then_retry", () => {
		const guard = makeGuard();
		const err = new Error(
			"LLM request failed: 400 Assistant message must contain either 'content' or 'tool_calls'!",
		);
		const result = guard.handleError(err);

		assert.strictEqual(result.action, "compact_then_retry");
		assert.strictEqual(result.attempt, 1);
		assert.strictEqual(result.isRetryable, true);
		assert.ok(result.message?.includes("Malformed assistant message"));
	});

	it("recovers from malformed assistant message even without BackendError wrapper", () => {
		const guard = makeGuard();
		const err = new Error(
			"Assistant message must contain either 'content' or 'tool_calls'!",
		);
		const result = guard.handleError(err);

		assert.strictEqual(result.action, "compact_then_retry");
		assert.strictEqual(result.isRetryable, true);
	});
});
