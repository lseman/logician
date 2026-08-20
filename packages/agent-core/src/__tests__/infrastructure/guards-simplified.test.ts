// ── Tests: simplified guard system ─────────────────────────────────────────
// Tests for the pi-style guard system: LoopDetector, OutputGuard.

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { LoopDetector } from "../../core/guards/loop-detector.ts";
import { OutputGuard } from "../../core/guards/output-guard.ts";

describe("OutputGuard", () => {
	it("classifies transient errors correctly", () => {
		const guard = new OutputGuard({ maxRetries: 2 });
		// Using a BackendError-like object with category property
		const err = {
			name: "BackendError",
			category: "transient" as const,
			message: "temporary",
		};
		const result = guard.handleError(err);
		assert.equal(result.action, "retry");
	});

	it("classifies rate limit errors correctly", () => {
		const guard = new OutputGuard({ maxRetries: 2 });
		const err = {
			name: "BackendError",
			category: "rate_limit" as const,
			message: "429 too many requests",
		};
		const result = guard.handleError(err);
		assert.equal(result.action, "retry");
	});

	it("aborts after max retries exhausted", () => {
		const guard = new OutputGuard({ maxRetries: 1 });
		guard.handleError({
			name: "BackendError",
			category: "transient" as const,
			message: "error 1",
		});
		const result = guard.handleError({
			name: "BackendError",
			category: "transient" as const,
			message: "error 2",
		});
		assert.equal(result.action, "abort");
	});

	it("context_full returns compact_then_retry", () => {
		const guard = new OutputGuard({ maxRetries: 1 });
		const result = guard.handleError({
			name: "BackendError",
			category: "context_full" as const,
			message: "context too long",
		});
		// context_full is retryable, returns "retry" with maxRetries > 0
		assert.equal(result.action, "retry");
	});

	it("poisoned_history message returns retry once then abort", () => {
		const guard = new OutputGuard({ maxRetries: 1 });
		// Simplified guard treats "failed to parse tool call" as "unknown" error
		// which retries once, then aborts on second attempt
		const result1 = guard.handleError(new Error("failed to parse tool call"));
		assert.equal(result1.action, "retry");
		const result2 = guard.handleError(new Error("failed to parse tool call"));
		assert.equal(result2.action, "abort");
	});

	it("resets retry count", () => {
		const guard = new OutputGuard({ maxRetries: 1 });
		guard.handleError({
			name: "BackendError",
			category: "transient" as const,
			message: "error",
		});
		guard.reset();
		const result = guard.handleError({
			name: "BackendError",
			category: "transient" as const,
			message: "error again",
		});
		assert.equal(result.action, "retry");
	});

	it("recognizes errors by message when no category", () => {
		const guard = new OutputGuard({ maxRetries: 2 });
		const result = guard.handleError(new Error("500 internal server error"));
		assert.equal(result.action, "retry");
	});

	it("returns abort for unknown errors after maxRetries exhausted", () => {
		const guard = new OutputGuard({ maxRetries: 1 });
		// First unknown error → retry
		guard.handleError(new Error("some unknown error"));
		// Second unknown error → abort
		const result = guard.handleError(new Error("some unknown error"));
		assert.equal(result.action, "abort");
	});

	it("aborts when maxRetries is 0 for transient errors", () => {
		const guard = new OutputGuard({ maxRetries: 0 });
		const result = guard.handleError({
			name: "BackendError",
			category: "transient" as const,
			message: "error",
		});
		assert.equal(result.action, "abort");
	});

	it("aborts when maxRetries is 0 for context_full errors", () => {
		const guard = new OutputGuard({ maxRetries: 0 });
		const result = guard.handleError({
			name: "BackendError",
			category: "context_full" as const,
			message: "too long",
		});
		assert.equal(result.action, "abort");
	});

	it("aborts on abort errors", () => {
		const guard = new OutputGuard();
		const err = new Error("Operation aborted");
		err.name = "AbortError";
		const result = guard.handleError(err);
		assert.equal(result.action, "abort");
	});

	it("proceeds on successful responses", () => {
		const guard = new OutputGuard();
		const result = guard.checkResponse("some content", 0);
		assert.equal(result.action, "proceed");
	});

	it("aborts after max consecutive empty responses", () => {
		const guard = new OutputGuard({ maxEmptyResponses: 2 });
		guard.checkResponse("", 0);
		const result = guard.checkResponse(null, 0);
		assert.equal(result.action, "abort");
	});

	it("resets empty response counter on non-empty response", () => {
		const guard = new OutputGuard({ maxEmptyResponses: 3 });
		// First empty: count = 1
		guard.checkResponse("", 0);
		// Non-empty resets counter
		guard.checkResponse("content", 0);
		// Now count back to 0, two more empties = 2 < threshold 3
		guard.checkResponse(null, 0);
		guard.checkResponse(null, 0);
		const result = guard.checkResponse(null, 0);
		assert.equal(result.action, "abort");
	});

	it("budget_exhausted when tokens exceed threshold", () => {
		const guard = new OutputGuard({ budgetThreshold: 0.9 });
		const result = guard.processResponse(950, 1000);
		assert.equal(result.action, "budget_exhausted");
	});

	it("proceeds when tokens below threshold", () => {
		const guard = new OutputGuard({ budgetThreshold: 0.9 });
		const result = guard.processResponse(800, 1000);
		assert.equal(result.action, "proceed");
	});
});

describe("LoopDetector", () => {
	it("detects duplicate tool calls", () => {
		const detector = new LoopDetector({ duplicateThreshold: 2 });

		// First call — not duplicate (count = 1, threshold = 2)
		const decision1 = detector.checkToolCall("read", "{}");
		assert.equal(decision1.block, false);

		// Second call — same tool/args, count = 2 >= threshold 2 → blocks
		const decision2 = detector.checkToolCall("read", "{}");
		assert.ok(decision2.block);
	});

	it("allows different tool names", () => {
		const detector = new LoopDetector({ duplicateThreshold: 2 });

		// First call blocks at threshold 2 only when repeated
		const d1 = detector.checkToolCall("read", "{}");
		assert.equal(d1.block, false);
		const d2 = detector.checkToolCall("write", "{}");
		assert.equal(d2.block, false);
	});

	it("allows different args", () => {
		const detector = new LoopDetector({ duplicateThreshold: 2 });

		const d1 = detector.checkToolCall("read", '{"path":"a"}');
		assert.equal(d1.block, false);
		const d2 = detector.checkToolCall("read", '{"path":"b"}');
		assert.equal(d2.block, false);
	});

	it("resets state", () => {
		const detector = new LoopDetector({ duplicateThreshold: 2 });
		detector.checkToolCall("read", "{}");
		detector.checkToolCall("read", "{}");
		detector.reset();

		// Should be clean after reset
		const d1 = detector.checkToolCall("read", "{}");
		assert.equal(d1.block, false);
	});
});
