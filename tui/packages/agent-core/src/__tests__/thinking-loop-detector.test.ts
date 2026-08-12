import { describe, it } from "bun:test";
import assert from "node:assert/strict";
import { ThinkingLoopDetector } from "../agent/guards/thinking-loop-detector.ts";

describe("ThinkingLoopDetector", () => {
	it("does not trip on a single short thinking turn", () => {
		const detector = new ThinkingLoopDetector();
		const result = detector.recordTurn("short text", 0, 1);
		assert.strictEqual(result, null);
	});

	it("does not trip on a long thinking turn below threshold", () => {
		const detector = new ThinkingLoopDetector();
		const longText = "x".repeat(400);
		const result = detector.recordTurn(longText, 0, 1);
		assert.strictEqual(result, null);
	});

	it("detects thinking-only turns after threshold", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			thinkingOnlyThreshold: 3,
		});
		const longText = "y".repeat(200);
		// First two turns — no trigger
		assert.strictEqual(detector.recordTurn(longText, 0, 1), null);
		assert.strictEqual(detector.recordTurn(longText, 0, 2), null);
		// Third turn — should trigger
		const result = detector.recordTurn(longText, 0, 3);
		assert.ok(
			result !== null,
			"should trigger on 3rd consecutive thinking turn",
		);
		assert.ok(result?.includes("Thinking loop detected"));
		assert.ok(result?.includes("consecutive turns with no tool calls"));
	});

	it("resets consecutive counter when a tool call turns happens", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			thinkingOnlyThreshold: 3,
		});
		const longText = "z".repeat(200);
		// Turn 1: thinking only
		assert.strictEqual(detector.recordTurn(longText, 0, 1), null);
		// Turn 2: tool call — resets all counters
		assert.strictEqual(detector.recordTurn(longText, 1, 2), null);
		// Turn 3: thinking only — counter starts fresh
		assert.strictEqual(detector.recordTurn(longText, 0, 3), null);
		// Turn 4: thinking only — still below threshold
		assert.strictEqual(detector.recordTurn(longText, 0, 4), null);
	});

	it("detects escalation when thinking grows significantly", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			escalationRatio: 1.5,
		});
		detector.recordTurn("a".repeat(100), 0, 1);
		// Next turn is 2x longer — should trigger escalation
		const result = detector.recordTurn("b".repeat(300), 0, 2);
		assert.ok(result !== null, "should detect escalation");
		assert.ok(result?.includes("spiral"));
	});

	it("resets escalation after non-thinking turn", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			escalationRatio: 1.5,
		});
		detector.recordTurn("a".repeat(100), 0, 1);
		// Non-thinking turn resets lastThinkingLength
		detector.recordTurn("b".repeat(100), 1, 2);
		// Now a long thinking turn — no escalation since counter was reset
		const result = detector.recordTurn("c".repeat(300), 0, 3);
		assert.strictEqual(result, null, "should not escalate after reset");
	});

	it("detects meta-reasoning patterns on thinking-only turns", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 50,
			metaReasoningThreshold: 2,
		});
		const metaText1 =
			"I need to think about how to approach this problem and figure out the best strategy and plan forward.";
		const metaText2 =
			"Let me think about how to handle this differently and reconsider my approach and what to do next.";
		assert.strictEqual(detector.recordTurn(metaText1, 0, 1), null);
		const result = detector.recordTurn(metaText2, 0, 2);
		assert.ok(result !== null, "should detect meta-reasoning loop");
		assert.ok(result?.includes("meta-reasoning"));
	});

	it("does not trip meta-reasoning on non-thinking turns", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			metaReasoningThreshold: 2,
		});
		const metaText =
			"I need to think about how to approach this. I'll use the read_file tool to check.";
		// Has tool calls — not a thinking turn
		assert.strictEqual(detector.recordTurn(metaText, 2, 1), null);
	});

	it("resets consecutive thinking after a non-thinking turn", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			thinkingOnlyThreshold: 3,
		});
		const longText = "a".repeat(200);
		detector.recordTurn(longText, 0, 1);
		detector.recordTurn(longText, 0, 2);
		// Non-thinking turn (tool call) — resets
		detector.recordTurn(longText, 1, 3);
		// Only 1 thinking turn again
		detector.recordTurn(longText, 0, 4);
		assert.strictEqual(detector.recordTurn(longText, 0, 5), null);
	});

	it("resets on reset()", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 100,
			thinkingOnlyThreshold: 2,
		});
		const longText = "a".repeat(200);
		detector.recordTurn(longText, 0, 1);
		detector.recordTurn(longText, 0, 2); // should trigger
		detector.reset();
		// After reset, no state remains
		assert.strictEqual(detector.recordTurn(longText, 0, 3), null);
	});

	it("exposes correct stats", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 50,
		});
		detector.recordTurn("short text here", 0, 1); // below 50 chars, not a thinking turn
		detector.recordTurn("tool call", 1, 2);
		detector.recordTurn("another short text here", 0, 3); // below 50 chars
		const stats = detector.getStats();
		assert.strictEqual(stats.consecutiveThinkingOnly, 0);
		assert.strictEqual(stats.totalThinkingTurns, 0);
	});

	it("tracks thinking turns correctly in stats", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 20,
		});
		detector.recordTurn("a".repeat(30), 0, 1); // thinking turn
		detector.recordTurn("tool call", 1, 2); // not thinking
		detector.recordTurn("b".repeat(30), 0, 3); // thinking turn
		const stats = detector.getStats();
		assert.strictEqual(stats.consecutiveThinkingOnly, 1);
		assert.strictEqual(stats.totalThinkingTurns, 2);
	});

	it("respects custom options", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 10,
			thinkingOnlyThreshold: 2,
			maxTotalThinkingTokens: 100,
		});
		const shortText = "a".repeat(20);
		detector.recordTurn(shortText, 0, 1);
		const result = detector.recordTurn(shortText, 0, 2);
		assert.ok(result !== null);
		assert.ok(result?.includes("Thinking loop detected"));
	});

	it("detects budget exhaustion", () => {
		const detector = new ThinkingLoopDetector({
			minThinkingLength: 10,
			maxTotalThinkingTokens: 100,
		});
		detector.recordTurn("x".repeat(50), 0, 1, 60); // 60 tokens
		const result = detector.recordTurn("y".repeat(50), 0, 2, 60); // total 120
		assert.ok(result !== null);
		assert.ok(result?.includes("budget"));
	});
});
