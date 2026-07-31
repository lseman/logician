import assert from "node:assert/strict";
import { test } from "node:test";
import { mapAgentEvent } from "../runtime/event-mapping.ts";
import { STEERING_INTERRUPT_SUMMARY } from "@logician/agent-core";

void test("context updates preserve unavailable provider telemetry", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "context_update",
			tokens: 1_000,
			maxTokens: 32_768,
			cachedTokens: null,
			promptTokens: null,
			completionTokens: null,
		}),
		{
			type: "context_update",
			tokens: 1_000,
			max_tokens: 32_768,
			compacted: undefined,
			cached_tokens: null,
			prompt_tokens: null,
			completion_tokens: null,
		},
	);
});

void test("steering cancellation maps to one informational notice", () => {
	assert.deepEqual(
		mapAgentEvent({
			type: "run_outcome",
			status: "cancelled",
			summary: STEERING_INTERRUPT_SUMMARY,
			source: "runtime",
		}),
		{
			type: "notice",
			level: "info",
			label: "Steering",
			text: STEERING_INTERRUPT_SUMMARY,
		},
	);
});
