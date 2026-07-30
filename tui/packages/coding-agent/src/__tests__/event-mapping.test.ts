import assert from "node:assert/strict";
import { test } from "node:test";
import { mapAgentEvent } from "../runtime/event-mapping.ts";

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
