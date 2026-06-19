import assert from "node:assert/strict";
import { test } from "node:test";
import { runStatelessAgentLoop } from "../core/stateless-loop.ts";
import type { AgentConfig, Tool } from "../core/types.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

const noop: Tool = {
	name: "noop",
	description: "does nothing",
	parameters: { type: "object", properties: {} },
	execute: async () => "ok",
};

function makeConfig(): AgentConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [noop],
	};
}

void test("stateless loop returns full transcript and only new messages", async () => {
	const initialMessages = [
		{ role: "user" as const, content: "older q" },
		{ role: "assistant" as const, content: "older a" },
	];
	const result = await runStatelessAgentLoop({
		config: makeConfig(),
		backend: new FakeBackend([() => textResponse("fresh a")]),
		prompt: "fresh q",
		initialMessages,
	});

	assert.deepEqual(
		result.messages.map((m) => `${m.role}:${m.content ?? ""}`),
		["system:test", "user:older q", "assistant:older a", "user:fresh q", "assistant:fresh a"],
	);
	assert.deepEqual(
		result.newMessages.map((m) => `${m.role}:${m.content ?? ""}`),
		["user:fresh q", "assistant:fresh a"],
	);
});
