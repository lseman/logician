import { test } from "bun:test";
import assert from "node:assert/strict";
import { resolveAgentSettings } from "../../core/configuration/agent-settings.ts";
import type { AgentConfig } from "../../core/types/types-config.ts";

const baseConfig: AgentConfig = {
	baseUrl: "http://fake",
	model: "fake",
};

void test("agent settings use Pi-like direct defaults", () => {
	assert.deepEqual(resolveAgentSettings(baseConfig), {
		executionProfile: "minimal",
		inferenceMode: "none",
		maxIterations: 30,
		thinkingLevel: "off",
		toolExecution: "parallel",
	});
});

void test("agent settings preserve explicit choices", () => {
	assert.deepEqual(
		resolveAgentSettings({
			...baseConfig,
			executionProfile: "autonomous",
			inferenceMode: "thinking-coding",
			maxIterations: 8,
			thinkingLevel: "high",
			toolExecution: "sequential",
		}),
		{
			executionProfile: "autonomous",
			inferenceMode: "thinking-coding",
			maxIterations: 8,
			thinkingLevel: "high",
			toolExecution: "sequential",
		},
	);
});
