import { describe, expect, test } from "bun:test";
import type { AgentConfig } from "@logician/log-core";
import type { PluginCommandResult } from "../../adapters/claude-code/plugin-runtime.ts";
import {
	PluginLifecycle,
	type PluginResourceHost,
} from "../../runtime/bridge/application/plugin-lifecycle.ts";

class FakePluginResources implements PluginResourceHost {
	mcpContext = "<mcp-status>connected</mcp-status>";
	skillsContext = "<available-skills>one</available-skills>";

	getMcpSystemContext(): string {
		return this.mcpContext;
	}

	getSkillsContext(): string {
		return this.skillsContext;
	}

	async injectSkillsFromPlugins(): Promise<void> {}
	async injectPrompts(): Promise<void> {}
}

function createLifecycle(): {
	lifecycle: PluginLifecycle;
	config: AgentConfig;
} {
	const config = {
		baseUrl: "http://localhost",
		model: "test",
		systemPrompt: "base prompt",
		tools: [],
		cwd: "/tmp",
		maxIterations: 1,
	} as AgentConfig;
	const lifecycle = new PluginLifecycle({
		config: () => config,
		baseSystemPrompt: () => "base prompt",
		sessionId: () => "session-1",
		tools: new FakePluginResources(),
		injectSubagents: async () => {},
	});
	return { lifecycle, config };
}

describe("PluginLifecycle", () => {
	test("composes deduplicated hook, MCP, and skill context", () => {
		const { lifecycle, config } = createLifecycle();
		lifecycle.applyContext({
			additional_contexts: ["hook context", "hook context"],
			context_messages: [
				{
					plugin_id: "one",
					plugin_name: "One",
					matcher: "startup",
					content: "message context",
				},
				{
					plugin_id: "two",
					plugin_name: "Two",
					matcher: "startup",
					content: "hook context",
				},
			],
			initial_user_message: "welcome",
		});

		expect(config.systemPrompt).toContain("base prompt");
		expect(config.systemPrompt).toContain("<startup-hook-context>");
		expect(config.systemPrompt).toContain("<mcp-status>");
		expect(config.systemPrompt).toContain("<available-skills>");
		expect(config.systemPrompt?.match(/hook context/g)).toHaveLength(1);
	});

	test("ignores malformed message contexts", () => {
		const { lifecycle, config } = createLifecycle();
		lifecycle.applyContext({
			additional_contexts: ["valid"],
			context_messages: [null, "bad", {}, { content: 42 }],
		} as unknown as PluginCommandResult);
		expect(config.systemPrompt).toContain("valid");
		expect(config.systemPrompt).not.toContain("[object Object]");
	});

	test("refresh preserves the last accepted hook result", () => {
		const { lifecycle, config } = createLifecycle();
		lifecycle.applyContext({ additional_contexts: ["durable context"] });
		lifecycle.refreshContext();
		expect(config.systemPrompt).toContain("durable context");
	});
});
