import { describe, expect, test } from "bun:test";
import type { AgentConfig } from "@logician/log-core";
import {
	projectInitializationStatus,
	projectRuntimeStatus,
} from "../../runtime/bridge/application/runtime-status.ts";

const config = {
	baseUrl: "http://localhost",
	model: "test-model",
	tools: [],
	cwd: "/tmp",
	maxIterations: 3,
	webSearch: { baseUrl: "https://search.test", maxResults: 5 },
	runtimeHooksEnabled: true,
	hookTranscriptPath: "/tmp/transcript.jsonl",
} as AgentConfig;

describe("runtime status projection", () => {
	test("projects stable client state with an idle fallback", () => {
		const status = projectRuntimeStatus({
			config,
			toolNames: ["bash", "web_search"],
			mcpServerCount: 2,
			mcpToolCount: 4,
			mcpErrors: [],
			contextTokens: 120,
			contextMaxTokens: 1_000,
			reasoner: "none",
		});
		expect(status.agent_name).toBe("logician");
		expect(status.web_search_enabled).toBe(true);
		expect(status.runtime_state).toEqual({
			phase: "idle",
			isStreaming: false,
			pendingToolCalls: [],
			abortRequested: false,
		});
	});

	test("adds startup, plugin, and skill facts to initialization status", () => {
		const status = projectInitializationStatus({
			config,
			toolNames: ["bash"],
			mcpServerCount: 1,
			mcpToolCount: 2,
			mcpErrors: [],
			contextTokens: 0,
			reasoner: "reasoner-a",
			mcpLoaded: true,
			mcpLoading: false,
			enabledPluginRoots: [{ name: "plugin-a" }],
			loadedSkills: [
				{
					name: "Review",
					displayName: "Review",
					slashName: "review",
					description: "Review changes",
					content: "review",
					filePath: "/tmp/review/SKILL.md",
					baseDir: "/tmp/review",
					disableModelInvocation: false,
					source: "project",
				},
			],
			skillsInjected: true,
			skillsVisible: true,
			pluginCount: 1,
			hookResult: {
				hook_count: 2,
				additional_contexts: ["context"],
			},
		});
		expect(status.startup_plugins).toEqual(["plugin-a"]);
		expect(status.startup_hooks_loaded).toBe(2);
		expect(status.skills_injected).toBe(1);
		expect(status.loaded_skills).toEqual([
			{
				name: "Review",
				slash_name: "review",
				description: "Review changes",
				model_visible: true,
			},
		]);
	});
});
