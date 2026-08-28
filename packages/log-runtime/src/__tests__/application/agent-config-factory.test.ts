import { describe, expect, test } from "bun:test";
import { createAgentConfig } from "../../runtime/bridge/application/agent-config-factory.ts";

function createConfig(overrides: Record<string, unknown> = {}) {
	const onPermissionRequest = async () => "allow" as const;
	const onQuestionRequest = async () => "answer";
	const onTurnEnd = () => {};
	const onEvent = () => {};
	const hooks = {};
	const config = createAgentConfig({
		bridge: {
			baseUrl: "http://localhost:8080",
			model: "test-model",
			...overrides,
		},
		cwd: "/workspace",
		sessionId: "session-1",
		transcriptPath: "/tmp/session-1.jsonl",
		systemPrompt: "system",
		tools: [],
		webSearch: { baseUrl: "http://search", maxResults: 8 },
		permissions: undefined,
		hooks,
		onPermissionRequest,
		onQuestionRequest,
		onTurnEnd,
		onEvent,
	});
	return {
		config,
		hooks,
		onPermissionRequest,
		onQuestionRequest,
		onTurnEnd,
		onEvent,
	};
}

describe("createAgentConfig", () => {
	test("owns runtime defaults independently of AgentRuntime construction", () => {
		const { config } = createConfig();
		expect(config.maxIterations).toBe(30);
		expect(config.thinkingLevel).toBe("off");
		expect(config.inferenceMode).toBe("none");
		expect(config.toolExecution).toBe("parallel");
		expect(config.graphicianEnabled).toBe(true);
		expect(config.fffgrepEnabled).toBe(true);
	});

	test("forwards policy options, paths, hooks, and callbacks", () => {
		const state = createConfig({
			maxIterations: 12,
			thinkingLevel: "high",
			toolExecution: "sequential",
			graphicianEnabled: false,
			allowedPaths: ["/shared"],
			allowAllPaths: true,
		});
		const { config } = state;
		expect(config.maxIterations).toBe(12);
		expect(config.thinkingLevel).toBe("high");
		expect(config.toolExecution).toBe("sequential");
		expect(config.graphicianEnabled).toBe(false);
		expect(config.allowedPaths).toEqual(["/shared"]);
		expect(config.allowAllPaths).toBe(true);
		expect(config.hookSessionId).toBe("session-1");
		expect(config.hookTranscriptPath).toBe("/tmp/session-1.jsonl");
		expect(config.eventLogPath).toBe("/tmp/session-1.events.jsonl");
		expect(config.hooks).toBe(state.hooks);
		expect(config.onPermissionRequest).toBe(state.onPermissionRequest);
		expect(config.onQuestionRequest).toBe(state.onQuestionRequest);
		expect(config.turnEndCallback).toBe(state.onTurnEnd);
		expect(config.onEvent).toBe(state.onEvent);
	});
});
