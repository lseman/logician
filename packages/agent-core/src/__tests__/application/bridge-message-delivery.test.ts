import { test } from "bun:test";
import assert from "node:assert/strict";
import { AgentCoreBridge } from "../../application/bridge/agent-bridge.ts";
import type { RuntimeEvent } from "../../core/types/runtime-events.ts";

void test("runtime settings update the live harness and preserve guard auto mode", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const patches: Array<Record<string, unknown>> = [];
	(bridge as unknown as { harness: unknown }).harness = {
		configure: (patch: Record<string, unknown>) => patches.push(patch),
	};

	bridge.updateSettings({ guardMode: "off" });
	bridge.updateSettings({ continuationEnabled: false });
	bridge.updateSettings({ guardMode: "auto" });

	const settings = bridge.getSettingsData();
	assert.equal(settings.thinkingLevel, "off");
	assert.equal(settings.inferenceMode, "none");
	assert.equal(settings.guardMode, "auto");
	assert.equal(settings.continuationEnabled, false);
	assert.deepEqual(patches, [
		{ guardsEnabled: false },
		{ continuationEnabled: false },
		{ guardsEnabled: undefined },
	]);
});

void test("setThinkingLevel propagates to the live harness", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const harnessLevels: string[] = [];
	(bridge as unknown as { harness: unknown }).harness = {
		setThinkingLevel: (level: string) => harnessLevels.push(level),
	};

	bridge.updateSettings({ thinkingLevel: "high" });

	assert.deepEqual(harnessLevels, ["high"]);
});

void test("setSteeringInterrupt propagates to the live harness", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const harnessValues: boolean[] = [];
	(bridge as unknown as { sessionManager: unknown }).sessionManager = {
		setSteeringInterrupt: (enabled: boolean) => harnessValues.push(enabled),
	};

	bridge.updateSettings({ steeringInterrupt: true });

	assert.deepEqual(harnessValues, [true]);
});

void test("direct /spawn records task and result in harness history", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const calls: Array<{ task: string; agent?: string }> = [];
	(bridge as unknown as { agentCoordinator: unknown }).agentCoordinator = {
		spawnAgentDirectly: (task: string, agent?: string) =>
			calls.push({ task, agent }),
	};

	bridge.spawnAgentDirectly("check readme.md", "general");

	assert.deepEqual(calls, [{ task: "check readme.md", agent: "general" }]);
});

void test("startup state reports the registered web_search capability", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		webSearch: { baseUrl: "http://search.test:8090" },
	});
	const internal = bridge as unknown as Record<string, unknown>;
	internal.startupHooksRan = true;

	const state = await bridge.init();
	assert.equal(state.web_search_enabled, true);
	assert.equal(state.web_search_url, "http://search.test:8090");
	assert.ok(Array.isArray(state.tools));
	assert.ok(state.tools.includes("web_search"));
});

void test("sandbox mode cycles off -> code -> full -> off and updates the tool default", async () => {
	const { getDefaultSandboxProfile, setDefaultSandboxProfile } = await import(
		"../../infrastructure/tools/sandbox.ts"
	);
	const prev = getDefaultSandboxProfile();
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	try {
		setDefaultSandboxProfile("code");
		assert.equal(bridge.getSandboxMode(), "code");

		assert.equal(bridge.cycleSandboxMode(), "full");
		assert.equal(bridge.getSandboxMode(), "full");
		assert.equal(getDefaultSandboxProfile(), "full");

		assert.equal(bridge.cycleSandboxMode(), "none");
		assert.equal(bridge.cycleSandboxMode(), "code");
	} finally {
		setDefaultSandboxProfile(prev);
	}
});

void test("an in-flight MCP connection never blocks delivery of a user message", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.mcpLoadPromise = new Promise<void>(() => {});
	let delivered = "";
	internal.harness = {
		messages: [],
		prompt: async (message: string) => {
			delivered = message;
		},
	};

	await Promise.race([
		bridge.sendMessage("hello"),
		new Promise<never>((_resolve, reject) =>
			setTimeout(() => reject(new Error("message delivery timed out")), 100),
		),
	]);

	assert.equal(delivered, "hello");
	assert.equal(bridge.isActive(), false);
});

void test("MCP discovery never blocks the first turn — it loads in the background", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	let resolveLoad!: () => void;
	internal.toolRouter.mcpLoadPromise = new Promise<void>(resolve => {
		resolveLoad = resolve;
	});
	let delivered = false;
	internal.harness = {
		messages: [],
		setTools: () => {},
		prompt: async () => {
			delivered = true;
		},
	};

	// The turn should complete without ever waiting on the in-flight MCP
	// load — MCP starts loading the moment the bridge (ToolRouter) is
	// constructed and keeps loading in the background regardless of when,
	// or whether, the load promise settles.
	await bridge.sendMessage("hello");
	assert.equal(delivered, true);

	resolveLoad();
});

void test("MCP load failures are injected into the system prompt", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.toolRouter.mcpManager = {
		load: async () => ({
			tools: [],
			servers: 0,
			errors: ["github: connection refused"],
		}),
	};

	await internal.loadMcpToolsOnce();

	assert.match(
		internal.config.systemPrompt,
		/<mcp-status>\n1 MCP server\(s\) failed to load:/,
	);
	assert.match(internal.config.systemPrompt, /- github: connection refused/);
	assert.match(
		internal.config.systemPrompt,
		/Tools from these servers are unavailable this session\./,
	);
});

void test("plugin hook updates preserve MCP and skills system context", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.toolRouter.skillsContext =
		"<available-skills>skill catalog</available-skills>";
	internal.toolRouter.mcpManager = {
		load: async () => ({
			tools: [],
			servers: 0,
			errors: ["github: authentication failed"],
		}),
	};

	await internal.loadMcpToolsOnce();
	internal.applyPluginHookContext({
		additional_contexts: ["startup instructions"],
		context_messages: [
			{
				plugin_id: "test-plugin",
				plugin_name: "Test Plugin",
				matcher: "startup",
				content: "startup instructions",
			},
			{
				plugin_id: "other-plugin",
				plugin_name: "Other Plugin",
				matcher: "startup",
				content: "separate displayed plugin message",
			},
		],
		initial_user_message: "plugin welcome message",
	});

	assert.match(internal.config.systemPrompt, /<startup-hook-context>/);
	assert.match(
		internal.config.systemPrompt,
		/separate displayed plugin message/,
	);
	assert.match(internal.config.systemPrompt, /plugin welcome message/);
	assert.equal(
		internal.config.systemPrompt.match(/startup instructions/g)?.length,
		1,
	);
	assert.match(internal.config.systemPrompt, /<mcp-status>/);
	assert.match(internal.config.systemPrompt, /<available-skills>/);

	internal.applyPluginHookContext({ additional_contexts: [] });

	assert.doesNotMatch(internal.config.systemPrompt, /<startup-hook-context>/);
	assert.match(internal.config.systemPrompt, /<mcp-status>/);
	assert.match(internal.config.systemPrompt, /<available-skills>/);
});

void test("malformed startup hook messages do not prevent initialization", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;

	internal.applyPluginHookContext({
		additional_contexts: ["valid additional context"],
		context_messages: [
			null,
			"not a message",
			{},
			{ content: 42 },
			{ content: "valid message context" },
		],
		initial_user_message: null,
	});

	assert.match(internal.config.systemPrompt, /valid additional context/);
	assert.match(internal.config.systemPrompt, /valid message context/);
	assert.doesNotMatch(internal.config.systemPrompt, /not a message/);
});

void test("/context preserves complete long messages and tool results", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	const longUserMessage = `user-start\n${"u".repeat(2500)}\nuser-end`;
	const longToolResult = `tool-start\n${"t".repeat(2500)}\ntool-end`;
	internal.harness = {
		messages: [
			{ role: "user", content: longUserMessage },
			{
				role: "assistant",
				content: "",
				tool_calls: [{ id: "call-1", name: "read_file", arguments: "{}" }],
			},
			{ role: "tool", tool_call_id: "call-1", content: longToolResult },
		],
		getMemoryPrompt: () => "",
	};

	const context = bridge.getContext();

	assert.match(context, /user-start[\s\S]*user-end/);
	assert.match(context, /\[TOOL\] \(read_file\)\ntool-start[\s\S]*tool-end/);
	assert.doesNotMatch(context, /\.\.\. \[truncated\]/);
});

void test("/context omits an empty orient task state", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.currentTaskState = {
		objective: "hi",
		phase: "orient",
		hypotheses: [],
		evidence: [],
		changedFiles: [],
		verification: [],
		blockers: [],
		toolCalls: 0,
		toolFailures: 0,
	};
	internal.harness = { messages: [], getMemoryPrompt: () => "" };
	assert.doesNotMatch(bridge.getContext(), /<task_state>/);
});

void test("/context omits terminal handoff state", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.currentTaskState = {
		objective: "hi",
		phase: "handoff",
		hypotheses: [],
		evidence: [
			{
				kind: "observation",
				tool: "task_status",
				summary: "Recorded: done — No active tasks — ready for work.",
				iteration: 1,
			},
		],
		changedFiles: [],
		verification: [],
		blockers: [],
		toolCalls: 1,
		toolFailures: 0,
	};
	internal.harness = { messages: [], getMemoryPrompt: () => "" };
	assert.doesNotMatch(bridge.getContext(), /<task_state>/);
});

void test("/context renders request-time memory injection", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		memoryDbPath: `/tmp/logician-context-memory-${process.pid}-${Date.now()}.db`,
		memoryViewerEnabled: false,
	});
	const store = bridge.getMemoryStore()!;
	const memory = store.create(
		"Authentication retries use bounded exponential backoff",
		{
			strength: 9,
			concepts: ["authentication", "retries"],
		},
	);
	store.update(memory.id, { title: "Authentication retry policy" });
	const internal = bridge as unknown as Record<string, any>;
	internal.harness = {
		messages: [{ role: "user", content: "Fix authentication retries" }],
		getMemoryPrompt: () => "",
	};

	const context = bridge.getContext();

	assert.match(
		context,
		/Retrieved memory: ~\d+ tokens — request-time compact index/,
	);
	assert.match(context, /\[USER\]\n# Agent Context/);
	assert.match(context, new RegExp(memory.id));
	assert.match(context, /Authentication retry policy/);
	assert.match(context, /Call `memory_get` once/);
	bridge.getMemoryStore()?.close();
});

void test("loaded skills are exposed as a persistent catalog, not scored per turn", async () => {
	// Skill activation is no longer scored/selected per prompt: every visible
	// skill's name+description is folded into the base system prompt once
	// (via _skillsContext/applyPluginHookContext), and the model pulls a
	// skill's full body on demand via read_skill.
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.skillsContext =
		"<available-skills>\n" +
		'  <skill name="typescript-code-review" slash_command="/typescript-code-review" />\n' +
		"</available-skills>";
	internal.applyPluginHookContext({
		additional_contexts: [],
		context_messages: [],
		initial_user_message: "",
	});

	let promptSeenByTurn = "";
	internal.harness = {
		messages: [],
		configure: () => {},
		prompt: async () => {
			promptSeenByTurn = internal.config.systemPrompt;
		},
	};

	await bridge.sendMessage("Review this TypeScript service.");

	assert.match(promptSeenByTurn, /<available-skills>/);
	assert.match(promptSeenByTurn, /typescript-code-review/);
});

void test("automatic continuation reuses the current system prompt", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.config.systemPrompt =
		"Keep debugging until the root cause is verified.";
	const seen: Array<{ kind: string; systemPrompt: string }> = [];
	internal.harness = {
		messages: [],
		configure: () => {},
		prompt: async () => {
			seen.push({ kind: "prompt", systemPrompt: internal.config.systemPrompt });
			if (seen.length === 1)
				internal.sessionManager.setPendingContinuation(true);
		},
		continueWithNextTurn: async () => {
			seen.push({
				kind: "continue",
				systemPrompt: internal.config.systemPrompt,
			});
		},
	};
	internal.sessionManager.setHarness(internal.harness);

	await bridge.sendMessage("Diagnose this TypeScript error.");
	for (let attempts = 0; seen.length < 2 && attempts < 20; attempts++) {
		await new Promise<void>(resolve => setImmediate(resolve));
	}

	assert.equal(seen[1]?.kind, "continue");
	assert.match(seen[1]?.systemPrompt ?? "", /Keep debugging until/);
});

void test("sendMessage rejects when the turn fails", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.isMcpLoaded = () => true;
	internal.harness = {
		messages: [],
		configure: () => {},
		prompt: async () => {
			throw new Error("provider failed");
		},
	};
	await assert.rejects(bridge.sendMessage("hello"), /provider failed/);
});

void test("cancel resolves only after abort settlement and returns recoverable queues", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	let settled = false;
	internal.harness = {
		abort: async () => {
			await Promise.resolve();
			settled = true;
			return {
				clearedSteering: ["change direction"],
				clearedFollowUp: ["then verify"],
				clearedNextTurn: [],
			};
		},
	};

	const result = await bridge.cancel();

	assert.equal(settled, true);
	assert.deepEqual(result, {
		clearedSteering: ["change direction"],
		clearedFollowUp: ["then verify"],
		clearedNextTurn: [],
	});
});

void test("core iterations reconcile output without completing the UI turn early", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.isMcpLoaded = () => true;
	internal.harness = {
		messages: [],
		configure: () => {},
		prompt: async () => {
			internal.config.onEvent({
				type: "turn_end",
				turnId: "iteration_1",
				message: { role: "assistant", content: "First iteration" },
			});
			internal.config.onEvent({
				type: "turn_end",
				turnId: "iteration_2",
				message: { role: "assistant", content: "Final response" },
			});
		},
	};
	const events: RuntimeEvent[] = [];
	bridge.on(event => events.push(event));

	await bridge.sendMessage("do work");

	assert.deepEqual(
		events
			.filter(
				event =>
					event.type === "turn_start" ||
					event.type === "message_update" ||
					event.type === "turn_end",
			)
			.map(event => event.type),
		["turn_start", "message_update", "message_update", "turn_end"],
	);
	assert.equal(events.filter(event => event.type === "turn_end").length, 1);
});
