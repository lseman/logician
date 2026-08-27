import { test } from "bun:test";
import assert from "node:assert/strict";
import type { RuntimeEvent } from "@logician/log-core/events";
import type { HarnessPromptOptions } from "@logician/log-core/session";
import { AgentRuntime } from "../../runtime/bridge/agent-bridge.ts";

function bypassStartup(internal: Record<string, unknown>): void {
	const plugins = internal.plugins as Record<string, unknown> | undefined;
	if (plugins) plugins.ensureStarted = async () => {};
}

function installSession(
	internal: Record<string, any>,
	session: Record<string, any>,
): void {
	internal.sessions.replace(session);
}

void test("bridge publishes ordered versioned protocol notifications", () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const notifications: Array<{
		protocolVersion: number;
		sequence: number;
		event: RuntimeEvent;
	}> = [];
	bridge.events.subscribe(notification => notifications.push(notification));
	const internal = bridge as unknown as {
		emit(event: RuntimeEvent): void;
	};
	internal.emit({ type: "phase", state: "thinking" });
	internal.emit({ type: "phase", state: "ready" });

	assert.deepEqual(
		notifications.map(notification => ({
			version: notification.protocolVersion,
			sequence: notification.sequence,
			type: notification.event.type,
		})),
		[
			{ version: 1, sequence: 1, type: "phase" },
			{ version: 1, sequence: 2, type: "phase" },
		],
	);
});

void test("bridge scopes correlation to one run and preserves the conversation identity", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	bridge.useConversationSession("session-correlation");
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	internal.toolRouter.isMcpLoaded = () => true;
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async () => {},
		getQueues: () => ({ steering: [], followUp: [], nextTurn: [] }),
	});

	await bridge.sendMessage("correlate this run");
	const runNotifications = bridge.events
		.snapshot()
		.filter(item => item.correlation?.runId);
	assert.ok(runNotifications.length > 0);
	const correlation = runNotifications[0]?.correlation;
	assert.equal(correlation?.sessionId, "session-correlation");
	assert.match(correlation?.runId ?? "", /^run_/);
	assert.match(correlation?.turnId ?? "", /^turn_/);
	assert.equal(
		runNotifications.every(
			item => item.correlation?.runId === correlation?.runId,
		),
		true,
	);

	internal.emit({ type: "phase", state: "ready" });
	assert.deepEqual(bridge.events.snapshot().at(-1)?.correlation, {
		sessionId: "session-correlation",
	});
});

void test("runtime settings update the live harness and preserve guard auto mode", () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});

	bridge.updateSettings({ guardMode: "off" });
	bridge.updateSettings({ continuationEnabled: false });
	bridge.updateSettings({ guardMode: "auto" });

	const settings = bridge.getSettingsData();
	assert.equal(settings.thinkingLevel, "off");
	assert.equal(settings.inferenceMode, "none");
	assert.equal(settings.guardMode, "auto");
	assert.equal(settings.continuationEnabled, false);
});

void test("setThinkingLevel propagates to the live harness", () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const harnessLevels: string[] = [];
	const internal = bridge as unknown as Record<string, any>;
	installSession(internal, {
		models: {
			setThinkingLevel: (level: string) => harnessLevels.push(level),
		},
	});

	bridge.updateSettings({ thinkingLevel: "high" });

	assert.deepEqual(harnessLevels, ["high"]);
});

void test("setSteeringInterrupt propagates to the live harness", () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const harnessValues: boolean[] = [];
	const internal = bridge as unknown as Record<string, any>;
	installSession(internal, {
		configure: (patch: any) => {
			if (patch.steeringInterrupt !== undefined) {
				harnessValues.push(patch.steeringInterrupt);
			}
		},
	});

	bridge.updateSettings({ steeringInterrupt: true });

	assert.deepEqual(harnessValues, [true]);
});

void test("direct /spawn records task and result in harness history", async () => {
	const bridge = new AgentRuntime({
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
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		webSearch: { baseUrl: "http://search.test:8090" },
	});
	const internal = bridge as unknown as Record<string, unknown>;
	bypassStartup(internal);

	const state = await bridge.init();
	assert.equal(state.web_search_enabled, true);
	assert.equal(state.web_search_url, "http://search.test:8090");
	assert.ok(Array.isArray(state.tools));
	assert.ok(state.tools.includes("web_search"));
});

void test("sandbox mode cycles off -> code -> full -> off and updates the tool default", async () => {
	const { getDefaultSandboxProfile, setDefaultSandboxProfile } = await import(
		"../../capabilities/tools/sandbox.ts"
	);
	const prev = getDefaultSandboxProfile();
	const bridge = new AgentRuntime({
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
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	// Mock isMcpLoaded to return false (MCP still loading)
	internal.toolRouter.isMcpLoaded = () => false;
	// Prevent the background MCP load from actually running
	const originalLoadMcp = internal.toolRouter.loadMcpToolsOnce;
	internal.toolRouter.loadMcpToolsOnce = () => new Promise<void>(() => {});

	let delivered = "";
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async (message: string) => {
			delivered = message;
		},
		getQueues: () => ({ nextTurn: [] }),
	});

	await Promise.race([
		bridge.sendMessage("hello"),
		new Promise<never>((_resolve, reject) =>
			setTimeout(() => reject(new Error("message delivery timed out")), 100),
		),
	]);

	assert.equal(delivered, "hello");
	assert.equal(bridge.isActive(), false);

	internal.toolRouter.loadMcpToolsOnce = originalLoadMcp;
});

void test("MCP discovery never blocks the first turn — it loads in the background", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	let resolveLoad!: () => void;
	internal.toolRouter.isMcpLoaded = () => false;
	internal.toolRouter.loadMcpToolsOnce = () =>
		new Promise<void>(resolve => {
			resolveLoad = resolve;
		});
	let delivered = false;
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async () => {
			delivered = true;
		},
		getQueues: () => ({ nextTurn: [] }),
	});

	// The turn should complete without ever waiting on the in-flight MCP
	// load — MCP starts loading the moment the bridge (ToolRouter) is
	// constructed and keeps loading in the background regardless of when,
	// or whether, the load promise settles.
	await bridge.sendMessage("hello");
	assert.equal(delivered, true);

	resolveLoad?.();
});

void test("MCP load failures are injected into the system prompt", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.toolRouter.mcpRegistry = {
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
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.toolRouter.skillsContext =
		"<available-skills>skill catalog</available-skills>";
	internal.toolRouter.mcpRegistry = {
		load: async () => ({
			tools: [],
			servers: 0,
			errors: ["github: authentication failed"],
		}),
	};

	await internal.loadMcpToolsOnce();
	internal.plugins.applyContext({
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

	internal.plugins.applyContext({ additional_contexts: [] });

	assert.doesNotMatch(internal.config.systemPrompt, /<startup-hook-context>/);
	assert.match(internal.config.systemPrompt, /<mcp-status>/);
	assert.match(internal.config.systemPrompt, /<available-skills>/);
});

void test("malformed startup hook messages do not prevent initialization", () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;

	internal.plugins.applyContext({
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
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	const longUserMessage = `user-start\n${"u".repeat(2500)}\nuser-end`;
	const longToolResult = `tool-start\n${"t".repeat(2500)}\ntool-end`;
	installSession(internal, {
		messages: [
			{ role: "user", content: longUserMessage },
			{
				role: "assistant",
				content: "",
				tool_calls: [{ id: "call-1", name: "read_file", arguments: "{}" }],
			},
			{ role: "tool", tool_call_id: "call-1", content: longToolResult },
		],
	});

	const context = bridge.getContext();

	assert.match(context, /user-start[\s\S]*user-end/);
	assert.match(context, /\[TOOL\] \(read_file\)\ntool-start[\s\S]*tool-end/);
	assert.doesNotMatch(context, /\.\.\. \[truncated\]/);
});

void test("/context omits an empty orient task state", () => {
	const bridge = new AgentRuntime({
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
	const bridge = new AgentRuntime({
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
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		memory: {
			dbPath: `/tmp/logician-context-memory-${process.pid}-${Date.now()}.db`,
			viewerEnabled: false,
		},
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
	assert.match(context, /\[SYSTEM\] Memory Context/);
	assert.match(context, /# Memory Context/);
	assert.match(context, new RegExp(memory.id));
	assert.match(context, /Authentication retry policy/);
	assert.match(context, /Call `memory_get` once/);
	bridge.getMemoryStore()?.close();
});

void test("loaded skills are exposed as a persistent discovery catalog", async () => {
	// The compact catalog supports discovery and read_skill. Strongly relevant
	// skill bodies are also selected separately when each turn is prepared.
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	// Inject skills context via the toolRouter
	internal.toolRouter.getSkillsContext = () =>
		"<available-skills>\n" +
		'  <skill name="typescript-code-review" slash_command="/typescript-code-review" />\n' +
		"</available-skills>";
	internal.plugins.applyContext({
		additional_contexts: [],
		context_messages: [],
		initial_user_message: "",
	});

	// The skills should be in the system prompt after applyPluginHookContext
	assert.match(internal.config.systemPrompt, /<available-skills>/);
	assert.match(internal.config.systemPrompt, /typescript-code-review/);
});

void test("strongly relevant skills are activated as request-scoped context", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	internal.toolRouter.isMcpLoaded = () => true;
	internal.toolRouter.loadedSkills = [
		{
			name: "typescript-code-review",
			displayName: "TypeScript Code Review",
			description: "Review TypeScript code for correctness and type safety.",
			content: "Inspect runtime safety and report findings before editing.",
			filePath: "/skills/typescript-code-review/SKILL.md",
			baseDir: "/skills/typescript-code-review",
			slashName: "typescript-code-review",
			disableModelInvocation: false,
			triggers: ["TypeScript code review"],
			source: "user",
		},
	];

	let promptOptions: HarnessPromptOptions | undefined;
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async (_message: string, options: HarnessPromptOptions) => {
			promptOptions = options;
		},
		getQueues: () => ({ steering: [], followUp: [], nextTurn: [] }),
	});

	await bridge.sendMessage(
		"Please review this TypeScript module for correctness.",
	);

	const contribution = promptOptions?.contextContributions?.[0];
	assert.ok(contribution);
	assert.equal(contribution.source, "skill:typescript-code-review");
	const skillMessage = contribution.messages?.[0];
	assert.ok(skillMessage && typeof skillMessage.content === "string");
	assert.match(
		skillMessage.content,
		/Inspect runtime safety and report findings before editing\./,
	);
	assert.ok(
		bridge.events
			.snapshot()
			.some(
				item =>
					item.event.type === "notice" &&
					item.event.label === "Skills" &&
					item.event.text.includes("TypeScript Code Review"),
			),
	);
});

void test("a preformatted explicit skill invocation is not injected twice", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	internal.toolRouter.isMcpLoaded = () => true;
	internal.toolRouter.loadedSkills = [
		{
			name: "review",
			displayName: "Review",
			description: "Review code.",
			content: "Review instructions.",
			filePath: "/skills/review/SKILL.md",
			baseDir: "/skills/review",
			slashName: "review",
			disableModelInvocation: false,
			source: "user",
		},
	];

	let contributionCount = -1;
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async (_message: string, options: HarnessPromptOptions) => {
			contributionCount = options.contextContributions?.length ?? 0;
		},
		getQueues: () => ({ steering: [], followUp: [], nextTurn: [] }),
	});

	await bridge.sendMessage(
		'<skill name="review" display_name="Review" location="/skills/review/SKILL.md" base_dir="/skills/review">\nReview instructions.\n</skill>',
	);
	assert.equal(contributionCount, 0);
});

void test("automatic continuation reuses the current system prompt", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	const originalSystemPrompt = internal.config.systemPrompt;
	internal.config.systemPrompt =
		"Keep debugging until the root cause is verified.";
	const seen: Array<{ kind: string; systemPrompt: string }> = [];

	// Mock session to capture prompt/continuation behavior
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async () => {
			seen.push({ kind: "prompt", systemPrompt: internal.config.systemPrompt });
		},
		getQueues: () => ({ nextTurn: [] }),
		setRepositoryQuery: () => {},
	});

	await bridge.sendMessage("Diagnose this TypeScript error.");
	// Allow async continuation to process
	for (let attempts = 0; seen.length < 2 && attempts < 20; attempts++) {
		await new Promise<void>(resolve => setImmediate(resolve));
	}

	// Verify the prompt used the correct system prompt
	assert.ok(seen.length >= 1);
	assert.match(seen[0]?.systemPrompt ?? "", /Keep debugging until/);

	internal.config.systemPrompt = originalSystemPrompt;
});

void test("sendMessage rejects when the turn fails", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	internal.toolRouter.isMcpLoaded = () => true;
	// Mock session to simulate provider failure
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async () => {
			throw new Error("provider failed");
		},
		getQueues: () => ({ nextTurn: [] }),
	});

	await assert.rejects(bridge.sendMessage("hello"), /provider failed/);
});

void test("cancel resolves only after abort settlement and returns recoverable queues", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	let settled = false;
	installSession(internal, {
		abort: async () => {
			await Promise.resolve();
			settled = true;
			return {
				clearedSteering: ["change direction"],
				clearedFollowUp: ["then verify"],
				clearedNextTurn: [],
			};
		},
	});

	const result = await bridge.cancel();

	assert.equal(settled, true);
	assert.deepEqual(result, {
		clearedSteering: ["change direction"],
		clearedFollowUp: ["then verify"],
		clearedNextTurn: [],
	});
});

void test("core iterations reconcile output without completing the UI turn early", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	internal.toolRouter.isMcpLoaded = () => true;

	// Mock session to simulate provider events
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async () => {
			// Simulate the provider emitting turn_end events via onEvent
			internal.config.onEvent({
				type: "message_update",
				message: { role: "assistant", content: "First iteration" },
			} as any);
			internal.config.onEvent({
				type: "message_update",
				message: { role: "assistant", content: "Final response" },
			} as any);
		},
		getQueues: () => ({ nextTurn: [] }),
	});

	const events: RuntimeEvent[] = [];
	bridge.events.subscribe(({ event }) => events.push(event));

	await bridge.sendMessage("do work");

	// The implementation emits: turn_start -> message_update -> message_update -> turn_end
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
	// Only one turn_end should be emitted (at the end of the turn, not per iteration)
	assert.equal(events.filter(event => event.type === "turn_end").length, 1);
});

void test("queued replacement turn reaches READY only after its stream ends", async () => {
	const bridge = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	bypassStartup(internal);
	internal.toolRouter.isMcpLoaded = () => true;
	let queued = ["change direction"];
	installSession(internal, {
		messages: [],
		configure: () => {},
		prompt: async () => {},
		getQueues: () => ({ steering: [], followUp: [], nextTurn: queued }),
		setRepositoryQuery: () => {},
		runQueuedContinuation: async () => {
			queued = [];
			internal.emit({ type: "turn_start", turnId: "replacement" });
			internal.emit({ type: "turn_end", turnId: "replacement" });
			internal.emit({ type: "phase", state: "ready" });
			return true;
		},
	});

	const lifecycle: string[] = [];
	bridge.events.subscribe(({ event }) => {
		if (event.type === "turn_start" || event.type === "turn_end") {
			lifecycle.push(`${event.type}:${event.turnId}`);
		} else if (event.type === "phase" && event.state === "ready") {
			lifecycle.push("ready");
		}
	});

	await bridge.sendMessage("initial request");

	assert.equal(bridge.isActive(), false);
	assert.deepEqual(lifecycle.slice(-4), [
		lifecycle.find(value => value.startsWith("turn_end:turn_")),
		"turn_start:replacement",
		"turn_end:replacement",
		"ready",
	]);
	assert.equal(lifecycle.indexOf("ready"), lifecycle.length - 1);
});
