import { test } from "bun:test";
import assert from "node:assert/strict";
import { AgentCoreBridge } from "../application/agent-bridge.ts";
import type { RuntimeEvent } from "../runtime/events.ts";

void test("direct /spawn records task and result in harness history", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
	});
	const internal = bridge as unknown as {
		toolRouter: {
			getDefaultTools: () => Array<{
				name: string;
				execute: (
					args: Record<string, unknown>,
					ctx: { onUpdate?: (delta: string) => void },
				) => Promise<
					string | { content: string; isError?: boolean; details?: unknown }
				>;
			}>;
		};
		ensureHarness: () => {
			messages: unknown[];
			appendMessages: (m: unknown[]) => void;
		};
		subagents: { injected: boolean };
	};
	// defaultTools now lives on ToolRouter; stub its getter for this test's fake spawn_agent.
	internal.toolRouter.getDefaultTools = () => [
		{
			name: "spawn_agent",
			execute: async () => ({
				content: "Found README.md with install steps.",
				isError: false,
				details: { agent: "general", status: "completed" },
			}),
		},
	];
	// SubagentCoordinator's `injected` flag lives on the coordinator instance now.
	internal.subagents.injected = true;

	const recorded: unknown[][] = [];
	const fakeMessages: unknown[] = [];
	internal.ensureHarness = () => ({
		messages: fakeMessages,
		appendMessages: (msgs: unknown[]) => {
			recorded.push(msgs);
			fakeMessages.push(...msgs);
		},
	});
	// Patch the private method path: spawnAgentDirectly uses this.ensureHarness
	// which is a real method — override via prototype-style assignment on instance.
	(bridge as unknown as { ensureHarness: () => unknown }).ensureHarness =
		() => ({
			messages: fakeMessages,
			appendMessages: (msgs: unknown[]) => {
				recorded.push(msgs);
				fakeMessages.push(...msgs);
			},
		});

	bridge.spawnAgentDirectly("check readme.md");
	// Wait for the async execute().then path
	await new Promise(r => setTimeout(r, 20));

	assert.equal(recorded.length, 1);
	const msgs = recorded[0] as Array<{
		role: string;
		content: string | null;
		tool_calls?: Array<{ name: string; arguments: string }>;
		tool_call_id?: string;
		name?: string;
	}>;
	assert.equal(msgs.length, 3);
	assert.equal(msgs[0].role, "user");
	assert.equal(msgs[0].content, "/spawn check readme.md");
	assert.equal(msgs[1].role, "assistant");
	assert.equal(msgs[1].tool_calls?.[0]?.name, "spawn_agent");
	assert.match(msgs[1].tool_calls?.[0]?.arguments ?? "", /check readme\.md/);
	assert.equal(msgs[2].role, "tool");
	assert.equal(msgs[2].name, "spawn_agent");
	assert.equal(msgs[2].content, "Found README.md with install steps.");
});

void test("startup state reports the registered web_search capability", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
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
		"../tools/sandbox.ts"
	);
	const prev = getDefaultSandboxProfile();
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
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
		mcpEager: false,
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
		mcpEager: true,
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
		mcpEager: false,
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

	await internal.toolRouter.loadMcpToolsOnce();

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
		mcpEager: false,
		autoStartMcp: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.toolRouter.getSkillsContext = () =>
		"<available-skills>skill catalog</available-skills>";
	internal.toolRouter.mcpManager = {
		load: async () => ({
			tools: [],
			servers: 0,
			errors: ["github: authentication failed"],
		}),
	};

	await internal.toolRouter.loadMcpToolsOnce();
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
		mcpEager: false,
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
		mcpEager: false,
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

void test("/context renders the latest explicit task state", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.currentTaskState = {
		objective: "Fix authentication retries",
		phase: "verify",
		hypotheses: ["The retry cap is ignored"],
		evidence: [
			{
				kind: "change",
				tool: "edit_file",
				summary: "Updated retry cap",
				iteration: 2,
			},
		],
		changedFiles: ["src/auth.ts"],
		verification: [
			{ command: "bun test auth.test.ts", passed: true, summary: "4 pass" },
		],
		blockers: [],
		toolCalls: 3,
		toolFailures: 0,
	};
	internal.harness = { messages: [], getMemoryPrompt: () => "" };

	const context = bridge.getContext();

	assert.match(context, /<task_state>/);
	assert.match(context, /objective: Fix authentication retries/);
	assert.match(context, /phase: verify/);
	assert.match(context, /src\/auth\.ts/);
	assert.match(context, /verification: pass bun test auth\.test\.ts/);
	assert.match(context, /\[change\] edit_file: Updated retry cap/);
	assert.match(context, /<\/task_state>/);
	assert.doesNotMatch(context, /## Task state/);
	assert.match(context, /## Conversation[\s\S]*\[SYSTEM\]\n<task_state>/);
	assert.ok(
		context.indexOf("[SYSTEM]\n<task_state>") >
			context.indexOf("## Conversation"),
		"task state should appear in the final provider-message position",
	);
});

void test("/context omits an empty orient task state", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
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

void test("/context renders request-time memory injection", () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
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
	assert.match(context, /\[SYSTEM\]\n# Agent Context/);
	assert.match(context, new RegExp(memory.id));
	assert.match(context, /Authentication retry policy/);
	assert.match(context, /Call `memory_get` once/);
	bridge.getMemoryStore()?.close();
});

void test("matching skills are injected only for the selected turn", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.isMcpLoaded = () => true;
	internal.toolRouter.getLoadedSkills = () => [
		{
			name: "typescript-code-review",
			displayName: "TypeScript Code Review",
			description: "Review TypeScript code for correctness and type safety.",
			content: "Check runtime correctness before maintainability.",
			filePath: "/skills/typescript-code-review/SKILL.md",
			baseDir: "/skills/typescript-code-review",
			slashName: "typescript-code-review",
			disableModelInvocation: false,
			source: "user",
			triggers: ["TypeScript code review", "review TS"],
		},
	];
	const originalPrompt = internal.config.systemPrompt;
	let activePrompt = originalPrompt;
	let promptSeenByTurn = "";
	internal.harness = {
		messages: [],
		setSystemPrompt: (value: string) => {
			activePrompt = value;
		},
		prompt: async () => {
			promptSeenByTurn = activePrompt;
		},
	};

	await bridge.sendMessage("Review this TypeScript service.");

	assert.match(promptSeenByTurn, /<activated-skills>/);
	assert.match(promptSeenByTurn, /Check runtime correctness/);
	assert.equal(activePrompt, originalPrompt);
});

void test("automatic continuation retains the active skill", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.isMcpLoaded = () => true;
	internal.toolRouter.getLoadedSkills = () => [
		{
			name: "typescript-debugging",
			displayName: "TypeScript Debugging",
			description: "Diagnose TypeScript errors.",
			content: "Keep debugging until the root cause is verified.",
			filePath: "/skills/typescript-debugging/SKILL.md",
			baseDir: "/skills/typescript-debugging",
			slashName: "typescript-debugging",
			disableModelInvocation: false,
			source: "user",
			triggers: ["TypeScript error"],
		},
	];
	let activePrompt = internal.config.systemPrompt;
	const seen: Array<{ kind: string; systemPrompt: string }> = [];
	internal.harness = {
		messages: [],
		setSystemPrompt: (value: string) => {
			activePrompt = value;
		},
		prompt: async () => {
			seen.push({ kind: "prompt", systemPrompt: activePrompt });
			if (seen.length === 1) internal.pendingAutoContinue = true;
		},
		continueWithNextTurn: async () => {
			seen.push({ kind: "continue", systemPrompt: activePrompt });
		},
		requestContinuation: () => ({
			action: "continue",
			state: { continuationRuns: 1 },
		}),
		failRun: () => {},
	};

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
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.isMcpLoaded = () => true;
	internal.harness = {
		messages: [],
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
		mcpEager: false,
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
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.toolRouter.isMcpLoaded = () => true;
	internal.harness = {
		messages: [],
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
