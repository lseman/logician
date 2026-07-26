import assert from "node:assert/strict";
import { test } from "node:test";
import type { ParsedBridgeEvent } from "../runtime/events.ts";
import { AgentCoreBridge } from "../runtime/bridge.ts";

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
	});
	const internal = bridge as unknown as Record<string, unknown>;
	internal.startupHooksRan = true;
	internal.mcpLoadPromise = new Promise<void>(() => {});
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

void test("MCP load failures are injected into the system prompt", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.mcpManager = {
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
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.skillsContext = "<available-skills>skill catalog</available-skills>";
	internal.mcpManager = {
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

void test("matching skills are injected only for the selected turn", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;
	internal.mcpLoaded = true;
	internal.loadedSkills = [
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
	internal.mcpLoaded = true;
	internal.loadedSkills = [
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
	const seen: Array<{ message: string; systemPrompt: string }> = [];
	internal.harness = {
		messages: [],
		setSystemPrompt: (value: string) => {
			activePrompt = value;
		},
		prompt: async (message: string) => {
			seen.push({ message, systemPrompt: activePrompt });
			if (seen.length === 1) internal.pendingAutoContinue = true;
		},
	};

	await bridge.sendMessage("Diagnose this TypeScript error.");
	for (let attempts = 0; seen.length < 2 && attempts < 20; attempts++) {
		await new Promise<void>((resolve) => setImmediate(resolve));
	}

	assert.equal(seen[1]?.message, "continue");
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
	internal.mcpLoaded = true;
	internal.harness = {
		messages: [],
		prompt: async () => {
			throw new Error("provider failed");
		},
	};
	await assert.rejects(bridge.sendMessage("hello"), /provider failed/);
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
	internal.mcpLoaded = true;
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
	const events: ParsedBridgeEvent[] = [];
	bridge.on((event) => events.push(event));

	await bridge.sendMessage("do work");

	assert.deepEqual(
		events
			.filter((event) =>
				event.type === "turn_start" ||
				event.type === "message_update" ||
				event.type === "turn_end"
			)
			.map((event) => event.type),
		["turn_start", "message_update", "message_update", "turn_end"],
	);
	assert.equal(
		events.filter((event) => event.type === "turn_end").length,
		1,
	);
});
