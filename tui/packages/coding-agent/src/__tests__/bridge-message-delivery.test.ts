import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
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

void test("startup state includes persisted project observational memory", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-startup-memory-"));
	const memoryDir = join(cwd, ".logician", "observational-memory");
	mkdirSync(memoryDir, { recursive: true });
	writeFileSync(
		join(memoryDir, "memory.json"),
		JSON.stringify({
			type: "om.folded",
			version: 1,
			fullFold: true,
			observations: [
				{
					id: "aaaaaaaaaaaa",
					content: "Keep observational memory in the project directory.",
					timestamp: "2026-07-24T00:00:00.000Z",
					relevance: "high",
					sourceEntryIds: ["bbbbbbbbbbbb"],
					tokenCount: 8,
				},
			],
			reflections: [
				{
					id: "cccccccccccc",
					content: "Project memory should be visible when Logician opens.",
					supportingObservationIds: ["aaaaaaaaaaaa"],
					tokenCount: 9,
				},
			],
			droppedObservationIds: [],
		}),
	);

	try {
		const bridge = new AgentCoreBridge({
			baseUrl: "http://127.0.0.1:1",
			model: "test",
			cwd,
			runtimeHooksEnabled: false,
			mcpEager: false,
		});
		const internal = bridge as unknown as Record<string, unknown>;
		internal.startupHooksRan = true;

		const state = await bridge.init();
		assert.deepEqual(state.observational_memory, {
			observation_count: 1,
			active_observation_count: 1,
			reflection_count: 1,
			dropped_count: 0,
			observations: [
				{
					id: "aaaaaaaaaaaa",
					content: "Keep observational memory in the project directory.",
					relevance: "high",
					timestamp: "2026-07-24T00:00:00.000Z",
				},
			],
			reflections: [
				{
					id: "cccccccccccc",
					content: "Project memory should be visible when Logician opens.",
				},
			],
			diagnostics: {},
		});
	} finally {
		rmSync(cwd, { recursive: true, force: true });
	}
});

void test("observational memory can be disabled", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
		observationalMemoryEnabled: false,
	});
	const internal = bridge as unknown as Record<string, unknown>;
	internal.startupHooksRan = true;

	const state = await bridge.init();
	assert.equal(state.observational_memory, null);
	assert.ok(Array.isArray(state.tools));
	assert.ok(!state.tools.includes("memory_search"));
	assert.ok(!state.tools.includes("recall"));
	assert.equal(
		bridge.getSettingsData().observationalMemoryEnabled,
		false,
	);
	assert.match(bridge.getSettingsText(), /Observational memory: off/);
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

void test("setObservationalMemoryEnabled toggles the feature at runtime", async () => {
	const bridge = new AgentCoreBridge({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
		mcpEager: false,
		observationalMemoryEnabled: false,
	});
	const internal = bridge as unknown as Record<string, any>;
	internal.startupHooksRan = true;

	assert.equal(bridge.getSettingsData().observationalMemoryEnabled, false);
	assert.match(bridge.getSettingsText(), /Observational memory: off/);

	bridge.setObservationalMemoryEnabled(true);
	assert.equal(bridge.getSettingsData().observationalMemoryEnabled, true);
	assert.match(bridge.getSettingsText(), /Observational memory: on/);

	bridge.setObservationalMemoryEnabled(false);
	assert.equal(bridge.getSettingsData().observationalMemoryEnabled, false);
	assert.match(bridge.getSettingsText(), /Observational memory: off/);
});
