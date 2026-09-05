import { test } from "bun:test";
import assert from "node:assert/strict";
import type {
	AgentConfig,
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "@logician/log-core";
import {
	BUILTIN_AGENTS,
	createSpawnAgentsTool,
	createSpawnAgentTool,
	createSubagentConcurrencyLimiter,
} from "../../capabilities/delegation/definitions.ts";
import { createHubMessageBus } from "../../capabilities/delegation/hub.ts";
import { hubSendTool, hubWaitTool, hubJobsTool, hubInboxTool } from "../../capabilities/delegation/hub-tools.ts";

// ── Fake backend ──────────────────────────────────────────────────────────────

class FakeBackend implements LLMBackend {
	readonly model = "fake";
	private readonly responses: LLMResponse[];
	constructor(responses: LLMResponse[]) {
		this.responses = [...responses];
	}
	withModel(): LLMBackend {
		return this;
	}
	async generate(
		_messages: Record<string, unknown>[],
		_options?: GenerateOptions,
	): Promise<LLMResponse> {
		const resp = this.responses.shift() ?? { content: "done", toolCalls: [], stopReason: "stop" };
		return resp;
	}
}

const baseConfig: AgentConfig = {
	baseUrl: "http://localhost:11434",
	model: "fake",
	cwd: process.cwd(),
	temperature: 0,
	maxTokens: 8192,
	contextWindowTokens: 131072,
	systemPrompt: "",
	tools: [],
	toolExecution: "sequential",
	permissions: undefined,
	allowedPaths: [],
	allowAllPaths: true,
	hooks: {},
	thinkingLevel: "off" as const,
	autoRetryEnabled: false,
	maxRetries: 2,
	turnTimeoutMs: 0,
	webSearch: { baseUrl: "http://localhost:11434" },
	truncation: { subagentResultMaxChars: 8192 },
	runBudget: undefined,
	maxTotalTokens: undefined,
	runtimeHooksEnabled: true,
	continuationEnabled: false,
};

// ── Helper: build minimal spawn_agents deps with a hub ────────────────────────

function makeDeps(hub: ReturnType<typeof createHubMessageBus>) {
	return {
		config: () => ({ ...baseConfig, tools: [] }),
		backend: new FakeBackend([]),
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
		concurrencyLimiter: createSubagentConcurrencyLimiter(2),
		hub,
	};
}

// ── Hub registration via spawn_agents ─────────────────────────────────────────

test("spawn_agents registers agents in the hub", async () => {
	const hub = createHubMessageBus();
	const deps = makeDeps(hub);
	const tool = createSpawnAgentsTool(deps);

	await tool.execute(
		{ tasks: [{ task: "Task A" }, { task: "Task B" }] },
		{ onUpdate: () => {} },
	);

	// Check that agents were registered by querying jobs().
	const jobs = hub.jobs();
	assert.ok(jobs.length >= 2, `Expected >= 2 registered agents, got ${jobs.length}`);
	for (const job of jobs) {
		assert.ok(job.id.startsWith("agent_"), `Agent ID should start with "agent_": ${job.id}`);
		assert.equal(job.agent, "general");
		assert.ok(job.status === "running" || job.status === "completed", `Unexpected status: ${job.status}`);
	}
});

// ── Hub completion ────────────────────────────────────────────────────────────

test("hub.complete is called with correct status on subagent exit", async () => {
	const hub = createHubMessageBus();
	const deps = makeDeps(hub);
	const tool = createSpawnAgentsTool(deps);

	await tool.execute(
		{ tasks: [{ task: "Task A" }, { task: "Task B" }] },
		{ onUpdate: () => {} },
	);

	const jobs = hub.jobs();
	const completed = jobs.filter(j => j.status === "completed");
	assert.ok(completed.length >= 2, `Expected >= 2 completed agents, got ${completed.length}`);
});

// ── Hub tools availability ────────────────────────────────────────────────────

test("hub tools have correct names and signatures", async () => {
	const hub = createHubMessageBus();
	const hubTools = [
		hubSendTool({ hub, agentId: "test" }),
		hubWaitTool({ hub, agentId: "test" }),
		hubJobsTool({ hub, agentId: "test" }),
		hubInboxTool({ hub, agentId: "test" }),
	];

	assert.equal(hubTools[0].name, "hub_send");
	assert.equal(hubTools[1].name, "hub_wait");
	assert.equal(hubTools[2].name, "hub_jobs");
	assert.equal(hubTools[3].name, "hub_inbox");

	for (const t of hubTools) {
		assert.ok(t.parameters, `${t.name} has parameters`);
		assert.ok(typeof t.execute === "function", `${t.name} has execute`);
	}
});

// ── Hub send / receive ────────────────────────────────────────────────────────

test("hub-send delivers messages to another agent", async () => {
	const hub = createHubMessageBus();
	const senderId = "agent_sender";
	const receiverId = "agent_receiver";

	hub.register(senderId, { id: senderId, agent: "general", task: "Send", status: "running" });
	hub.register(receiverId, { id: receiverId, agent: "general", task: "Receive", status: "running" });

	const sendTool = hubSendTool({ hub, agentId: senderId });
	const result = await sendTool.execute(
		{ to: receiverId, body: "hello from sender" },
		{ onUpdate: () => {} },
	);

	if (typeof result === "string") {
		assert.equal(result, "Message sent.");
	} else {
		assert.ok(
			result.content.includes("Message sent"),
			`Expected "Message sent" in result: ${result.content}`,
		);
	}

	const inboxTool = hubInboxTool({ hub, agentId: receiverId });
	const inboxResult = await inboxTool.execute({}, { onUpdate: () => {} });

	if (typeof inboxResult === "string") {
		assert.ok(inboxResult.includes("hello from sender"), `Inbox should contain sent message: ${inboxResult}`);
	} else {
		assert.ok(
			inboxResult.content.includes("hello from sender"),
			`Inbox content should contain sent message: ${inboxResult.content}`,
		);
	}
});

// ── Hub jobs ──────────────────────────────────────────────────────────────────

test("hub-jobs returns registered agents", async () => {
	const hub = createHubMessageBus();
	const agentId = "agent_0";
	hub.register(agentId, { id: agentId, agent: "general", task: "Work", status: "running" });

	const jobsTool = hubJobsTool({ hub, agentId: "test" });
	const result = await jobsTool.execute({}, { onUpdate: () => {} });

	if (typeof result === "string") {
		assert.ok(result.includes(agentId), `Jobs output should include ${agentId}: ${result}`);
	} else {
		assert.ok(
			result.content.includes(agentId),
			`Jobs content should include ${agentId}: ${result.content}`,
		);
	}
});

// ── Hub wait ──────────────────────────────────────────────────────────────────

test("hub-wait returns messages after target completes", async () => {
	const hub = createHubMessageBus();
	const targetId = "agent_target";

	hub.register(targetId, { id: targetId, agent: "general", task: "Work", status: "running" });

	const waitTool = hubWaitTool({ hub, agentId: "waiter" });
	// Wait returns immediately with an empty array when no messages are pending.
	const result = await waitTool.execute({ handles: [targetId], timeout_ms: 100 }, { onUpdate: () => {} });

	// Result should indicate the target (empty or with messages).
	if (typeof result === "string") {
		assert.ok(result.length > 0, `Wait result should not be empty: ${result}`);
	} else {
		assert.ok(result.content.includes(targetId) || result.content.includes("[]"),
			`Wait content should mention target or empty: ${result.content}`);
	}
});

// ── spawn_agent single tool works with hub ────────────────────────────────────

test("spawn_agent works when hub is present", async () => {
	const hub = createHubMessageBus();
	const deps = {
		config: () => ({ ...baseConfig, tools: [] }),
		backend: new FakeBackend([]),
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
		concurrencyLimiter: createSubagentConcurrencyLimiter(1),
		hub,
	};
	const tool = createSpawnAgentTool(deps);

	const result = await tool.execute(
		{ task: "Single task" },
		{ onUpdate: () => {} },
	);

	assert.ok(
		typeof result !== "string" || !result.includes("Error:"),
		`spawn_agent should not fail due to hub: ${typeof result === "string" ? result : "structured"}`,
	);

	// Hub should have received registration and completion for the spawned agent.
	const jobs = hub.jobs();
	assert.ok(jobs.length >= 1, `Hub should have registered agents, got ${jobs.length}`);
});
