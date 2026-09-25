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
import {
	createHubMessageBus,
	type HubMessageBus,
} from "../../capabilities/delegation/hub.ts";
import { createHubTool } from "../../capabilities/hub/hub-tool.ts";
import { defaultHub } from "../../capabilities/hub/process-manager.ts";

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
		const resp = this.responses.shift() ?? {
			content: "done",
			toolCalls: [],
			stopReason: "stop",
		};
		return resp;
	}
	async remote(_messages: Record<string, unknown>[]) {
		return { summary: "[remote compaction not configured]", preserveData: {} };
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

function makeDeps(hub: HubMessageBus) {
	return {
		config: () => ({ ...baseConfig, tools: [] }),
		backend: new FakeBackend([]),
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
		concurrencyLimiter: createSubagentConcurrencyLimiter(2),
		hub,
	};
}

// Builds a unified hub tool bound to the shared process manager + a message bus.
// Used by every bus-dependent test so the shared process manager is consistent.
const makeHubTool = (agentId: string, hub: HubMessageBus) =>
	createHubTool({ manager: defaultHub, bus: hub, agentId });

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
	assert.ok(
		jobs.length >= 2,
		`Expected >= 2 registered agents, got ${jobs.length}`,
	);
	for (const job of jobs) {
		assert.ok(
			job.id.startsWith("agent_"),
			`Agent ID should start with "agent_": ${job.id}`,
		);
		assert.equal(job.agent, "general");
		assert.ok(
			job.status === "running" || job.status === "completed",
			`Unexpected status: ${job.status}`,
		);
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
	assert.ok(
		completed.length >= 2,
		`Expected >= 2 completed agents, got ${completed.length}`,
	);
});

// ── Hub tools availability ────────────────────────────────────────────────────

test("unified hub tool exposes process and coordination ops", async () => {
	const hub = createHubMessageBus();

	const withBus = createHubTool({
		manager: defaultHub,
		bus: hub,
		agentId: "test",
	});
	assert.equal(withBus.name, "hub");
	assert.ok(withBus.parameters, "hub has parameters");
	assert.ok(typeof withBus.execute === "function", "hub has execute");

	const withBusParams = withBus.parameters as {
		properties: { op: { enum: string[] } };
	}; // Tool.parameters is an untyped schema; the op enum is what I set.
	const ops = withBusParams.properties.op.enum;
	assert.ok(ops.includes("start"));
	assert.ok(ops.includes("ps"));
	assert.ok(ops.includes("logs"));
	assert.ok(ops.includes("stop"));
	assert.ok(ops.includes("restart"));
	assert.ok(ops.includes("send"));
	assert.ok(ops.includes("wait"));
	assert.ok(ops.includes("describe"));
	// Coordination ops only exist when a message bus is wired in.
	assert.ok(ops.includes("jobs"));
	assert.ok(ops.includes("inbox"));
	assert.ok(!ops.includes("hub_send"), "no legacy tool name");

	const withoutBus = createHubTool({ manager: defaultHub });
	const withoutBusParams = withoutBus.parameters as {
		properties: { op: { enum: string[] } };
	}; // Tool.parameters is an untyped schema; the op enum is what I set.
	const opsNoBus = withoutBusParams.properties.op.enum;
	assert.ok(opsNoBus.includes("start"));
	assert.ok(opsNoBus.includes("ps"));
	assert.ok(!opsNoBus.includes("jobs"), "no jobs without a bus");
	assert.ok(!opsNoBus.includes("inbox"), "no inbox without a bus");
});

// ── Hub send / receive ────────────────────────────────────────────────────────

test("hub-send delivers messages to another agent", async () => {
	const hub = createHubMessageBus();
	const senderId = "agent_sender";
	const receiverId = "agent_receiver";

	hub.register(senderId, {
		id: senderId,
		agent: "general",
		task: "Send",
		status: "running",
	});
	hub.register(receiverId, {
		id: receiverId,
		agent: "general",
		task: "Receive",
		status: "running",
	});

	const sendTool = makeHubTool(senderId, hub);
	const result = await sendTool.execute(
		{ op: "send", to: receiverId, body: "hello from sender" },
		{ onUpdate: () => {} },
	);

	if (typeof result === "string") {
		assert.ok(
			result.includes("Message sent"),
			`Expected "Message sent": ${result}`,
		);
	} else {
		assert.ok(
			result.content.includes("Message sent"),
			`Expected "Message sent" in result: ${result.content}`,
		);
	}

	const inboxTool = makeHubTool(receiverId, hub);
	const inboxResult = await inboxTool.execute(
		{ op: "inbox" },
		{ onUpdate: () => {} },
	);

	if (typeof inboxResult === "string") {
		assert.ok(
			inboxResult.includes("hello from sender"),
			`Inbox should contain sent message: ${inboxResult}`,
		);
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
	hub.register(agentId, {
		id: agentId,
		agent: "general",
		task: "Work",
		status: "running",
	});

	const jobsTool = makeHubTool("test", hub);
	const result = await jobsTool.execute({ op: "jobs" }, { onUpdate: () => {} });

	if (typeof result === "string") {
		assert.ok(
			result.includes(agentId),
			`Jobs output should include ${agentId}: ${result}`,
		);
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

	hub.register(targetId, {
		id: targetId,
		agent: "general",
		task: "Work",
		status: "running",
	});

	const waitTool = makeHubTool("waiter", hub);
	// Wait returns immediately with an empty array when no messages are pending.
	const result = await waitTool.execute(
		{ op: "wait", handles: [targetId], timeout_ms: 100 },
		{ onUpdate: () => {} },
	);

	// Result should indicate the target (empty or with messages).
	if (typeof result === "string") {
		assert.ok(result.length > 0, `Wait result should not be empty: ${result}`);
	} else {
		assert.ok(
			result.content.includes(targetId) || result.content.includes("[]"),
			`Wait content should mention target or empty: ${result.content}`,
		);
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
		`spawn_agent should not fail due to hub: ${
			typeof result === "string" ? result : "structured"
		}`,
	);

	// Hub should have received registration and completion for the spawned agent.
	const jobs = hub.jobs();
	assert.ok(
		jobs.length >= 1,
		`Hub should have registered agents, got ${jobs.length}`,
	);
});
