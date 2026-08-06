import assert from "node:assert/strict";
import { test } from "node:test";
import type { AgentConfig, Tool } from "@logician/agent-core";
import type {
	GenerateOptions,
	LLMBackend,
	LLMResponse,
} from "@logician/agent-core/agent/backend.ts";
import {
	BUILTIN_AGENTS,
	createSpawnAgentsTool,
	createSpawnAgentTool,
	createSubagentConcurrencyLimiter,
} from "../delegation/definitions.ts";
import { runDelegatedAgent } from "../delegation/runtime.ts";
import { task_status } from "../tasks/task-status.ts";

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
		return (
			this.responses.shift() ?? {
				content: "done",
				toolCalls: [],
				stopReason: "stop",
			}
		);
	}
}

const baseConfig: AgentConfig = {
	baseUrl: "http://test",
	model: "fake",
	systemPrompt: "You are the specialist agent.",
	tools: [],
};

function report(status: "satisfied" | "failed", answer: string): LLMResponse {
	return {
		content: `${answer}\n\n\`\`\`acceptance-report\n{"criteriaSatisfied":[{"id":"criterion-1","status":"${status}","evidence":"checked"}]}\n\`\`\``,
		toolCalls: [],
		stopReason: "stop",
	};
}

void test("delegated contracts retry failed output and preserve a clean final result", async () => {
	const result = await runDelegatedAgent({
		task: "Produce the result",
		config: baseConfig,
		backend: new FakeBackend([
			report("failed", "incomplete"),
			report("satisfied", "corrected result"),
		]),
		tools: [],
		maxIterations: 4,
		contract: { expectedOutput: "a corrected result", maxValidationRetries: 1 },
		onEvent: () => {},
	});

	assert.equal(result.status, "completed");
	assert.equal(result.content, "corrected result");
	assert.equal(result.validationAttempts, 2);
	assert.equal(result.turns, 2);
});

void test("valid acceptance reports do not trigger redundant continuation nudges", async () => {
	let calls = 0;
	const responses = [
		{
			content: "",
			toolCalls: [{ id: "probe-1", name: "probe", arguments: "{}" }],
			stopReason: "stop" as const,
		},
		report("satisfied", "verified result"),
	];
	const backend: LLMBackend = {
		model: "fake",
		withModel() {
			return this;
		},
		async generate() {
			calls++;
			return responses.shift() ?? report("satisfied", "repeated result");
		},
	};
	const tools: Tool[] = [
		{
			name: "probe",
			description: "Inspect",
			parameters: { type: "object", properties: {} },
			execute: async () => "evidence",
		},
		{
			name: "task_status",
			description: "Conclude",
			parameters: { type: "object", properties: {} },
			execute: async () => "done",
		},
	];

	const result = await runDelegatedAgent({
		task: "Inspect and report",
		config: { ...baseConfig, tools, continuationEnabled: true },
		backend,
		tools,
		maxIterations: 4,
		contract: { successCriteria: ["Inspection has evidence"] },
		onEvent: () => {},
	});

	assert.equal(result.status, "completed");
	assert.equal(result.content, "verified result");
	assert.equal(calls, 2);
});

void test("a real task_status(done) call ends a delegated run in one turn instead of looping on continuation nudges", async () => {
	let calls = 0;
	const backend: LLMBackend = {
		model: "fake",
		withModel() {
			return this;
		},
		async generate() {
			calls++;
			return {
				content: "",
				toolCalls: [
					{
						id: "ts-1",
						name: "task_status",
						arguments: JSON.stringify({
							status: "done",
							summary: "Listed files.",
						}),
					},
				],
				stopReason: "stop",
			};
		},
	};

	const result = await runDelegatedAgent({
		task: "List files",
		config: { ...baseConfig, tools: [task_status], continuationEnabled: true },
		backend,
		tools: [task_status],
		maxIterations: 10,
		onEvent: () => {},
	});

	// Without the afterToolCall termination hook, the runner-level nudge
	// ("call task_status with the accurate status") re-prompts the model,
	// which just calls task_status(done) again — repeating until
	// maxIterations and surfacing a spurious "failed" status.
	assert.equal(calls, 1);
	assert.equal(result.turns, 1);
	assert.equal(result.status, "completed");
});

void test("delegated tool-call budgets are shared across the whole run", async () => {
	let executions = 0;
	const tool: Tool = {
		name: "probe",
		description: "probe",
		parameters: { type: "object", properties: {} },
		execute: async () => {
			executions++;
			return "ok";
		},
	};
	const toolCall = (id: string): LLMResponse => ({
		content: "",
		toolCalls: [{ id, name: "probe", arguments: "{}" }],
		stopReason: "stop",
	});
	const result = await runDelegatedAgent({
		task: "Probe twice",
		config: { ...baseConfig, tools: [tool] },
		backend: new FakeBackend([
			toolCall("one"),
			toolCall("two"),
			{
				content: "finished",
				toolCalls: [],
				stopReason: "stop",
			},
		]),
		tools: [tool],
		maxIterations: 3,
		budget: { maxToolCalls: 1 },
		onEvent: () => {},
	});

	assert.equal(executions, 1);
	assert.equal(result.toolCalls, 1);
	assert.equal(result.toolCallsByName.probe, 1);
	assert.equal(result.status, "failed");
});

void test("whole-task deadlines cancel a delegated run", async () => {
	const backend: LLMBackend = {
		model: "slow",
		withModel() {
			return this;
		},
		generate: async (_messages, options = {}) =>
			new Promise<LLMResponse>((_resolve, reject) => {
				const rejectAbort = () =>
					reject(new DOMException("timed out", "AbortError"));
				if (options.signal?.aborted) rejectAbort();
				else
					options.signal?.addEventListener("abort", rejectAbort, {
						once: true,
					});
			}),
	};
	const result = await runDelegatedAgent({
		task: "Never finishes",
		config: baseConfig,
		backend,
		tools: [],
		maxIterations: 3,
		budget: { timeoutMs: 10 },
		onEvent: () => {},
	});

	assert.equal(result.status, "cancelled");
	assert.ok(result.durationMs < 1_000);
});

void test("subagent progress callbacks emit deltas rather than accumulated prefixes", async () => {
	const backend: LLMBackend = {
		model: "streaming",
		withModel() {
			return this;
		},
		generate: async (_messages, options = {}) => {
			options.callbacks?.onTextStart?.();
			options.callbacks?.onDelta?.("I");
			options.callbacks?.onDelta?.(" finished.");
			options.callbacks?.onTextEnd?.();
			return { content: "Task complete.", toolCalls: [], stopReason: "stop" };
		},
	};
	const tool = createSpawnAgentTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend,
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});
	const updates: string[] = [];
	await tool.execute(
		{ task: "Complete it", agent: "general" },
		{ onUpdate: value => updates.push(value) },
	);

	assert.deepEqual(updates, ["I", " finished."]);
});

void test("spawn_agents honors maxParallelAgents and preserves its plural API", async () => {
	let active = 0;
	let peak = 0;
	const backend: LLMBackend = {
		model: "concurrent",
		withModel() {
			return this;
		},
		async generate(_messages, options = {}) {
			active++;
			peak = Math.max(peak, active);
			options.callbacks?.onDelta?.("working");
			await new Promise<void>(resolve => setImmediate(resolve));
			active--;
			return { content: "done", toolCalls: [], stopReason: "stop" };
		},
	};
	const emitted: Array<{ type: string; taskIndex?: number }> = [];
	const tool = createSpawnAgentsTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend,
		agents: () => BUILTIN_AGENTS,
		emit: event => emitted.push(event as { type: string; taskIndex?: number }),
		concurrencyLimiter: createSubagentConcurrencyLimiter(2),
	});
	const updates: string[] = [];

	assert.equal(tool.name, "spawn_agents");
	const result = await tool.execute(
		{
			tasks: Array.from({ length: 5 }, (_, index) => ({
				task: `Task ${index}`,
			})),
		},
		{ onUpdate: update => updates.push(update) },
	);

	assert.equal(peak, 2);
	if (typeof result === "string") {
		assert.fail(`Expected a structured batch result, got: ${result}`);
	}
	assert.equal(result.details?.total, 5);
	assert.equal(result.details?.completed, 5);
	assert.match(result.content, /## Subagent 1: general \(completed\)/);
	assert.match(result.content, /## Subagent 5: general \(completed\)/);
	assert.equal(result.content.match(/\bdone\b/g)?.length, 5);
	// Per-task lifecycle is now structured (subagent_start/subagent_end with
	// taskIndex), not a `▶/↳/✓` marker-string stream on onUpdate.
	const starts = emitted.filter(event => event.type === "subagent_start");
	const ends = emitted.filter(event => event.type === "subagent_end");
	assert.equal(starts.length, 5);
	assert.equal(ends.length, 5);
	assert.deepEqual(
		new Set(starts.map(event => event.taskIndex)),
		new Set([0, 1, 2, 3, 4]),
	);
	assert.deepEqual(
		new Set(ends.map(event => event.taskIndex)),
		new Set([0, 1, 2, 3, 4]),
	);
	// Per-task text streaming is carried via subagent_event/childChunks (see
	// the taskIndex assertions above), not the top-level tool's onUpdate —
	// spawn_agents no longer forwards child text deltas there.
	assert.equal(updates.length, 0);
});

void test("spawn_agents requires more than one task", async () => {
	const tool = createSpawnAgentsTool({
		config: () => ({ ...baseConfig, tools: [] }),
		backend: new FakeBackend([]),
		agents: () => BUILTIN_AGENTS,
		emit: () => {},
	});

	assert.equal(
		await tool.execute({ tasks: [{ task: "Use spawn_agent instead" }] }, {}),
		"Error: spawn_agents requires at least two tasks.",
	);
});

void test("subagent concurrency limits are isolated between sessions", async () => {
	let active = 0;
	let peak = 0;
	const backend: LLMBackend = {
		model: "session-isolation",
		withModel() {
			return this;
		},
		async generate() {
			active++;
			peak = Math.max(peak, active);
			await new Promise<void>(resolve => setImmediate(resolve));
			active--;
			return { content: "done", toolCalls: [], stopReason: "stop" };
		},
	};
	const createSessionTool = () =>
		createSpawnAgentsTool({
			config: () => ({ ...baseConfig, tools: [] }),
			backend,
			agents: () => BUILTIN_AGENTS,
			emit: () => {},
			concurrencyLimiter: createSubagentConcurrencyLimiter(1),
		});
	const tasks = {
		tasks: [{ task: "First" }, { task: "Second" }],
	};

	await Promise.all([
		createSessionTool().execute(tasks, {}),
		createSessionTool().execute(tasks, {}),
	]);

	assert.equal(peak, 2);
});
