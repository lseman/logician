import assert from "node:assert/strict";
import { test } from "node:test";
import { runAgentLoop } from "../core/agent-loop-runner.ts";
import { BackendError } from "../core/backend.ts";
import { OutputGuard } from "../core/output-guard.ts";
import { recordTaskStatus } from "../core/task-status-state.ts";
import type { AgentConfig, AgentEvent, Message, Tool } from "../core/types.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

const noop: Tool = {
	name: "noop",
	description: "does nothing",
	parameters: { type: "object", properties: {} },
	execute: async () => "ok",
};

// Minimal fixture standing in for the real task_status Tool (defined in
// @logician/agent-capabilities, which depends on this package — importing it here
// would cycle). The loop only reads getTaskStatus() state, not tool
// identity, so this is behaviorally equivalent for these tests.
const task_status: Tool = {
	name: "task_status",
	description: "test fixture",
	parameters: { type: "object", properties: {} },
	execute: async (args) => {
		const status = args.status as "done" | "blocked" | "needs_input" | "failed";
		const summary = typeof args.summary === "string" ? args.summary : "";
		recordTaskStatus({ status, summary, ts: Date.now() });
		return `Recorded: ${status}`;
	},
};

function makeConfig(overrides: Partial<AgentConfig> = {}): AgentConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [noop],
		...overrides,
	};
}

function user(content: string): Message {
	return { role: "user", content };
}

void test("runAgentLoop injects steering before the next assistant call", async () => {
	const backend = new FakeBackend([
		(messages) => {
			assert.ok(messages.some((m) => m.role === "user" && m.content === "steer"));
			return textResponse("done");
		},
	]);
	let drained = false;
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			getSteeringMessages: () => {
				if (drained) return [];
				drained = true;
				return [user("steer")];
			},
		},
		() => {},
	);
	assert.deepEqual(newMessages.map((m) => `${m.role}:${m.content ?? ""}`), [
		"user:prompt",
		"user:steer",
		"assistant:done",
	]);
});

void test("provider payload hooks preserve transport fields", async () => {
	const backend = new FakeBackend([
		async (_messages, options) => {
			const payload = await options.transformPayload?.({
				model: "fake",
				messages: [],
				stream: true,
				stream_options: { include_usage: true },
				max_tokens: 128,
			});
			assert.equal(payload?.stream, true);
			assert.deepEqual(payload?.stream_options, { include_usage: true });
			assert.equal(payload?.max_tokens, 128);
			assert.equal(payload?.custom, "kept");
			return textResponse("done");
		},
	]);
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			hooks: {
				beforeProviderPayload: ({ payload }) => ({
					payload: { ...payload, custom: "kept" },
				}),
			},
		},
		() => {},
	);
});

void test("runAgentLoop stamps events with monotonic sequence and timestamps", async () => {
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend: new FakeBackend([() => textResponse("done")]) },
		(event) => {
			events.push(event);
		},
	);
	assert.ok(events.length > 0);
	assert.deepEqual(
		events.map((event) => event.seq),
		Array.from({ length: events.length }, (_, index) => index + 1),
	);
	assert.ok(events.every((event) => typeof event.ts === "number" && event.ts > 0));
});

void test("runAgentLoop estimates context usage when provider usage is absent", async () => {
	const backend = new FakeBackend([() => textResponse("done")]);
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("a prompt with enough content to consume context")],
		{ ...makeConfig({ contextWindowTokens: 4096 }), backend },
		(event) => {
			events.push(event);
		},
	);
	const update = events.find((event) => event.type === "context_update");
	assert.ok(update && update.type === "context_update");
	assert.ok(update.tokens > 0);
	assert.equal(update.maxTokens, 4096);
});

void test("structured run outcomes take precedence and reset between runs", async () => {
	const structuredEvents: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [task_status] },
		[user("prompt")],
		{
			...makeConfig({ tools: [task_status] }),
			backend: new FakeBackend([
				() => ({
					content: "",
					toolCalls: [
						{
							id: "status",
							name: "task_status",
							arguments: JSON.stringify({
								status: "needs_input",
								summary: "Choose a target.",
							}),
						},
					],
					stopReason: "stop",
				}),
				() => textResponse("waiting"),
			]),
		},
		(event) => {
			structuredEvents.push(event);
		},
	);
	assert.ok(
		structuredEvents.some(
			(event) =>
				event.type === "run_outcome" &&
				event.status === "needs_input" &&
				event.source === "structured",
		),
	);

	const nextEvents: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("another prompt")],
		{ ...makeConfig(), backend: new FakeBackend([() => textResponse("done")]) },
		(event) => {
			nextEvents.push(event);
		},
	);
	assert.ok(
		nextEvents.some(
			(event) =>
				event.type === "run_outcome" &&
				event.status === "completed" &&
				event.source === "heuristic",
		),
	);
});

void test("provider retry stays within one iteration and ends only after success", async () => {
	const backend = new FakeBackend([
		() => {
			throw new BackendError({ category: "transient", message: "temporary" });
		},
		() => textResponse("recovered"),
	]);
	const events: AgentEvent[] = [];
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			maxIterations: 1,
			outputGuard: new OutputGuard({ retryBaseDelayMs: 0 }),
		},
		(event) => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 2);
	assert.equal(messages.at(-1)?.content, "recovered");
	assert.equal(events.filter((event) => event.type === "turn_start").length, 1);
	const retryEnds = events.filter((event) => event.type === "auto_retry_end");
	assert.deepEqual(retryEnds.map((event) => event.success), [true]);
	assert.equal(events.filter((event) => event.type === "error").length, 0);
});

void test("context-full retry compacts and publishes the live transcript", async () => {
	const history: Message[] = Array.from({ length: 12 }, (_, index) =>
		user(`old message ${index} ${"x".repeat(2000)}`),
	);
	let compacted: Message[] | undefined;
	const backend = new FakeBackend([
		() => {
			throw new BackendError({ category: "context_full", message: "too long" });
		},
		(messages) => {
			assert.ok(messages.length < history.length + 2);
			return textResponse("recovered");
		},
	]);
	await runAgentLoop(
		{ systemPrompt: "test", messages: history, tools: [noop] },
		[user("current prompt")],
		{
			...makeConfig({ contextWindowTokens: 4096 }),
			backend,
			outputGuard: new OutputGuard(),
			onContextCompacted: (messages) => {
				compacted = messages;
			},
		},
		() => {},
	);
	assert.ok(compacted);
	assert.ok(compacted.some((message) => String(message.content).includes("context-compaction")));
	assert.equal(compacted.at(-1)?.content, "recovered");
});

void test("aborting retry backoff prevents another provider request", async () => {
	const controller = new AbortController();
	const backend = new FakeBackend([
		() => {
			throw new BackendError({
				category: "rate_limit",
				message: "busy",
				retryAfterMs: 10_000,
			});
		},
		() => textResponse("must not run"),
	]);
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			signal: controller.signal,
			outputGuard: new OutputGuard(),
		},
		(event) => {
			if (event.type === "auto_retry_start") controller.abort();
		},
	);
	assert.equal(backend.calls, 1);
});

void test("runAgentLoop processes follow-up messages after a stop", async () => {
	const backend = new FakeBackend([
		() => textResponse("first"),
		(messages) => {
			assert.ok(messages.some((m) => m.role === "user" && m.content === "follow"));
			return textResponse("second");
		},
	]);
	let followUps = 0;
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			getFollowUpMessages: () => {
				if (followUps++ === 0) return [user("follow")];
				return [];
			},
		},
		() => {},
	);
	assert.deepEqual(newMessages.map((m) => `${m.role}:${m.content ?? ""}`), [
		"user:prompt",
		"assistant:first",
		"user:follow",
		"assistant:second",
	]);
});

void test("runAgentLoop executes a tool batch and returns ordered tool results", async () => {
	const tool: Tool = {
		name: "echo",
		description: "echoes",
		parameters: { type: "object", properties: { text: { type: "string" } } },
		execute: async (args) => `echo:${String(args.text)}`,
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{ id: "a", name: "echo", arguments: JSON.stringify({ text: "one" }) },
				{ id: "b", name: "echo", arguments: JSON.stringify({ text: "two" }) },
			],
			stopReason: "stop",
		}),
		() => textResponse("final"),
	]);
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{ ...makeConfig({ tools: [tool] }), backend },
		() => {},
	);
	assert.deepEqual(
		newMessages.filter((m) => m.role === "tool").map((m) => `${m.tool_call_id}:${m.content ?? ""}`),
		["a:echo:one", "b:echo:two"],
	);
});

void test("runAgentLoop executes independent tool calls in parallel and preserves result order", async () => {
	const completions: string[] = [];
	const tool: Tool = {
		name: "wait",
		description: "waits",
		parameters: { type: "object", properties: {} },
		execute: async (args) => {
			const name = String(args.name);
			await new Promise((resolve) => setTimeout(resolve, Number(args.delay)));
			completions.push(name);
			return name;
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{ id: "slow", name: "wait", arguments: JSON.stringify({ name: "slow", delay: 20 }) },
				{ id: "fast", name: "wait", arguments: JSON.stringify({ name: "fast", delay: 0 }) },
			],
			stopReason: "stop",
		}),
		() => textResponse("done"),
	]);
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{ ...makeConfig({ tools: [tool], toolExecution: "parallel" }), backend },
		() => {},
	);
	assert.deepEqual(completions, ["fast", "slow"]);
	assert.deepEqual(messages.filter((m) => m.role === "tool").map((m) => m.tool_call_id), ["slow", "fast"]);
});

void test("cancelled sequential batches produce a result for every tool call", async () => {
	const controller = new AbortController();
	const calls: string[] = [];
	const tools: Tool[] = ["one", "two", "three"].map((name, index) => ({
		name,
		description: name,
		parameters: { type: "object", properties: {} },
		execute: async () => {
			calls.push(name);
			if (index === 0) controller.abort();
			return name;
		},
	}));
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools },
		[user("prompt")],
		{
			...makeConfig({ tools, toolExecution: "sequential" }),
			backend: new FakeBackend([
				() => ({
					content: "",
					toolCalls: tools.map((tool, index) => ({
						id: `call_${index}`,
						name: tool.name,
						arguments: "{}",
					})),
					stopReason: "stop",
				}),
			]),
			signal: controller.signal,
			maxIterations: 1,
		},
		() => {},
	);

	const results = messages.filter((message) => message.role === "tool");
	assert.equal(results.length, 3);
	assert.deepEqual(calls, ["one"]);
	assert.match(String(results[1].content), /cancelled/);
	assert.match(String(results[2].content), /cancelled/);
});

void test("parallel tool batches complete deterministic preflight before execution", async () => {
	const order: string[] = [];
	const tool: Tool = {
		name: "planned",
		description: "planned tool",
		parameters: { type: "object", properties: {} },
		execute: async (args) => {
			order.push(`execute:${String(args.id)}`);
			return "ok";
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{ id: "one", name: "planned", arguments: JSON.stringify({ id: 1 }) },
				{ id: "two", name: "planned", arguments: JSON.stringify({ id: 2 }) },
			],
			stopReason: "stop",
		}),
		() => textResponse("done"),
	]);
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{
			...makeConfig({ tools: [tool], toolExecution: "parallel" }),
			backend,
			hooks: {
				beforeToolCall: ({ args }) => {
					order.push(`preflight:${String(args.id)}`);
					return undefined;
				},
			},
		},
		() => {},
	);
	assert.deepEqual(order.slice(0, 2), ["preflight:1", "preflight:2"]);
	assert.deepEqual(new Set(order.slice(2)), new Set(["execute:1", "execute:2"]));
});

void test("tool progress and thrown failures produce accurate lifecycle events", async () => {
	const tool: Tool = {
		name: "progress",
		description: "progress tool",
		parameters: { type: "object", properties: {} },
		execute: async (args, ctx) => {
			ctx.onUpdate?.("halfway");
			if (args.fail) throw new Error("boom");
			return "ok";
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{ id: "ok", name: "progress", arguments: "{}" },
				{ id: "bad", name: "progress", arguments: JSON.stringify({ fail: true }) },
			],
			stopReason: "stop",
		}),
		() => textResponse("done"),
	]);
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{ ...makeConfig({ tools: [tool], toolExecution: "parallel" }), backend },
		(event) => {
			events.push(event);
		},
	);
	assert.equal(events.filter((event) => event.type === "tool_call_update").length, 2);
	assert.ok(
		events.some(
			(event) =>
				event.type === "tool_call_end" &&
				event.toolCallId === "bad" &&
				event.isError === true,
		),
	);
});

void test("runAgentLoop does not execute tool calls from a length-truncated response", async () => {
	let executions = 0;
	const tool: Tool = {
		name: "write",
		description: "writes",
		parameters: { type: "object", properties: {} },
		execute: async () => {
			executions++;
			return "written";
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [{ id: "cut", name: "write", arguments: "{\"path\":\"partial" }],
			stopReason: "length",
		}),
		() => textResponse("done"),
	]);
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{ ...makeConfig({ tools: [tool] }), backend },
		() => {},
	);
	assert.equal(executions, 0);
	assert.match(String(messages.find((m) => m.role === "tool")?.content), /not executed.*truncated/i);
});

void test("runAgentLoop reports provider errors as terminal new messages", async () => {
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => ({ content: null, toolCalls: [], stopReason: "error", errorMessage: "boom" }),
	]);
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend },
		(event) => {
			events.push(event);
		},
	);
	assert.equal(newMessages.at(-1)?.role, "assistant");
	assert.ok(events.some((event) => event.type === "error" && event.message === "boom"));
	assert.ok(events.some((event) => event.type === "agent_end"));
});

void test("runAgentLoop stops before provider call when aborted", async () => {
	const controller = new AbortController();
	controller.abort();
	const backend = new FakeBackend([() => textResponse("should not run")]);
	const events: AgentEvent[] = [];
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend, signal: controller.signal },
		(event) => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 0);
	assert.deepEqual(newMessages.map((m) => `${m.role}:${m.content ?? ""}`), ["user:prompt"]);
	assert.ok(events.some((event) => event.type === "error" && event.message.includes("aborted")));
});
