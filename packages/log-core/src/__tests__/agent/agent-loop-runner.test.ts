import { test } from "bun:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	createSteeringInterruptReason,
	type RunAgentLoopConfig,
	runAgentLoop,
	STEERING_INTERRUPT_SUMMARY,
} from "../../runtime/execution/agent-loop-runner.ts";
import { OutputGuard } from "../../control/guards/output-guard.ts";
import { resolveExecutionPolicy } from "../../control/policy/execution-policy.ts";
import { BackendError } from "../../capabilities/provider/backend.ts";
import { PermissionManager } from "../../capabilities/tools/permissions.ts";
import type { AgentConfig } from "../../system/types/types-config.ts";
import type {
	AgentEvent,
	Message,
	Tool,
} from "../../system/types/types-messages.ts";
import { FakeBackend, textResponse } from "../fake-backend.ts";

const noop: Tool = {
	name: "noop",
	description: "does nothing",
	parameters: { type: "object", properties: {} },
	execute: async () => "ok",
};

// Legacy fixture retained only by skipped regression cases documenting the
// removed structured-conclusion protocol.
const _task_status: Tool = {
	name: "task_status",
	description: "legacy test fixture",
	parameters: { type: "object", properties: {} },
	execute: async () => "recorded",
};

function makeConfig(overrides: Partial<AgentConfig> = {}): RunAgentLoopConfig {
	return {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		executionProfile: "autonomous",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [noop],
		...overrides,
	} as unknown as RunAgentLoopConfig;
}

void test("the default execution profile is autonomous", () => {
	assert.deepEqual(resolveExecutionPolicy(undefined), {
		profile: "autonomous",
		embeddedPoliciesEnabled: true,
	});
});

void test("minimal profile disables embedded policies", () => {
	assert.deepEqual(resolveExecutionPolicy("minimal"), {
		profile: "minimal",
		embeddedPoliciesEnabled: false,
	});
});

void test("failed deterministic verification gets one bounded repair turn", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-verification-repair-"));
	const marker = join(cwd, "fixed");
	const fix: Tool = {
		name: "fix",
		description: "fix the fixture",
		parameters: { type: "object", properties: {} },
		execute: async () => {
			writeFileSync(marker, "ok");
			return "fixed";
		},
	};
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => textResponse("I think it is done."),
		messages => {
			assert.match(String(messages.at(-1)?.content), /verification-repair/);
			return {
				content: "Applying the repair.",
				toolCalls: [{ id: "fix-1", name: "fix", arguments: "{}" }],
				stopReason: "stop" as const,
			};
		},
		() =>
			textResponse(
				'```acceptance-report\n{"criteriaSatisfied":[],"commandsRun":[],"residualRisks":[]}\n```',
			),
	]);
	try {
		await runAgentLoop(
			{ systemPrompt: "test", messages: [], tools: [fix], cwd },
			[user("fix it")],
			{
				...makeConfig({
					cwd,
					tools: [fix],
					maxIterations: 4,
					acceptance: {
						verify: [{ id: "marker", command: "test -f fixed" }],
					},
				}),
				backend,
			},
			event => {
				events.push(event);
			},
		);
		assert.equal(existsSync(marker), true);
		assert.equal(backend.calls, 3);
		assert.ok(
			events.some(
				event =>
					event.type === "harness_intervention" &&
					event.kind === "verification",
			),
		);
		assert.ok(
			events.some(
				event =>
					event.type === "acceptance_complete" && event.status === "passed",
			),
		);
	} finally {
		rmSync(cwd, { recursive: true, force: true });
	}
});

void test("three consecutive permission denials pause autonomous execution", async () => {
	const bash: Tool = {
		name: "bash",
		description: "run",
		parameters: { type: "object", properties: {} },
		execute: async () => "should not execute",
	};
	const denied = () => ({
		content: "trying",
		toolCalls: [{ id: crypto.randomUUID(), name: "bash", arguments: "{}" }],
		stopReason: "stop" as const,
	});
	const backend = new FakeBackend([denied, denied, denied]);
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [bash] },
		[user("run it")],
		{
			...makeConfig({
				tools: [bash],
				permissions: new PermissionManager({ mode: "plan" }),
				maxIterations: 5,
			}),
			backend,
		},
		event => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 3);
	assert.ok(
		events.some(
			event => event.type === "agent_end" && event.status === "needs_input",
		),
	);
});

function user(content: string): Message {
	return { role: "user", content };
}

void test("runAgentLoop injects steering before the next assistant call", async () => {
	const backend = new FakeBackend([
		messages => {
			assert.ok(messages.some(m => m.role === "user" && m.content === "steer"));
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
			hooks: {
				getSteeringMessages: () => {
					if (drained) return [];
					drained = true;
					return [user("steer")];
				},
			},
		},
		() => {},
	);
	assert.deepEqual(
		newMessages.map(m => `${m.role}:${m.content ?? ""}`),
		["user:prompt", "user:steer", "assistant:done"],
	);
});

void test("auto inference adapts sampling params from the objective", async () => {
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		(_messages, options) => {
			assert.equal(options.temperature, 0.2);
			assert.equal(options.topP, 0.7);
			return textResponse("Analysis complete.");
		},
	]);
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("Review and diagnose this authentication failure")],
		{ ...makeConfig({ inferenceMode: "auto" }), backend },
		event => {
			events.push(event);
		},
	);
	const selection = events.find(
		event => event.type === "inference_mode_selected",
	);
	assert.ok(selection && selection.type === "inference_mode_selected");
	assert.equal(selection.effectiveMode, "analytical");
});

void test("auto inference escalates after repeated tool failures", async () => {
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => ({
			content: "trying the build",
			toolCalls: [{ id: "1", name: "noop", arguments: "{}" }],
			stopReason: "stop" as const,
		}),
		() => ({
			content: "retrying the build",
			toolCalls: [{ id: "2", name: "noop", arguments: "{}" }],
			stopReason: "stop" as const,
		}),
		(_messages, options) => {
			assert.equal(options.temperature, 0.6);
			return textResponse("Task complete.");
		},
	]);
	const toolWithFailures: Tool = {
		name: "noop",
		description: "fails twice then the loop stops asking",
		parameters: { type: "object", properties: {} },
		execute: async () => "error: build failed",
	};
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [toolWithFailures] },
		[user("fix the build")],
		{
			...makeConfig({ inferenceMode: "auto", tools: [toolWithFailures] }),
			backend,
		},
		event => {
			events.push(event);
		},
	);
	const selections = events.filter(
		event => event.type === "inference_mode_selected",
	);
	assert.ok(
		selections.some(
			event =>
				event.type === "inference_mode_selected" &&
				event.effectiveMode === "thinking-coding" &&
				event.reason.includes("repeated tool failures"),
		),
	);
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
		event => {
			events.push(event);
		},
	);
	assert.ok(events.length > 0);
	assert.deepEqual(
		events.map(event => event.seq),
		Array.from({ length: events.length }, (_, index) => index + 1),
	);
	assert.ok(
		events.every(event => typeof event.ts === "number" && event.ts > 0),
	);
});

void test("runAgentLoop estimates context usage when provider usage is absent", async () => {
	const backend = new FakeBackend([() => textResponse("done")]);
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("a prompt with enough content to consume context")],
		{ ...makeConfig({ contextWindowTokens: 4096 }), backend },
		event => {
			events.push(event);
		},
	);
	const update = events.find(event => event.type === "context_update");
	assert.ok(update && update.type === "context_update");
	assert.ok(update.tokens > 0);
	assert.equal(update.maxTokens, 4096);
	assert.equal(
		events.filter(event => event.type === "context_update").at(-1)
			?.cachedTokens,
		null,
	);
	assert.equal(
		events.filter(event => event.type === "context_update").at(-1)
			?.promptTokens,
		null,
	);
	assert.equal(
		events.filter(event => event.type === "context_update").at(-1)
			?.completionTokens,
		null,
	);
});

void test("runAgentLoop propagates provider cache reads through context_update", async () => {
	const backend = new FakeBackend([
		() => ({
			...textResponse("done"),
			usage: {
				promptTokens: 20_000,
				completionTokens: 50,
				totalTokens: 20_050,
				cachedTokens: 12_400,
			},
		}),
	]);
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig({ contextWindowTokens: 32_768 }), backend },
		event => {
			events.push(event);
		},
	);
	const updates = events.filter(event => event.type === "context_update");
	assert.equal(updates.at(-1)?.cachedTokens, 12_400);
	assert.equal(updates.at(-1)?.promptTokens, 20_000);
	assert.equal(updates.at(-1)?.completionTokens, 50);
});

void test("undeclared outcome after tool work completes cleanly", async () => {
	const events: AgentEvent[] = [];
	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("do work")],
		{
			...makeConfig({
				tools: [noop],
				hooks: {
					afterToolCall: () => ({ terminate: true }),
				},
			}),
			backend: new FakeBackend([
				() => ({
					content: "",
					toolCalls: [{ id: "1", name: "noop", arguments: "{}" }],
					stopReason: "stop",
				}),
			]),
		},
		event => {
			events.push(event);
		},
	);
	const outcome = events.find(e => e.type === "agent_end");
	assert.ok(outcome);
	assert.equal(outcome.status, "completed");
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
		event => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 2);
	assert.equal(messages.at(-1)?.content, "recovered");
	assert.equal(events.filter(event => event.type === "turn_start").length, 1);
	const retryEnds = events.filter(event => event.type === "agent_retry_end");
	assert.deepEqual(
		retryEnds.map(event => event.success),
		[true],
	);
	const retryInterventions = events.filter(
		event => event.type === "harness_intervention" && event.kind === "retry",
	);
	assert.equal(retryInterventions.length, 1);
	const retryIntervention = retryInterventions[0];
	assert.ok(retryIntervention?.type === "harness_intervention");
	assert.equal(retryIntervention.action, "recover");
	assert.equal(events.filter(event => event.type === "error").length, 0);
});

void test("context-full retry compacts and publishes the live transcript", async () => {
	const history: Message[] = Array.from({ length: 12 }, (_, index) =>
		user(`old message ${index} ${"x".repeat(2000)}`),
	);
	let compacted: Message[] | undefined;
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => {
			throw new BackendError({ category: "context_full", message: "too long" });
		},
		messages => {
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
			onContextCompacted: messages => {
				compacted = messages;
			},
		},
		event => {
			events.push(event);
		},
	);
	assert.ok(compacted);
	assert.ok(
		compacted.some(message =>
			String(message.content).includes("context-compaction"),
		),
	);
	assert.equal(compacted.at(-1)?.content, "recovered");
	assert.equal(
		events.filter(
			event =>
				event.type === "harness_intervention" && event.kind === "compaction",
		).length,
		1,
	);
});

void test("retry proceeds without delay after abort signal during retry_start", async () => {
	const controller = new AbortController();
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => {
			throw new BackendError({
				category: "rate_limit",
				message: "busy",
				retryAfterMs: 10_000,
			});
		},
		() => textResponse("recovered after retry"),
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
		event => {
			events.push(event);
			if (event.type === "agent_retry_start") controller.abort();
		},
	);
	// Without retry delay, the retry proceeds immediately
	assert.equal(backend.calls, 2);
	assert.ok(backend.calls === 2);
});

void test("aborting an in-flight provider request does not emit a fake retry", async () => {
	const controller = new AbortController();
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => {
			controller.abort();
			throw new DOMException("Operation aborted", "AbortError");
		},
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
		event => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 1);
	assert.equal(
		events.some(
			event =>
				event.type === "agent_retry_start" || event.type === "agent_retry_end",
		),
		false,
	);
	const outcome = events.find(event => event.type === "agent_end");
	assert.ok(outcome);
	assert.equal(outcome.status, "cancelled");
});

void test("steering interruption suppresses provider errors and retries", async () => {
	const controller = new AbortController();
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => {
			controller.abort(createSteeringInterruptReason());
			// Several provider clients normalize an aborted request to a generic error.
			throw new Error("Unknown error");
		},
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
		event => {
			events.push(event);
		},
	);
	assert.equal(
		events.some(
			event =>
				event.type === "error" ||
				event.type === "agent_retry_start" ||
				event.type === "agent_retry_end",
		),
		false,
	);
	assert.ok(
		events.some(
			event =>
				event.type === "agent_end" &&
				event.status === "cancelled" &&
				event.summary === STEERING_INTERRUPT_SUMMARY,
		),
	);
});

void test("runAgentLoop processes follow-up messages after a stop", async () => {
	const backend = new FakeBackend([
		() => textResponse("first"),
		messages => {
			assert.ok(
				messages.some(m => m.role === "user" && m.content === "follow"),
			);
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
			hooks: {
				getFollowUpMessages: () => {
					if (followUps++ === 0) return [user("follow")];
					return [];
				},
			},
		},
		() => {},
	);
	assert.deepEqual(
		newMessages.map(m => `${m.role}:${m.content ?? ""}`),
		["user:prompt", "assistant:first", "user:follow", "assistant:second"],
	);
});

void test("runAgentLoop executes a tool batch and returns ordered tool results", async () => {
	const tool: Tool = {
		name: "echo",
		description: "echoes",
		parameters: { type: "object", properties: { text: { type: "string" } } },
		execute: async args => `echo:${String(args.text)}`,
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
		newMessages
			.filter(m => m.role === "tool")
			.map(m => `${m.tool_call_id}:${m.content ?? ""}`),
		["a:echo:one", "b:echo:two"],
	);
});

void test("runAgentLoop promotes textual XML tool calls and hides their markup", async () => {
	let received: Record<string, unknown> | undefined;
	const grepTool: Tool = {
		name: "grep",
		description: "search",
		parameters: { type: "object", properties: {} },
		execute: async args => {
			received = args;
			return "match";
		},
	};
	const textualCall = [
		"Searching now.",
		"<tool_call>",
		"<function=grep>",
		"<parameter=pattern>notice|NoticeEvent</parameter>",
		"<parameter=path>/tmp/bridge.ts</parameter>",
		"<parameter=limit>50</parameter>",
		"</function>",
		"</tool_call>",
	].join("\n");
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [grepTool] },
		[user("search")],
		{
			...makeConfig({ tools: [grepTool] }),
			backend: new FakeBackend([
				() => textResponse(textualCall),
				() => textResponse("Search complete."),
			]),
		},
		() => {},
	);

	assert.deepEqual(received, {
		pattern: "notice|NoticeEvent",
		path: "/tmp/bridge.ts",
		limit: 50,
	});
	const promoted = messages.find(
		message => message.role === "assistant" && message.tool_calls?.length,
	);
	assert.equal(promoted?.content, "Searching now.");
	assert.doesNotMatch(String(promoted?.content), /tool_call|function=grep/);
});

void test("runAgentLoop executes independent tool calls in parallel and preserves result order", async () => {
	const completions: string[] = [];
	const tool: Tool = {
		name: "wait",
		description: "waits",
		parameters: { type: "object", properties: {} },
		execute: async args => {
			const name = String(args.name);
			await new Promise(resolve => setTimeout(resolve, Number(args.delay)));
			completions.push(name);
			return name;
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{
					id: "slow",
					name: "wait",
					arguments: JSON.stringify({ name: "slow", delay: 20 }),
				},
				{
					id: "fast",
					name: "wait",
					arguments: JSON.stringify({ name: "fast", delay: 0 }),
				},
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
	assert.deepEqual(
		messages.filter(m => m.role === "tool").map(m => m.tool_call_id),
		["slow", "fast"],
	);
});

void test("sequential tools are barriers without disabling parallel stages", async () => {
	const events: string[] = [];
	let releaseReads!: () => void;
	const readsDone = new Promise<void>(resolve => {
		releaseReads = resolve;
	});
	let completedReads = 0;
	const read: Tool = {
		name: "read",
		description: "parallel read",
		parameters: { type: "object", properties: {} },
		executionMode: "parallel",
		execute: async args => {
			events.push(`start:${String(args.id)}`);
			completedReads++;
			if (completedReads === 2) releaseReads();
			await readsDone;
			events.push(`end:${String(args.id)}`);
			return "read";
		},
	};
	const write: Tool = {
		name: "write",
		description: "sequential write",
		parameters: { type: "object", properties: {} },
		executionMode: "sequential",
		execute: async () => {
			events.push("write");
			return "written";
		},
	};
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [
				{ id: "a", name: "read", arguments: '{"id":"a"}' },
				{ id: "b", name: "read", arguments: '{"id":"b"}' },
				{ id: "w", name: "write", arguments: "{}" },
				{ id: "c", name: "read", arguments: '{"id":"c"}' },
			],
			stopReason: "stop",
		}),
		() => textResponse("done"),
	]);
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [read, write] },
		[user("prompt")],
		{
			...makeConfig({ tools: [read, write], toolExecution: "parallel" }),
			backend,
		},
		() => {},
	);
	assert.ok(events.indexOf("start:b") < events.indexOf("end:a"));
	assert.ok(events.indexOf("write") > events.indexOf("end:a"));
	assert.ok(events.indexOf("start:c") > events.indexOf("write"));
	assert.deepEqual(
		messages
			.filter(message => message.role === "tool")
			.map(message => message.tool_call_id),
		["a", "b", "w", "c"],
	);
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

	const results = messages.filter(message => message.role === "tool");
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
		execute: async args => {
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
	assert.deepEqual(
		new Set(order.slice(2)),
		new Set(["execute:1", "execute:2"]),
	);
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
				{
					id: "bad",
					name: "progress",
					arguments: JSON.stringify({ fail: true }),
				},
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
		event => {
			events.push(event);
		},
	);
	assert.equal(
		events.filter(event => event.type === "tool_execution_update").length,
		2,
	);
	assert.ok(
		events.some(
			event =>
				event.type === "tool_call_end" &&
				event.toolCallId === "bad" &&
				event.isError === true,
		),
	);
});

void test("runAgentLoop does not execute tool calls from a length-truncated response", async () => {
	let executions = 0;
	let preflights = 0;
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
			toolCalls: [{ id: "cut", name: "write", arguments: '{"path":"partial' }],
			stopReason: "length",
		}),
		() => textResponse("done"),
	]);
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [tool] },
		[user("prompt")],
		{
			...makeConfig({ tools: [tool] }),
			backend,
			hooks: {
				beforeToolCall: () => {
					preflights++;
					return undefined;
				},
			},
		},
		() => {},
	);
	assert.equal(executions, 0);
	assert.equal(preflights, 0);
	assert.match(
		String(messages.find(m => m.role === "tool")?.content),
		/not executed.*truncated/i,
	);
});

void test("tool hooks run before and after tool execution", async () => {
	let beforeCalls = 0;
	let afterCalls = 0;
	const backend = new FakeBackend([
		() => ({
			content: "",
			toolCalls: [{ id: "noop", name: "noop", arguments: "{}" }],
			stopReason: "stop",
		}),
		() => textResponse("done"),
	]);

	await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{
			...makeConfig(),
			backend,
			hooks: {
				beforeToolCall: () => {
					beforeCalls++;
					return undefined;
				},
				afterToolCall: () => {
					afterCalls++;
					return undefined;
				},
			},
		},
		() => {},
	);

	assert.equal(beforeCalls, 1);
	assert.equal(afterCalls, 1);
});

void test("runAgentLoop reports provider errors as terminal new messages", async () => {
	const events: AgentEvent[] = [];
	const backend = new FakeBackend([
		() => ({
			content: null,
			toolCalls: [],
			stopReason: "error",
			errorMessage: "boom",
		}),
	]);
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend },
		event => {
			events.push(event);
		},
	);
	assert.equal(newMessages.at(-1)?.role, "assistant");
	assert.ok(
		events.some(event => event.type === "error" && event.message === "boom"),
	);
	assert.ok(events.some(event => event.type === "agent_end"));
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
		event => {
			events.push(event);
		},
	);
	assert.equal(backend.calls, 0);
	assert.deepEqual(
		newMessages.map(m => `${m.role}:${m.content ?? ""}`),
		["user:prompt"],
	);
	assert.ok(
		events.some(
			event => event.type === "error" && event.message.includes("aborted"),
		),
	);
});

void test("continuation does not turn a conversational reply into hidden extra turns", async () => {
	const backend = new FakeBackend([
		() => textResponse("Hi! How can I help you today?"),
	]);
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("hi")],
		{ ...makeConfig({ continuationEnabled: true }), backend },
		() => {},
	);

	assert.equal(backend.calls, 1);
	assert.equal(messages.at(-1)?.content, "Hi! How can I help you today?");
});

void test("minimal profile stops naturally when no embedded features enabled", async () => {
	// The executionProfile is unified (embedded policies always enabled by default),
	// Callers can opt out of continuation and acceptance. When neither is
	// enabled, the loop stops after one provider call.
	const backend = new FakeBackend([
		() => textResponse("I still need to check the test output."),
	]);
	const events: AgentEvent[] = [];
	const messages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("check it")],
		{
			...makeConfig({
				executionProfile: "minimal",
				// No continuation, reflection, or acceptance enabled
			}),
			backend,
		},
		event => {
			events.push(event);
		},
	);

	assert.equal(backend.calls, 1);
	assert.equal(
		messages.at(-1)?.content,
		"I still need to check the test output.",
	);
	assert.ok(
		events.some(
			event => event.type === "agent_end" && event.status === "completed",
		),
	);
	assert.equal(
		events.some(event => event.type === "acceptance_complete"),
		false,
	);
});

void test("a tool call with unparseable JSON arguments is sanitized before it reaches history", async () => {
	const backend = new FakeBackend([
		() => ({
			content: "writing the file",
			toolCalls: [
				{
					id: "call1",
					name: "write_file",
					// Truncated mid-argument, as happens when stopReason is "length".
					arguments:
						'{"path":"big.txt","content":"start of a huge file that got cut off',
				},
			],
			stopReason: "length",
		}),
		() => textResponse("done"),
	]);
	const newMessages = await runAgentLoop(
		{ systemPrompt: "test", messages: [], tools: [noop] },
		[user("prompt")],
		{ ...makeConfig(), backend },
		() => {},
	);

	// The persisted assistant message keeps the call (so its id still pairs
	// with the tool-result below), but its arguments must always be valid
	// JSON so the backend never fails to re-parse it on a later turn.
	const persistedCall = newMessages
		.find(
			m =>
				m.role === "assistant" && m.tool_calls?.some(tc => tc.id === "call1"),
		)
		?.tool_calls?.find(tc => tc.id === "call1");
	assert.ok(persistedCall, "tool_call must still be present for id pairing");
	assert.doesNotThrow(() => JSON.parse(persistedCall?.arguments));

	// The executor's own truncation handling still produces the paired
	// "not executed" tool-result using the real call id.
	const toolResult = newMessages.find(
		m => m.role === "tool" && m.tool_call_id === "call1",
	);
	assert.match(String(toolResult?.content), /not executed.*truncated/i);
});
