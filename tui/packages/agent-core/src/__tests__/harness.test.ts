import { test } from "bun:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BackendError } from "../agent/backend.ts";
import { AgentHarness, HarnessBusyError } from "../agent/harness.ts";
import { RunKernel } from "../agent/run-kernel.ts";
import { Session } from "../agent/session.ts";
import type { AgentConfig } from "../agent/types.ts";
import { PermissionManager } from "../tools/shared/permissions.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

function makeHarness(backend: FakeBackend, cwd?: string): AgentHarness {
	const config: AgentConfig = {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		// One no-op tool so the loop doesn't register the default tool set
		// (bash etc.) in a unit test.
		tools: [
			{
				name: "noop",
				description: "does nothing",
				parameters: { type: "object", properties: {} },
				execute: async () => "ok",
			},
		],
	};
	return new AgentHarness({ config, backend, cwd, maxIterations: 5 });
}

void test("prompt persists history; setHistory replaces it and drops system messages", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("hi!")]));
	await harness.prompt("hello");
	const roles = harness.messages.map(m => m.role);
	assert.deepEqual(roles, ["system", "user", "assistant"]);

	harness.setHistory([
		{ role: "system", content: "stale prompt" },
		{ role: "user", content: "restored q" },
		{ role: "assistant", content: "restored a" },
	]);
	assert.deepEqual(
		harness.messages.map(m => m.role),
		["user", "assistant"],
	);
});

void test("appendMessages adds to history without dropping prior turns", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("hi!")]));
	await harness.prompt("hello");
	harness.appendMessages([
		{ role: "user", content: "/spawn list files" },
		{
			role: "assistant",
			content: null,
			tool_calls: [
				{
					id: "spawn_1",
					name: "spawn_agent",
					arguments: JSON.stringify({ task: "list files", agent: "general" }),
				},
			],
		},
		{
			role: "tool",
			content: "a.md\nb.md",
			tool_call_id: "spawn_1",
			name: "spawn_agent",
		},
	]);
	const roles = harness.messages.map(m => m.role);
	assert.deepEqual(roles, [
		"system",
		"user",
		"assistant",
		"user",
		"assistant",
		"tool",
	]);
	const tool = harness.messages.at(-1);
	assert.equal(tool?.content, "a.md\nb.md");
	assert.equal(tool?.tool_call_id, "spawn_1");
});

void test("constructor stream options reach the first provider request", async () => {
	const backend = new FakeBackend([
		(_messages, options) => {
			assert.equal(options.timeoutMs, 1234);
			assert.equal(options.maxRetries, 7);
			assert.deepEqual(options.headers, { "x-test": "yes" });
			assert.equal(options.cacheRetention, "persistent");
			return textResponse("done");
		},
	]);
	const harness = new AgentHarness({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			continuationEnabled: false,
			streamOptions: {
				timeoutMs: 1234,
				maxRetries: 7,
				headers: { "x-test": "yes" },
				cacheRetention: "persistent",
			},
		},
		backend,
		maxIterations: 1,
	});

	await harness.prompt("hello");
});

void test("legacy turn timeout and disabled retries reach the provider", async () => {
	const backend = new FakeBackend([
		(_messages, options) => {
			assert.equal(options.timeoutMs, 4321);
			assert.equal(options.maxRetries, 0);
			return textResponse("done");
		},
	]);
	const harness = new AgentHarness({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			continuationEnabled: false,
			turnTimeoutMs: 4321,
			autoRetryEnabled: false,
		},
		backend,
		maxIterations: 1,
	});

	await harness.prompt("hello");
});

void test("retryBaseDelayMs configures output-guard retry events", async () => {
	const backend = new FakeBackend([
		() => {
			throw new BackendError({ category: "transient", message: "temporary" });
		},
		() => textResponse("recovered"),
	]);
	const harness = new AgentHarness({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			contextWindowTokens: 4096,
			continuationEnabled: false,
			maxRetries: 1,
			retryBaseDelayMs: 0,
		},
		backend,
		maxIterations: 1,
	});
	const delays: number[] = [];
	harness.subscribe(event => {
		if (event.type === "agent_retry_start") delays.push(event.delayMs);
	});

	await harness.prompt("hello");
	assert.deepEqual(delays, [0]);
	assert.equal(backend.calls, 2);
});

void test("context-full recovery compacts inside an active turn and persists it", async () => {
	const backend = new FakeBackend([
		() => {
			throw new BackendError({ category: "context_full", message: "too long" });
		},
		() => textResponse("recovered"),
	]);
	const config: AgentConfig = {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		contextWindowTokens: 4096,
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		tools: [],
	};
	const harness = new AgentHarness({ config, backend, maxIterations: 1 });
	harness.setHistory(
		Array.from({ length: 12 }, (_, index) => ({
			role: "user" as const,
			content: `old ${index} ${"x".repeat(2000)}`,
		})),
	);
	await harness.prompt("current prompt");
	assert.equal(backend.calls, 2);
	assert.ok(
		harness.messages.some(message =>
			String(message.content).includes("context-compaction"),
		),
	);
	assert.equal(harness.messages.at(-1)?.content, "recovered");
});

void test("runtimeState is canonical across streaming, tools, and settlement", async () => {
	// eslint-disable-next-line prefer-const -- harness used in closures before assignment
	let harness!: AgentHarness;
	const phases: Array<[string, string]> = [];
	const tool = {
		name: "inspect",
		description: "inspect runtime state",
		parameters: { type: "object", properties: {} },
		execute: async () => {
			assert.equal(harness.runtimeState.isStreaming, true);
			assert.deepEqual(harness.runtimeState.pendingToolCalls, ["call_1"]);
			return "ok";
		},
	};
	const backend = new FakeBackend([
		(_messages, options) => {
			options.callbacks?.onDelta?.("partial");
			assert.equal(harness.runtimeState.streamingMessage?.content, "partial");
			options.callbacks?.onToolCallStart?.("call_1", "inspect", "{}");
			return {
				content: "",
				toolCalls: [{ id: "call_1", name: "inspect", arguments: "{}" }],
				stopReason: "stop",
			};
		},
		() => textResponse("done"),
	]);
	harness = new AgentHarness({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			runtimeHooksEnabled: false,
			proactiveCompactionEnabled: false,
			continuationEnabled: false,
			tools: [tool],
		},
		backend,
	});
	harness.setOnPhaseChange((phase, previous) => {
		phases.push([phase, previous]);
	});
	await harness.prompt("hello");
	assert.deepEqual(phases, [
		["turn", "idle"],
		["idle", "turn"],
	]);
	const settled = harness.runtimeState;
	assert.equal(settled.phase, "idle");
	assert.equal(settled.isStreaming, false);
	assert.equal(settled.turnId, undefined);
	assert.equal(settled.streamingMessage, undefined);
	assert.deepEqual(settled.pendingToolCalls, []);
	assert.equal(settled.retry, undefined);
	assert.equal(settled.abortRequested, false);
	assert.deepEqual(settled.outcome, {
		status: "completed",
		summary: "done",
		source: "heuristic",
	});
	assert.ok((settled.lastEventSeq ?? 0) > 0);
	assert.ok((settled.lastTurnDurationMs ?? -1) >= 0);
	assert.ok((settled.lastRunDurationMs ?? -1) >= 0);
});

void test("steer outside a turn throws HarnessBusyError", () => {
	const harness = makeHarness(new FakeBackend([]));
	assert.throws(() => harness.steer("now"), HarnessBusyError);
});

void test("rewind restores the pre-prompt conversation, then returns null", async () => {
	const harness = makeHarness(
		new FakeBackend([() => textResponse("a1"), () => textResponse("a2")]),
	);
	await harness.prompt("q1");
	const afterFirst = harness.messages.length;
	await harness.prompt("q2");
	assert.ok(harness.messages.length > afterFirst);

	// Undo the second turn: history is back to the post-q1 state (minus the
	// system message the loop re-injects each run).
	const restored = harness.rewind();
	assert.ok(restored);
	assert.equal(restored.messages, afterFirst);
	// Undo the first turn too.
	assert.ok(harness.rewind());
	// Nothing left.
	assert.equal(harness.rewind(), null);
});

void test("nextTurn queue survives until the next prompt and is injected before it", async () => {
	const backend = new FakeBackend([
		messages => {
			// The queued note must appear as a user message before the prompt.
			const contents = messages.map(m => String(m.content ?? ""));
			const noteAt = contents.findIndex(c => c.includes("queued note"));
			const promptAt = contents.findIndex(c => c.includes("real prompt"));
			assert.ok(noteAt >= 0, "queued note injected");
			assert.ok(promptAt >= 0, "prompt present");
			assert.ok(noteAt < promptAt, "note precedes the prompt");
			return textResponse("saw it");
		},
	]);
	const harness = makeHarness(backend);
	harness.nextTurn("queued note");
	assert.deepEqual(harness.getQueues().nextTurn, ["queued note"]);
	await harness.prompt("real prompt");
	assert.deepEqual(harness.getQueues().nextTurn, []);
});

void test("prompt hook messages remain adjacent to the prompt that produced them", async () => {
	const harness = makeHarness(
		new FakeBackend([
			() => textResponse("first answer"),
			() => textResponse("second answer"),
		]),
	);
	harness.setBeforeAgentStart(prompt => ({
		messages: [{ role: "user", content: `hook for ${prompt}` }],
	}));

	await harness.prompt("first prompt");
	await harness.prompt("second prompt");

	assert.deepEqual(
		harness.messages.map(message => message.content),
		[
			"test",
			"hook for first prompt",
			"first prompt",
			"first answer",
			"hook for second prompt",
			"second prompt",
			"second answer",
		],
	);
});

void test("nextTurn queued during a run waits for the following user prompt", async () => {
	// eslint-disable-next-line prefer-const -- harness used in closures before assignment
	let harness!: AgentHarness;
	const backend = new FakeBackend([
		() => {
			harness.nextTurn("future guidance");
			return textResponse("first answer");
		},
		messages => {
			const contents = messages.map(message => String(message.content ?? ""));
			assert.ok(contents.includes("future guidance"));
			assert.ok(
				contents.indexOf("future guidance") < contents.indexOf("second prompt"),
			);
			return textResponse("second answer");
		},
	]);
	harness = makeHarness(backend);

	await harness.prompt("first prompt");
	assert.deepEqual(harness.getQueues().nextTurn, ["future guidance"]);
	await harness.prompt("second prompt");
	assert.deepEqual(harness.getQueues().nextTurn, []);
});

void test("abort preserves nextTurn messages and waits for settlement", async () => {
	let providerSettled = false;
	const backend = new FakeBackend([
		(_messages, options) =>
			new Promise(resolve => {
				options.signal?.addEventListener("abort", () => {
					providerSettled = true;
					resolve(textResponse("aborted"));
				});
			}),
	]);
	const harness = makeHarness(backend);
	const run = harness.prompt("work");
	await new Promise(resolve => setImmediate(resolve));
	harness.nextTurn("keep me");

	const cleared = await harness.abort();
	await run;
	assert.equal(providerSettled, true);
	assert.deepEqual(cleared.clearedNextTurn, []);
	assert.deepEqual(harness.getQueues().nextTurn, ["keep me"]);
	assert.equal(harness.phase, "idle");
});

void test("fork + discardBranch restores the parent conversation", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("base")]));
	await harness.prompt("q");
	const baseLen = harness.messages.length;
	harness.fork();
	harness.setHistory([]); // diverge wildly on the branch...
	// (setHistory clears branches, so re-fork to test discard properly)
	const harness2 = makeHarness(new FakeBackend([() => textResponse("base")]));
	await harness2.prompt("q");
	const base2 = harness2.messages.length;
	harness2.fork();
	assert.equal(harness2.listBranches().length, 1);
	assert.equal(harness2.discardBranch(), true);
	assert.equal(harness2.messages.length, base2);
	assert.equal(harness2.listBranches().length, 0);
	assert.ok(baseLen > 0);
});

void test("enabled session persists real turn messages without placeholders", async () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-session-"));
	const harness = makeHarness(
		new FakeBackend([() => textResponse("answer")]),
		dir,
	);
	await harness.enableSession(dir);
	await harness.prompt("question");

	const persisted = harness.listSessions();
	assert.equal(persisted.length, 1);
	const resumed = makeHarness(new FakeBackend([]));
	assert.equal(await resumed.resumeSession(persisted[0].id, dir), true);
	assert.deepEqual(
		resumed.messages.map(m => `${m.role}:${m.content ?? ""}`),
		["user:question", "assistant:answer"],
	);
});

void test("enabled sessions use the kernel as the sole execution journal", async () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-journal-"));
	const harness = makeHarness(
		new FakeBackend([() => textResponse("answer")]),
		dir,
	);
	await harness.enableSession(dir);
	await harness.prompt("question");

	const sessionInfo = harness.listSessions()[0];
	const session = new Session(sessionInfo.id, { baseDir: dir, enabled: true });
	assert.ok(session);
	assert.equal(
		existsSync(join(dir, ".logician", "runtime", `${sessionInfo.id}.jsonl`)),
		false,
	);
	assert.equal(
		existsSync(
			join(dir, ".logician", "trajectories", `${sessionInfo.id}.jsonl`),
		),
		false,
	);
	const trajectory = new RunKernel(dir, sessionInfo.id).snapshot().state
		.trajectory;
	assert.ok(trajectory.some(event => event.kind === "run_start"));
	assert.ok(
		trajectory.some(
			event =>
				event.kind === "agent_event" && event.payload.type === "turn_start",
		),
	);
	assert.ok(trajectory.some(event => event.kind === "run_finish"));
});

void test("harness records run state and trajectory in the unified kernel ledger", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-harness-"));
	const sessionId = "kernel-session";
	const harness = new AgentHarness({
		cwd,
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			runtimeHooksEnabled: false,
			proactiveCompactionEnabled: false,
			continuationEnabled: false,
			tools: [],
		},
		backend: new FakeBackend([() => textResponse("answer")]),
		maxIterations: 5,
	});
	harness.setSessionId(sessionId);
	await harness.prompt("question");

	const replay = new RunKernel(cwd, sessionId).snapshot();
	assert.deepEqual(replay.violations, []);
	assert.equal(replay.state.rootPrompt, "question");
	assert.ok(replay.state.trajectory.some(item => item.kind === "run_start"));
	assert.ok(replay.state.trajectory.some(item => item.kind === "run_finish"));
	assert.ok(
		replay.state.trajectory.some(
			item =>
				item.kind === "agent_event" && item.payload.type === "run_outcome",
		),
	);
});

void test("provider budget remains exhausted across native continuation runs", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-budget-"));
	let backendCalls = 0;
	const backend = new FakeBackend([
		() => {
			backendCalls++;
			return textResponse("first turn complete");
		},
		() => {
			backendCalls++;
			return textResponse("must not run");
		},
	]);
	const harness = new AgentHarness({
		cwd,
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			runtimeHooksEnabled: false,
			proactiveCompactionEnabled: false,
			continuationEnabled: false,
			runBudget: { maxProviderCalls: 1, maxToolCalls: 10 },
			tools: [],
		},
		backend,
		maxIterations: 5,
	});
	harness.setSessionId("budget-session");
	await harness.prompt("start");
	harness.nextTurn("continue internally");
	assert.equal(
		harness.requestContinuation("test", "progress").action,
		"continue",
	);
	await harness.continueWithNextTurn();

	assert.equal(backendCalls, 1);
	const replay = new RunKernel(cwd, "budget-session").snapshot();
	assert.equal(replay.state.budgets.provider_call, 1);
	assert.equal(replay.state.status, "blocked");
});

void test("token usage is persisted before a token-budget stop", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-tokens-"));
	const harness = new AgentHarness({
		cwd,
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			runtimeHooksEnabled: false,
			proactiveCompactionEnabled: false,
			continuationEnabled: false,
			maxTotalTokens: 50,
			tools: [],
		},
		backend: new FakeBackend([
			() => ({
				content: "over budget",
				toolCalls: [],
				stopReason: "stop",
				usage: { totalTokens: 60, promptTokens: 40, completionTokens: 20 },
			}),
		]),
		maxIterations: 5,
	});
	harness.setSessionId("token-session");
	await harness.prompt("start");
	const state = new RunKernel(cwd, "token-session").snapshot().state;
	assert.equal(state.budgets.token, 60);
	assert.equal(state.status, "blocked");
});

void test("kernel restores task and budget state in a fresh harness process", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-restart-"));
	const config: AgentConfig = {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		runBudget: { maxProviderCalls: 1, maxToolCalls: 10 },
		tools: [],
	};
	const first = new AgentHarness({
		cwd,
		config: { ...config },
		backend: new FakeBackend([() => textResponse("done")]),
		maxIterations: 5,
	});
	first.setSessionId("restart-session");
	await first.prompt("original task");

	let restartedBackendCalls = 0;
	const restarted = new AgentHarness({
		cwd,
		config: { ...config },
		backend: new FakeBackend([
			() => {
				restartedBackendCalls++;
				return textResponse("must not execute");
			},
		]),
		maxIterations: 5,
	});
	restarted.setSessionId("restart-session");
	restarted.setHistory([{ role: "user", content: "continue saved task" }]);
	assert.equal(
		restarted.requestContinuation("restart", "new-progress").action,
		"continue",
	);
	await restarted.continue();

	assert.equal(restartedBackendCalls, 0);
	const state = new RunKernel(cwd, "restart-session").snapshot().state;
	assert.equal(state.rootPrompt, "original task");
	assert.equal(state.budgets.provider_call, 1);
	assert.equal(state.status, "blocked");
	assert.ok(state.leaseEpoch >= 3);
});

void test("kernel restores pending next-turn guidance after restart", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-queue-"));
	const first = makeHarness(
		new FakeBackend([() => textResponse("waiting")]),
		cwd,
	);
	first.setSessionId("queue-session");
	await first.prompt("start");
	first.nextTurn("preserve this guidance");

	const restarted = makeHarness(new FakeBackend([]), cwd);
	restarted.setSessionId("queue-session");
	assert.deepEqual(restarted.getQueues().nextTurn, ["preserve this guidance"]);
	assert.deepEqual(
		new RunKernel(cwd, "queue-session").snapshot().state.queues.nextTurn,
		["preserve this guidance"],
	);
});

void test("real tool execution durably records intent, result, and commit", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-tool-"));
	let receivedIdempotencyKey: string | undefined;
	const harness = new AgentHarness({
		cwd,
		config: {
			baseUrl: "http://fake",
			model: "fake",
			systemPrompt: "test",
			runtimeHooksEnabled: false,
			proactiveCompactionEnabled: false,
			continuationEnabled: false,
			tools: [
				{
					name: "lookup",
					description: "pure lookup",
					parameters: { type: "object" },
					readOnly: true,
					recoverySemantics: "receipt_recoverable",
					execute: async (args, context) => {
						receivedIdempotencyKey = context.idempotencyKey;
						return {
							content: `value:${String(args.key)}`,
							recoveryReceipt: "provider-receipt-1",
						};
					},
				},
			],
		},
		backend: new FakeBackend([
			() => ({
				content: "",
				toolCalls: [
					{ id: "call-lookup", name: "lookup", arguments: '{"key":"x"}' },
				],
				stopReason: "stop",
			}),
			() => textResponse("done"),
		]),
		maxIterations: 5,
	});
	harness.setSessionId("tool-session");
	await harness.prompt("look it up");

	const state = new RunKernel(cwd, "tool-session").snapshot().state;
	const operations = Object.values(state.operations);
	assert.equal(operations.length, 1);
	assert.equal(operations[0]?.toolName, "lookup");
	assert.equal(operations[0]?.recovery, "receipt_recoverable");
	assert.equal(operations[0]?.status, "committed");
	assert.equal(operations[0]?.receipt, "provider-receipt-1");
	assert.match(operations[0]?.argumentsDigest ?? "", /^[a-f0-9]{64}$/);
	assert.match(operations[0]?.resultDigest ?? "", /^[a-f0-9]{64}$/);
	assert.equal(receivedIdempotencyKey, operations[0]?.idempotencyKey);
});

void test("kernel audits and restores session-scoped permission decisions", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-kernel-permission-"));
	const permissions = new PermissionManager({ mode: "ask" });
	const config: AgentConfig = {
		baseUrl: "http://fake",
		model: "fake",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		permissions,
		onPermissionRequest: async () => "always",
		tools: [
			{
				name: "mutate",
				description: "mutate",
				parameters: { type: "object", properties: {} },
				execute: async () => "changed",
			},
		],
	};
	const harness = new AgentHarness({
		cwd,
		config,
		backend: new FakeBackend([
			() => ({
				content: null,
				toolCalls: [{ id: "permission-call", name: "mutate", arguments: "{}" }],
				stopReason: "stop",
			}),
			() => textResponse("done"),
		]),
	});
	harness.setSessionId("permission-session");
	await harness.prompt("change it");
	const decision = new RunKernel(cwd, "permission-session").snapshot().state
		.permissionDecisions[0];
	assert.deepEqual(decision && { ...decision, sequence: undefined }, {
		toolCallId: "permission-call",
		toolName: "mutate",
		decision: "allow",
		source: "user",
		scope: "session",
		sequence: undefined,
	});

	const restoredPermissions = new PermissionManager({ mode: "ask" });
	const restarted = new AgentHarness({
		cwd,
		config: { ...config, permissions: restoredPermissions },
		backend: new FakeBackend([]),
	});
	restarted.setSessionId("permission-session");
	assert.equal(
		restoredPermissions.evaluate(
			{ id: "later", name: "mutate", arguments: "{}" },
			{},
		).decision,
		"allow",
	);
});

void test("continuation recovers recorded results and quarantines intent-only effects", async () => {
	for (const frontier of ["result", "intent"] as const) {
		const cwd = mkdtempSync(join(tmpdir(), `logician-recovery-${frontier}-`));
		const sessionId = `${frontier}-session`;
		const kernel = new RunKernel(cwd, sessionId);
		const ids = { taskId: "task", runId: "run", leaseEpoch: 1 };
		kernel.append(
			{ type: "task_started", rootPrompt: "recover", createdAt: Date.now() },
			ids,
		);
		kernel.append(
			{
				type: "operation_intent_recorded",
				operationId: "operation",
				toolCallId: "call",
				toolName: "noop",
				arguments: {},
				argumentsDigest: "args",
				idempotencyKey: "key",
				recovery: frontier === "intent" ? "at_most_once_unknown" : "pure",
			},
			ids,
		);
		if (frontier === "result")
			kernel.append(
				{
					type: "operation_result_recorded",
					operationId: "operation",
					resultDigest: "result",
					result: "durable result",
					isError: false,
				},
				ids,
			);

		let recoveredToolContent = "";
		const harness = makeHarness(
			new FakeBackend([
				messages => {
					const recovered = messages.find(message => message.role === "tool");
					recoveredToolContent = String(recovered?.content ?? "");
					return textResponse("done");
				},
			]),
			cwd,
		);
		harness.setSessionId(sessionId);
		harness.setHistory([
			{
				role: "assistant",
				content: null,
				tool_calls: [{ id: "call", name: "noop", arguments: "{}" }],
			},
		]);
		await harness.continue();
		const operation = new RunKernel(cwd, sessionId).snapshot().state.operations
			.operation;
		assert.equal(
			operation.status,
			frontier === "result" ? "committed" : "quarantined",
		);
		assert.match(
			recoveredToolContent,
			frontier === "result" ? /durable result/ : /indeterminate/,
		);
	}
});

void test("enabled sessions persist resumable checkpoints at tool boundaries", async () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-checkpoint-"));
	const harness = makeHarness(
		new FakeBackend([
			() => ({
				content: "",
				toolCalls: [{ id: "call_1", name: "noop", arguments: "{}" }],
				stopReason: "stop",
			}),
			() => textResponse("done"),
		]),
		dir,
	);
	await harness.enableSession(dir);
	await harness.prompt("run the tool");

	const sessionInfo = harness.listSessions()[0];
	const state = new RunKernel(dir, sessionInfo.id).snapshot().state;
	const toolEnd = state.trajectory.find(
		event =>
			event.kind === "agent_event" &&
			event.payload.type === "tool_execution_end" &&
			event.payload.toolCallId === "call_1",
	);
	assert.ok(toolEnd);
	assert.equal(Object.values(state.operations)[0]?.status, "committed");
});

void test("session typed entries build deterministic context", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-session-tree-"));
	const session = new Session("tree", { baseDir: dir, enabled: true });
	session.appendModelChange("model-a");
	session.appendThinkingLevelChange("high");
	session.appendActiveToolsChange(["read", "write"]);
	session.append({ role: "user", content: "old", timestamp: 1 });
	const firstKept = session.getLeafEntryId();
	session.appendCompaction("summarized old context", 100, firstKept);
	session.append({ role: "assistant", content: "new", timestamp: 2 });
	const lastId = session.getLeafEntryId();
	assert.ok(lastId);
	session.appendLabel(lastId, "answer");

	const context = session.buildContext();
	assert.equal(context.model, "model-a");
	assert.equal(context.thinkingLevel, "high");
	assert.deepEqual(context.activeToolNames, ["read", "write"]);
	assert.equal(context.labels.get(lastId), "answer");
	assert.deepEqual(
		context.messages.map(message => `${message.role}:${message.content ?? ""}`),
		[
			"system:<compaction_summary>summarized old context</compaction_summary>",
			"user:old",
			"assistant:new",
		],
	);
});

void test("session active leaf makes branch checkout durable without truncation", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-branch-"));
	const session = new Session("tree-session", { baseDir, enabled: true });
	session.append({ role: "user", content: "root", timestamp: 1 });
	const root = session.getLeafEntryId();
	assert.ok(root);
	session.append({ role: "assistant", content: "abandoned", timestamp: 2 });
	session.checkout(root);
	session.append({ role: "assistant", content: "selected", timestamp: 3 });

	const restored = new Session("tree-session", { baseDir, enabled: true });
	assert.deepEqual(
		restored.buildContext().messages.map(message => message.content),
		["root", "selected"],
	);
	assert.equal(restored.loadEntries().length, 3);
});
