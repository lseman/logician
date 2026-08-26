import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	appendFileSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	unlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BackendError } from "../../capabilities/provider/backend.ts";
import {
	SessionCorruptionError,
	SessionRegistry,
	SessionStore,
} from "../../capabilities/session/session-store.ts";
import {
	AgentSession,
	defineHarnessModule,
	HarnessBusyError,
	HarnessConfigurationError,
} from "../../runtime/harness/agent-session.ts";
import type { AgentConfig } from "../../system/types/types-config.ts";
import { FakeBackend, textResponse } from "../fake-backend.ts";

function makeHarness(backend: FakeBackend, cwd?: string): AgentSession {
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
	return new AgentSession({ config, backend, cwd, maxIterations: 5 });
}

void test("modules compose tools, defaults, and observers before construction", async () => {
	const phases: string[] = [];
	const moduleTool = {
		name: "module_tool",
		description: "module tool",
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
	};
	const module = defineHarnessModule({
		name: "example",
		config: { temperature: 0.2, maxTokens: 321, tools: [moduleTool] },
		observers: [{ phaseChange: phase => phases.push(phase) }],
	});
	const harness = new AgentSession({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			temperature: 0.8,
			continuationEnabled: false,
		},
		backend: new FakeBackend([() => textResponse("done")]),
		modules: [module],
	});

	assert.equal(harness.currentConfig.temperature, 0.8);
	assert.equal(harness.currentConfig.maxTokens, 321);
	assert.deepEqual(
		harness.currentConfig.tools?.map(tool => tool.name),
		["module_tool"],
	);
	await harness.prompt("hello");
	assert.deepEqual(phases, ["turn", "idle"]);
});

void test("modules reject duplicate identities and tool names", () => {
	const tool = {
		name: "duplicate",
		description: "duplicate",
		parameters: { type: "object", properties: {} },
		execute: async () => "ok",
	};
	const options = {
		config: { baseUrl: "http://fake", model: "fake", tools: [tool] },
		backend: new FakeBackend([]),
	};
	assert.throws(
		() =>
			new AgentSession({
				...options,
				modules: [{ name: "tools", config: { tools: [tool] } }],
			}),
		HarnessConfigurationError,
	);
	assert.throws(
		() =>
			new AgentSession({
				...options,
				config: { baseUrl: "http://fake", model: "fake" },
				modules: [{ name: "same" }, { name: "same" }],
			}),
		HarnessConfigurationError,
	);
});

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
	const harness = new AgentSession({
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
	const harness = new AgentSession({
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
	const harness = new AgentSession({
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
	harness.observe({
		event: event => {
			if (event.type === "agent_retry_start") delays.push(event.delayMs ?? 0);
		},
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
	const harness = new AgentSession({ config, backend, maxIterations: 1 });
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
	let harness!: AgentSession;
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
	harness = new AgentSession({
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
	harness.observe({
		phaseChange: (phase, previous) => {
			phases.push([phase, previous]);
		},
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
	});
	assert.ok((settled.lastEventSeq ?? 0) > 0);
	assert.ok((settled.lastTurnDurationMs ?? -1) >= 0);
	assert.ok((settled.lastRunDurationMs ?? -1) >= 0);
});

void test("steer outside a turn throws HarnessBusyError", () => {
	const harness = makeHarness(new FakeBackend([]));
	assert.throws(() => harness.steer("now"), HarnessBusyError);
});

void test("steering with steeringInterrupt survives the abort and reaches the next turn", async () => {
	const backend = new FakeBackend([
		() => {
			harness.steer("urgent correction");
			throw new DOMException("Operation aborted", "AbortError");
		},
		messages => {
			assert.ok(
				messages.some(
					m => m.role === "user" && m.content === "urgent correction",
				),
			);
			return textResponse("done");
		},
	]);
	const config: AgentConfig = {
		baseUrl: "http://fake",
		model: "fake",
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		steeringInterrupt: true,
		tools: [
			{
				name: "noop",
				description: "does nothing",
				parameters: { type: "object", properties: {} },
				execute: async () => "ok",
			},
		],
	};
	const harness = new AgentSession({ config, backend, maxIterations: 5 });
	let settledNextTurnCount = 0;
	harness.observe({
		settled: count => {
			settledNextTurnCount = count;
		},
	});

	await harness.prompt("initial question");
	assert.equal(settledNextTurnCount, 1);

	await harness.continueWithNextTurn();
	assert.ok(
		harness.messages.some(
			m => m.role === "user" && m.content === "urgent correction",
		),
	);
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
	let harness!: AgentSession;
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

void test("fork/branchSummary/discardBranch keep an attached session's leaf pointer in sync", async () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-branch-session-"));
	const session = new SessionStore("branch-test", {
		baseDir: dir,
		enabled: true,
	});

	const harness = makeHarness(new FakeBackend([() => textResponse("base")]));
	harness.attachSession(session);
	await harness.prompt("q");
	const rootLeaf = session.getLeafEntryId();
	assert.ok(rootLeaf);

	harness.fork();
	assert.equal(harness.discardBranch(), true);
	// discardBranch checks out the branch's recorded fork point, which was
	// the session's leaf at the moment of fork() — i.e. back to rootLeaf.
	assert.equal(session.getLeafEntryId(), rootLeaf);
});

void test("enabled session persists real turn messages without placeholders", async () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-session-"));
	const session = new SessionStore("persistence-test", {
		baseDir: dir,
		enabled: true,
	});
	const harness = makeHarness(
		new FakeBackend([() => textResponse("answer")]),
		dir,
	);
	harness.attachSession(session);
	await harness.prompt("question");

	const persisted = new SessionStore("persistence-test", {
		baseDir: dir,
		enabled: true,
	});
	assert.deepEqual(
		persisted.load().map(m => `${m.role}:${m.content ?? ""}`),
		["user:question", "assistant:answer"],
	);
});

void test("continuation is gated by the in-memory provider-call budget", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-continuation-budget-"));
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
	const harness = new AgentSession({
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
	await harness.continueWithNextTurn();

	assert.equal(backendCalls, 1);
	assert.equal(harness.runtimeState.outcome?.status, "blocked");
});

void test("token usage over budget stops the run", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-token-budget-"));
	const harness = new AgentSession({
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
	assert.equal(harness.runtimeState.outcome?.status, "blocked");
});

void test("enabled sessions persist tool results across a resume", async () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-checkpoint-"));
	const session = new SessionStore("tool-result-test", {
		baseDir: dir,
		enabled: true,
	});
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
	harness.attachSession(session);
	await harness.prompt("run the tool");

	const persisted = new SessionStore("tool-result-test", {
		baseDir: dir,
		enabled: true,
	});
	assert.ok(
		persisted
			.load()
			.some(m => m.role === "tool" && m.tool_call_id === "call_1"),
	);
});

void test("session typed entries build deterministic context", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-session-tree-"));
	const session = new SessionStore("tree", { baseDir: dir, enabled: true });
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
	const session = new SessionStore("tree-session", { baseDir, enabled: true });
	session.append({ role: "user", content: "root", timestamp: 1 });
	const root = session.getLeafEntryId();
	assert.ok(root);
	session.append({ role: "assistant", content: "abandoned", timestamp: 2 });
	session.checkout(root);
	session.append({ role: "assistant", content: "selected", timestamp: 3 });

	const restored = new SessionStore("tree-session", { baseDir, enabled: true });
	assert.deepEqual(
		restored.buildContext().messages.map(message => message.content),
		["root", "selected"],
	);
	assert.equal(restored.loadEntries().length, 3);
});

void test("session metadata is reconciled from the journal after an interrupted projection write", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-reconcile-"));
	const session = new SessionStore("repair", { baseDir, enabled: true });
	session.append({ role: "user", content: "one", timestamp: 1 });
	session.append({ role: "assistant", content: "two", timestamp: 2 });
	const expectedLeaf = session.getLeafEntryId();
	const metaPath = join(session.dirPath, "meta.json");
	const stale = {
		...session.getMeta(),
		messageCount: 1,
		activeLeafId: undefined,
	};
	writeFileSync(metaPath, JSON.stringify(stale), "utf8");

	const restored = new SessionStore("repair", { baseDir, enabled: true });
	assert.equal(restored.getMeta().messageCount, 2);
	assert.equal(restored.getLeafEntryId(), expectedLeaf);
	assert.equal(restored.getMeta().activeLeafId, expectedLeaf);
	assert.equal(
		readdirSync(restored.dirPath).some(name => name.endsWith(".tmp")),
		false,
	);
});

void test("session recovery removes only a truncated final JSONL record", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-tail-"));
	const session = new SessionStore("tail", { baseDir, enabled: true });
	session.append({ role: "user", content: "safe", timestamp: 1 });
	const journalPath = join(session.dirPath, "messages.jsonl");
	appendFileSync(journalPath, '{"type":"message","id":"partial', "utf8");

	const restored = new SessionStore("tail", { baseDir, enabled: true });
	assert.equal(restored.loadEntries().length, 1);
	assert.equal(readFileSync(journalPath, "utf8").endsWith("\n"), true);
	assert.doesNotMatch(readFileSync(journalPath, "utf8"), /partial/);
});

void test("session recovery rejects corruption before the final JSONL record", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-corrupt-"));
	const session = new SessionStore("corrupt", { baseDir, enabled: true });
	session.append({ role: "user", content: "safe", timestamp: 1 });
	const journalPath = join(session.dirPath, "messages.jsonl");
	appendFileSync(journalPath, "not-json\n", "utf8");

	assert.throws(
		() => new SessionStore("corrupt", { baseDir, enabled: true }),
		(error: unknown) =>
			error instanceof SessionCorruptionError && error.lineNumber === 2,
	);
});

void test("session registry rebuilds missing metadata from the durable journal", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-registry-"));
	const cwd = join(baseDir, "workspace");
	const session = new SessionStore("recoverable", {
		baseDir,
		enabled: true,
		cwd,
	});
	session.append({ role: "user", content: "recover me", timestamp: 10 });
	unlinkSync(join(session.dirPath, "meta.json"));

	const registry = new SessionRegistry({ baseDir });
	const recovered = registry.getSession("recoverable");
	assert.ok(recovered);
	assert.equal(recovered.getMeta().messageCount, 1);
	assert.equal(recovered.load()[0]?.content, "recover me");
	assert.equal(registry.listSessions()[0]?.id, "recoverable");
});

void test("session registry repairs corrupt metadata without hiding the session", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-meta-"));
	const session = new SessionStore("bad-meta", { baseDir, enabled: true });
	session.append({ role: "user", content: "still safe", timestamp: 10 });
	writeFileSync(join(session.dirPath, "meta.json"), "{truncated", "utf8");

	const registry = new SessionRegistry({ baseDir });
	assert.equal(registry.listSessions()[0]?.id, "bad-meta");
	assert.equal(
		registry.getSession("bad-meta")?.load()[0]?.content,
		"still safe",
	);
});

void test("session validation rejects unknown typed entries", () => {
	const baseDir = mkdtempSync(join(tmpdir(), "logician-session-schema-"));
	const session = new SessionStore("schema", { baseDir, enabled: true });
	const journalPath = join(session.dirPath, "messages.jsonl");
	appendFileSync(
		journalPath,
		'{"type":"future_unknown","id":"x","timestamp":1}\n',
		"utf8",
	);

	assert.throws(
		() => session.loadEntries(),
		(error: unknown) =>
			error instanceof SessionCorruptionError && error.lineNumber === 1,
	);
});
