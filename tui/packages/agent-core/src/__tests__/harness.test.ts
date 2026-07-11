import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { AgentHarness, HarnessBusyError } from "../core/harness.ts";
import { Session } from "../core/session.ts";
import type { AgentConfig } from "../core/types.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";
import { BackendError } from "../core/backend.ts";

function makeHarness(backend: FakeBackend): AgentHarness {
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
	return new AgentHarness({ config, backend, maxIterations: 5 });
}

void test("prompt persists history; setHistory replaces it and drops system messages", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("hi!")]));
	await harness.prompt("hello");
	const roles = harness.messages.map((m) => m.role);
	assert.deepEqual(roles, ["system", "user", "assistant"]);

	harness.setHistory([
		{ role: "system", content: "stale prompt" },
		{ role: "user", content: "restored q" },
		{ role: "assistant", content: "restored a" },
	]);
	assert.deepEqual(
		harness.messages.map((m) => m.role),
		["user", "assistant"],
	);
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
		harness.messages.some((message) =>
			String(message.content).includes("context-compaction"),
		),
	);
	assert.equal(harness.messages.at(-1)?.content, "recovered");
});

void test("runtimeState is canonical across streaming, tools, and settlement", async () => {
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
		(messages) => {
			// The queued note must appear as a user message before the prompt.
			const contents = messages.map((m) => String(m.content ?? ""));
			const noteAt = contents.findIndex((c) => c.includes("queued note"));
			const promptAt = contents.findIndex((c) => c.includes("real prompt"));
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
	const harness = makeHarness(new FakeBackend([() => textResponse("answer")]));
	await harness.enableSession(dir);
	await harness.prompt("question");

	const persisted = harness.listSessions();
	assert.equal(persisted.length, 1);
	const resumed = makeHarness(new FakeBackend([]));
	assert.equal(await resumed.resumeSession(persisted[0].id, dir), true);
	assert.deepEqual(
		resumed.messages.map((m) => `${m.role}:${m.content ?? ""}`),
		["user:question", "assistant:answer"],
	);
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
	session.appendLabel(lastId!, "answer");

	const context = session.buildContext();
	assert.equal(context.model, "model-a");
	assert.equal(context.thinkingLevel, "high");
	assert.deepEqual(context.activeToolNames, ["read", "write"]);
	assert.equal(context.labels.get(lastId!), "answer");
	assert.deepEqual(
		context.messages.map((message) => `${message.role}:${message.content ?? ""}`),
		[
			"system:<compaction_summary>summarized old context</compaction_summary>",
			"user:old",
			"assistant:new",
		],
	);
});
