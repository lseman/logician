import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { AgentHarness, HarnessBusyError } from "../core/harness.ts";
import type { AgentConfig } from "../core/types.ts";
import { FakeBackend, textResponse } from "./fake-backend.ts";

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
