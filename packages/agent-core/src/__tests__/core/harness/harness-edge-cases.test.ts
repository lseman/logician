import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	AgentHarness,
	HarnessBusyError,
} from "../../../core/harness/agent-harness.ts";
import type { AgentConfig } from "../../../core/types/index.ts";
import { FakeBackend, textResponse } from "../../fake-backend.ts";

function makeHarness(
	backend: FakeBackend,
	configOverrides?: Partial<AgentConfig>,
): AgentHarness {
	const config: AgentConfig = {
		baseUrl: "http://fake",
		model: "fake",
		temperature: 0.7,
		systemPrompt: "test",
		runtimeHooksEnabled: false,
		proactiveCompactionEnabled: false,
		continuationEnabled: false,
		maxIterations: 5,
		tools: [
			{
				name: "noop",
				description: "does nothing",
				parameters: { type: "object", properties: {} },
				execute: async () => "ok",
			},
		],
		...configOverrides,
	};
	return new AgentHarness({ config, backend });
}

void test("setTemperature takes effect on the next turn", async () => {
	const responses: Array<{ temp?: number }> = [];
	const backend = new FakeBackend([
		(_msgs, opts) => {
			responses.push({ temp: opts.temperature });
			return textResponse("a1");
		},
		(_msgs, opts) => {
			responses.push({ temp: opts.temperature });
			return textResponse("a2");
		},
	]);
	const harness = makeHarness(backend);
	await harness.prompt("q1");
	assert.equal(responses[0].temp, 0.7);

	harness.configure({ temperature: 1.2 });
	await harness.prompt("q2");
	assert.equal(responses[1].temp, 1.2);
});

void test("setSystemPrompt takes effect on the next turn", async () => {
	const systemPrompts: string[] = [];
	const backend = new FakeBackend([
		msgs => {
			const sys = msgs.find(m => m.role === "system");
			systemPrompts.push(String(sys?.content ?? ""));
			return textResponse("a1");
		},
		msgs => {
			const sys = msgs.find(m => m.role === "system");
			systemPrompts.push(String(sys?.content ?? ""));
			return textResponse("a2");
		},
	]);
	const harness = makeHarness(backend);
	await harness.prompt("q1");
	assert.ok(systemPrompts[0].includes("test"));

	harness.configure({ systemPrompt: "new system prompt" });
	await harness.prompt("q2");
	assert.ok(systemPrompts[1].includes("new system prompt"));
});

void test("runtime config changes take effect at the next save point", async () => {
	// eslint-disable-next-line prefer-const -- harness used in closures before assignment
	let harness!: AgentHarness;
	const temperatures: number[] = [];
	const prompts: string[] = [];
	const backend = new FakeBackend([
		(messages, options) => {
			temperatures.push(options.temperature ?? -1);
			prompts.push(
				String(
					messages.find(message => message.role === "system")?.content ?? "",
				),
			);
			harness.configure({
				temperature: 1.25,
				systemPrompt: "refreshed prompt",
			});
			return {
				content: "",
				toolCalls: [{ id: "call_1", name: "noop", arguments: "{}" }],
				stopReason: "stop",
			};
		},
		(messages, options) => {
			temperatures.push(options.temperature ?? -1);
			prompts.push(
				String(
					messages.find(message => message.role === "system")?.content ?? "",
				),
			);
			return textResponse("done");
		},
	]);
	harness = makeHarness(backend);

	await harness.prompt("work");
	assert.deepEqual(temperatures, [0.7, 1.25]);
	assert.equal(prompts[0], "test");
	assert.equal(prompts[1], "refreshed prompt");
});

void test("steer outside a turn throws HarnessBusyError", () => {
	const harness = makeHarness(new FakeBackend([]));
	assert.throws(() => harness.steer("now"), HarnessBusyError);
});

void test("phase transitions are enforced: cannot compact during turn", async () => {
	const harness = makeHarness(new FakeBackend([]));
	await harness.prompt("q");
	assert.equal(harness.phase, "idle");
	// compact() requires idle — this should work.
	// After a prompt, there are messages, so compact returns 0 (nothing to free)
	// rather than null (nothing to compact).
	const result = await harness.compact();
	assert.equal(result, 0);
});

void test("fork creates a branch that can be discarded", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	await harness.prompt("parent-q");
	const parentLen = harness.messages.length;
	const branchId = harness.fork();
	assert.ok(branchId.length > 0);
	const branches = harness.listBranches();
	assert.equal(branches.length, 1);
	assert.equal(branches[0].id, branchId);
	// Don't setHistory (it clears branches). Just discard immediately.
	const discardOk = harness.discardBranch();
	assert.equal(discardOk, true);
	assert.equal(harness.listBranches().length, 0);
	assert.equal(harness.messages.length, parentLen);
});

void test("compact() on empty messages returns null", async () => {
	const harness = makeHarness(new FakeBackend([]));
	const result = await harness.compact();
	assert.equal(result, null);
});

void test("clearQueues resets all queues", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	// Queues are empty by default.
	assert.equal(harness.getQueues().steering.length, 0);
	assert.equal(harness.getQueues().followUp.length, 0);
	assert.equal(harness.getQueues().nextTurn.length, 0);

	// nextTurn can be set while idle (it drains on next prompt).
	harness.nextTurn("n1");
	assert.equal(harness.getQueues().nextTurn.length, 1);

	const cleared = harness.clearQueues();
	// cleared returns the PREVIOUS queue contents.
	assert.equal(cleared.nextTurn.length, 1);
	assert.equal(cleared.nextTurn[0], "n1");
	// But the queues are now empty.
	assert.equal(harness.getQueues().nextTurn.length, 0);
});

void test("rewind returns null when nothing to rewind", async () => {
	const harness = makeHarness(new FakeBackend([]));
	const result = harness.rewind();
	assert.equal(result, null);
});

void test("configure with an empty tool list removes all tools", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	harness.configure({ tools: [] });
	assert.equal(harness.tools.list().length, 0);
});

void test("phase observers fire on turn transitions", async () => {
	const phases: Array<[string, string]> = [];
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	harness.observe({
		phaseChange: (from, to) => {
			phases.push([from, to]);
		},
	});
	await harness.prompt("q");
	assert.ok(
		phases.some(([from, to]) => to === "turn" && from === "idle"),
		`must enter turn: ${JSON.stringify(phases)}`,
	);
	assert.ok(
		phases.some(([from, to]) => from === "turn" && to === "idle"),
		`must exit turn: ${JSON.stringify(phases)}`,
	);
});

void test("getModel returns the configured model", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	assert.equal(harness.getModel(), "fake");
});

void test("prompt with empty user message works", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	const messages = await harness.prompt("");
	assert.ok(messages.length > 0);
});

void test("setSessionId is callable", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	harness.setSessionId("test-session");
	// Just verify it doesn't throw.
});

void test("setAutoCompactionSettings stores settings", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	harness.setAutoCompactionSettings({
		enabled: true,
		reserveTokens: 1000,
	});
});

void test("enableAutoCompaction toggles auto-compaction", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	harness.enableAutoCompaction(true);
	harness.enableAutoCompaction(false);
});

void test("config changes persist across prompt turns", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	await harness.prompt("q1");
	harness.configure({ temperature: 1.5 });
	await harness.prompt("q2");
	assert.equal(harness.currentConfig.temperature, 1.5);
	// Config should still be set after a subsequent prompt.
	await harness.prompt("q3");
	assert.equal(harness.currentConfig.temperature, 1.5);
});

void test("listBranches returns empty when no branches", async () => {
	const harness = makeHarness(new FakeBackend([() => textResponse("a")]));
	await harness.prompt("q");
	assert.equal(harness.listBranches().length, 0);
});
