import { test } from "bun:test";
import assert from "node:assert/strict";
import { HookBus } from "../../runtime/hooks/hook-bus.ts";
import type { ToolCall } from "../../system/types/types-messages.ts";

const ctx = {
	toolCall: { id: "1", name: "bash", arguments: "{}" } as ToolCall,
	args: {},
	iteration: 1,
};

void test("beforeToolCall: first content result short-circuits", async () => {
	const bus = new HookBus();
	bus.on("beforeToolCall", () => ({ content: "blocked", isError: true }));
	bus.on("beforeToolCall", () => ({ content: "never" }));
	const r = await bus.toHooks().beforeToolCall?.(ctx);
	assert.equal(r?.content, "blocked");
});

void test("beforeToolCall: args rewrites thread to later handlers", async () => {
	const bus = new HookBus();
	bus.on("beforeToolCall", () => ({ args: { a: 1 } }));
	let seen: Record<string, unknown> | undefined;
	bus.on("beforeToolCall", c => {
		seen = c.args;
		return undefined;
	});
	const r = await bus.toHooks().beforeToolCall?.(ctx);
	assert.deepEqual(seen, { a: 1 });
	assert.deepEqual(r?.args, { a: 1 });
});

void test("a throwing handler is skipped and reported, chain continues", async () => {
	const errors: string[] = [];
	const bus = new HookBus({
		onError: (e, event, source) =>
			errors.push(`${event}:${source}:${e.message}`),
	});
	bus.on(
		"shouldStopAfterTurn",
		() => {
			throw new Error("boom");
		},
		{ source: "bad" },
	);
	bus.on("shouldStopAfterTurn", () => true, { source: "good" });
	const r = await bus.toHooks().shouldStopAfterTurn?.({
		messages: [],
		iteration: 1,
		hadToolCalls: false,
	});
	assert.equal(r, true);
	assert.deepEqual(errors, ["shouldStopAfterTurn:bad:boom"]);
});

void test("beforeAgentStart threads context through every layer", async () => {
	const bus = new HookBus();
	bus.on("beforeAgentStart", ({ messages }) => ({
		messages: [...messages, { role: "user", content: "first" }],
		systemPrompt: "layer one",
	}));
	bus.on("beforeAgentStart", ({ messages, systemPrompt }) => ({
		messages: [...messages, { role: "user", content: systemPrompt }],
		systemPrompt: "layer two",
	}));
	const result = await bus.toHooks().beforeAgentStart?.({
		prompt: "prompt",
		systemPrompt: "base",
		messages: [],
	});
	assert.equal(result?.systemPrompt, "layer two");
	assert.deepEqual(
		result?.messages?.map(message =>
			message && "content" in message ? message.content : "",
		),
		["first", "layer one"],
	);
});

void test("provider request patches retain every supported field", async () => {
	const bus = new HookBus();
	bus.on("beforeProviderRequest", () => ({
		headers: { authorization: "one", remove: undefined },
		maxRetries: 4,
		cacheRetention: "persistent",
		metadata: { first: true },
	}));
	bus.on("beforeProviderRequest", () => ({
		timeoutMs: 2500,
		metadata: { second: true },
		transport: "sse",
	}));
	const result = await bus.toHooks().beforeProviderRequest?.({
		model: "test",
		sessionId: "session",
		iteration: 1,
		streamOptions: {},
	});
	assert.deepEqual(result, {
		headers: { authorization: "one", remove: undefined },
		timeoutMs: 2500,
		maxRetries: 4,
		cacheRetention: "persistent",
		metadata: { first: true, second: true },
		transport: "sse",
	});
});

void test("beforeCompact cancellation short-circuits later hooks", async () => {
	const bus = new HookBus();
	let laterRan = false;
	bus.on("beforeCompact", () => ({ cancel: true }));
	bus.on("beforeCompact", () => {
		laterRan = true;
		return { summary: "unused" };
	});
	const result = await bus.toHooks().beforeCompact?.({
		messages: [],
		tokensBefore: 100,
		reason: "manual",
	});
	assert.deepEqual(result, { cancel: true });
	assert.equal(laterRan, false);
});

void test("handlers run in registration order", async () => {
	const bus = new HookBus();
	const order: string[] = [];
	bus.on(
		"afterProviderResponse",
		() => {
			order.push("first");
		},
		{ id: "first", source: "builtin" },
	);
	bus.on(
		"afterProviderResponse",
		() => {
			order.push("second");
		},
		{ id: "second", source: "extension" },
	);
	await bus.toHooks().afterProviderResponse?.({
		content: "",
		toolCallCount: 0,
		iteration: 1,
		model: "test",
		stopReason: "stop",
	});
	assert.deepEqual(order, ["first", "second"]);
	assert.throws(
		() => bus.on("afterProviderResponse", () => {}, { id: "first" }),
		/Duplicate hook handler id/,
	);
});

void test("a parent abort signal propagates to the handler's signal", async () => {
	const bus = new HookBus();
	let observedAbort = false;
	bus.on(
		"afterProviderResponse",
		async (_ctx, signal) => {
			await new Promise<void>(resolve => {
				signal?.addEventListener(
					"abort",
					() => {
						observedAbort = true;
						resolve();
					},
					{ once: true },
				);
			});
		},
		{ id: "abort-aware" },
	);
	const controller = new AbortController();
	const run = bus.toHooks().afterProviderResponse?.(
		{
			content: "",
			toolCallCount: 0,
			iteration: 1,
			model: "test",
			stopReason: "stop",
		},
		controller.signal,
	);
	controller.abort();
	await run;
	assert.equal(observedAbort, true);
});

void test("policy modules expose kind, timing, and lifecycle decisions", async () => {
	const evaluations: Array<{ policyId: string; kind: string; status: string }> =
		[];
	const bus = new HookBus({
		onPolicyEvaluation: evaluation => evaluations.push(evaluation),
	});
	const unregister = bus.registerPolicy({
		id: "verified-stop",
		description: "Require independent verification before stopping",
		kind: "agent",
		hooks: {
			shouldStopAfterTurn: ({ hadToolCalls }) => hadToolCalls,
		},
	});
	assert.equal(
		await bus.toHooks().shouldStopAfterTurn?.({
			messages: [],
			iteration: 1,
			hadToolCalls: true,
		}),
		true,
	);
	assert.equal(evaluations.length, 1);
	assert.deepEqual(
		{
			policyId: evaluations[0]?.policyId,
			kind: evaluations[0]?.kind,
			status: evaluations[0]?.status,
		},
		{ policyId: "verified-stop", kind: "agent", status: "completed" },
	);
	unregister();
	assert.equal(
		await bus.toHooks().shouldStopAfterTurn?.({
			messages: [],
			iteration: 2,
			hadToolCalls: true,
		}),
		undefined,
	);
});

void test("policy module deadlines fail open and report a timeout", async () => {
	const statuses: string[] = [];
	const bus = new HookBus({
		onPolicyEvaluation: evaluation => statuses.push(evaluation.status),
	});
	bus.registerPolicy({
		id: "slow-judge",
		description: "A bounded prompt policy",
		kind: "prompt",
		timeoutMs: 5,
		hooks: {
			shouldStopAfterTurn: async (_context, signal) =>
				await new Promise<boolean>(resolve => {
					signal?.addEventListener("abort", () => resolve(false), {
						once: true,
					});
				}),
		},
	});
	assert.equal(
		await bus.toHooks().shouldStopAfterTurn?.({
			messages: [],
			iteration: 1,
			hadToolCalls: false,
		}),
		undefined,
	);
	assert.deepEqual(statuses, ["timed_out"]);
});
