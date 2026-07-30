import assert from "node:assert/strict";
import { test } from "node:test";
import { HookBus } from "../hooks/native/hook-bus.ts";
import type { ToolCall } from "../agent/types.ts";

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
	bus.on("beforeToolCall", (c) => {
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

void test("a slow handler is timed out and skipped", async () => {
	const errors: string[] = [];
	const bus = new HookBus({
		defaultTimeoutMs: 30,
		onError: (e) => errors.push(e.message),
	});
	bus.on(
		"afterToolCall",
		() => new Promise(() => {}), // never settles
		{ source: "stuck" },
	);
	bus.on("afterToolCall", () => ({ content: "patched" }));
	const r = await bus.toHooks().afterToolCall?.({
		...ctx,
		result: "raw",
		isError: false,
	});
	assert.equal(r?.content, "patched");
	assert.equal(errors.length, 1);
	assert.match(errors[0], /timed out/);
});

void test("per-registration timeout overrides bus default", async () => {
	const bus = new HookBus({ defaultTimeoutMs: 10 });
	bus.on(
		"getSteeringMessages",
		async () => {
			await new Promise((r) => setTimeout(r, 30));
			return [{ role: "user" as const, content: "late but allowed" }];
		},
		{ timeoutMs: 200 },
	);
	const r = await bus
		.toHooks()
		.getSteeringMessages?.({ messages: [], iteration: 1 });
	assert.equal(r?.[0]?.content, "late but allowed");
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
	assert.deepEqual(result?.messages?.map((message) =>
		message && "content" in message ? message.content : "",
	), [
		"first",
		"layer one",
	]);
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

void test("handlers use deterministic priority ordering and stable diagnostics", async () => {
	const bus = new HookBus();
	const order: string[] = [];
	bus.on("afterProviderResponse", () => { order.push("normal"); }, { id: "normal", source: "extension" });
	bus.on("afterProviderResponse", () => { order.push("policy"); }, { id: "policy", source: "builtin", priority: 100 });
	await bus.toHooks().afterProviderResponse?.({ content: "", toolCallCount: 0, iteration: 1, model: "test", stopReason: "stop" });
	assert.deepEqual(order, ["policy", "normal"]);
	assert.deepEqual(bus.getDiagnostics().map((item) => item.id), ["policy", "normal"]);
	assert.throws(() => bus.on("afterProviderResponse", () => {}, { id: "policy" }), /Duplicate hook handler id/);
});

void test("dispose runs cleanups once and rejects new registrations", async () => {
	const bus = new HookBus();
	const order: number[] = [];
	bus.addCleanup(() => { order.push(1); });
	bus.addCleanup(async () => { order.push(2); });
	await bus.dispose();
	await bus.dispose();
	assert.deepEqual(order, [2, 1]);
	assert.throws(() => bus.on("afterProviderResponse", () => {}), /disposed/);
});

void test("timeout aborts the handler signal instead of only abandoning its promise", async () => {
	const bus = new HookBus({ defaultTimeoutMs: 15 });
	let observedAbort = false;
	bus.on("afterProviderResponse", async (_ctx, signal) => {
		await new Promise<void>((resolve) => {
			signal?.addEventListener("abort", () => { observedAbort = true; resolve(); }, { once: true });
		});
	}, { id: "abort-aware" });
	await bus.toHooks().afterProviderResponse?.({ content: "", toolCallCount: 0, iteration: 1, model: "test", stopReason: "stop" });
	assert.equal(observedAbort, true);
	assert.equal(bus.getDiagnostics()[0]?.timeouts, 1);
});
