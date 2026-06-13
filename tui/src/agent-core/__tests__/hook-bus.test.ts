import assert from "node:assert/strict";
import { test } from "node:test";
import { HookBus } from "../hook-bus.ts";
import type { ToolCall } from "../types.ts";

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
		onError: (e, event, source) => errors.push(`${event}:${source}:${e.message}`),
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
		continuationCount: 0,
		isContinuation: false,
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
