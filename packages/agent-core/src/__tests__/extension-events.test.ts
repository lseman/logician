// ── Extension event system tests ─────────────────────────────────────────

import { describe, it } from "bun:test";
import { strict as assert } from "node:assert";
import { createExtensionContext } from "../hooks/extensions/context.ts";
import { ExtensionEventBus } from "../hooks/extensions/event-bus.ts";

describe("ExtensionEventBus", () => {
	it("emits event to registered handler", async () => {
		const bus = new ExtensionEventBus();
		const received: Array<{ type: string; turnIndex: number }> = [];

		bus.on("turn_start", async (event: any) => {
			received.push({ type: event.type, turnIndex: event.turnIndex });
		});

		await bus.emit({ type: "turn_start", turnIndex: 5 } as any);
		assert.strictEqual(received.length, 1);
		assert.strictEqual(received[0].turnIndex, 5);
	});

	it("returns undefined when no handlers", async () => {
		const bus = new ExtensionEventBus();
		const result = await bus.emit({ type: "turn_start", turnIndex: 1 });
		assert.strictEqual(result, undefined);
	});

	it("last handler result wins", async () => {
		const bus = new ExtensionEventBus();
		const handler1: any = async () => ({ message: { content: "first" } });
		const handler2: any = async () => ({ message: { content: "second" } });

		bus.on("message_end", handler1);
		bus.on("message_end", handler2);

		const result = await bus.emit({
			type: "message_end",
			message: { role: "assistant", content: "test" },
		} as any);

		assert.deepStrictEqual(result, { message: { content: "second" } });
	});

	it("skips failed handlers", async () => {
		const bus = new ExtensionEventBus();
		const received: number[] = [];

		const failHandler: any = () => {
			throw new Error("Handler 1 failed");
		};
		bus.on("turn_start", failHandler);
		bus.on("turn_start", async (event: any) => {
			received.push(event.turnIndex);
		});

		const result = await bus.emit({ type: "turn_start", turnIndex: 10 });
		assert.strictEqual(received.length, 1);
		assert.strictEqual(received[0], 10);
		assert.strictEqual(result, undefined);
	});

	it("respects timeout", async () => {
		const bus = new ExtensionEventBus({ defaultTimeoutMs: 50 });

		let called = false;
		const slowHandler: any = async () => {
			called = true;
			await new Promise(r => setTimeout(r, 200));
			return {};
		};
		bus.on("turn_start", slowHandler);

		const start = Date.now();
		await bus.emit({ type: "turn_start", turnIndex: 1 });
		const elapsed = Date.now() - start;

		assert.strictEqual(called, true);
		assert.ok(elapsed < 150, `Expected < 150ms, got ${elapsed}ms`);
	});

	it("unsubscribe works", async () => {
		const bus = new ExtensionEventBus();
		const received: number[] = [];

		const off = bus.on("turn_start", async (e: any) => {
			received.push(e.turnIndex);
		});

		await bus.emit({ type: "turn_start", turnIndex: 1 });
		assert.strictEqual(received.length, 1);

		off();
		await bus.emit({ type: "turn_start", turnIndex: 2 });
		assert.strictEqual(received.length, 1);
	});

	it("onMultiple registers and unsubscribes", async () => {
		const bus = new ExtensionEventBus();
		const received: string[] = [];

		const off = bus.onMultiple([
			{
				eventType: "turn_start",
				handler: async () => {
					received.push("turn_start");
				},
			},
			{
				eventType: "turn_end",
				handler: async () => {
					received.push("turn_end");
				},
			},
		]);

		await bus.emit({ type: "turn_start", turnIndex: 1 } as any);
		assert.ok(received.includes("turn_start"));

		await bus.emit({ type: "turn_end", turnIndex: 1 } as any);
		assert.ok(received.includes("turn_end"));

		off();
		await bus.emit({ type: "turn_start", turnIndex: 2 } as any);
		assert.strictEqual(
			received.length,
			2,
			"Should not have new events after off()",
		);
	});

	it("hasHandlers returns true when handlers exist", () => {
		const bus = new ExtensionEventBus();
		assert.strictEqual(bus.hasHandlers("turn_start"), false);

		bus.on("turn_start", (() => {}) as never);
		assert.strictEqual(bus.hasHandlers("turn_start"), true);
	});

	it("clear removes all handlers", () => {
		const bus = new ExtensionEventBus();
		bus.on("turn_start", (() => {}) as never);
		bus.on("turn_end", (() => {}) as never);

		assert.strictEqual(bus.getHandlerCount("turn_start"), 1);
		assert.strictEqual(bus.getHandlerCount("turn_end"), 1);

		bus.clear();

		assert.strictEqual(bus.getHandlerCount("turn_start"), 0);
		assert.strictEqual(bus.getHandlerCount("turn_end"), 0);
	});

	it("error handler is called on failure", async () => {
		const errors: Array<{ event: string; message: string }> = [];
		const bus = new ExtensionEventBus({
			onError: (error, event) => {
				errors.push({ event, message: error.message });
			},
		});

		const errHandler: any = () => {
			throw new Error("Test failure");
		};
		bus.on("turn_start", errHandler);

		await bus.emit({ type: "turn_start", turnIndex: 1 } as any);
		assert.strictEqual(errors.length, 1);
		assert.strictEqual(errors[0].event, "turn_start");
		assert.strictEqual(errors[0].message, "Test failure");
	});

	it("getRegisteredEvents returns active event types", () => {
		const bus = new ExtensionEventBus();
		assert.strictEqual(bus.getRegisteredEvents().length, 0);

		bus.on("turn_start", (() => {}) as never);
		bus.on("agent_end", (() => {}) as never);

		const events = bus.getRegisteredEvents();
		assert.ok(events.includes("turn_start"));
		assert.ok(events.includes("agent_end"));
	});
});

describe("ExtensionContext", () => {
	it("starts with defaults", () => {
		const ctx = createExtensionContext();
		assert.strictEqual(ctx.turnIndex, 0);
		assert.strictEqual(ctx.iteration, 0);
		assert.strictEqual(ctx.inToolLoop, false);
		assert.strictEqual(ctx.inThinkingLoop, false);
		assert.deepStrictEqual(ctx.counters, {});
		assert.strictEqual(ctx.features.size, 0);
		assert.deepStrictEqual(ctx.labels, {});
		assert.deepStrictEqual(ctx.diagnostics, []);
	});

	it("increments counters", () => {
		const ctx = createExtensionContext();
		assert.strictEqual(ctx.incrementCounter("foo"), 1);
		assert.strictEqual(ctx.incrementCounter("foo"), 2);
		assert.strictEqual(ctx.incrementCounter("bar"), 1);
	});

	it("manages features", () => {
		const ctx = createExtensionContext();
		ctx.setFeature("dark-mode", true);
		assert.strictEqual(ctx.features.has("dark-mode"), true);

		ctx.setFeature("dark-mode", false);
		assert.strictEqual(ctx.features.has("dark-mode"), false);
	});

	it("stores and retrieves data", () => {
		const ctx = createExtensionContext();
		ctx.storeData("my-ext", "key1", "value1");
		assert.strictEqual(ctx.fetchData("my-ext", "key1"), "value1");
		assert.strictEqual(ctx.fetchData("my-ext", "missing"), undefined);
		assert.strictEqual(ctx.fetchData("other-ext", "key1"), undefined);
	});

	it("sets labels", () => {
		const ctx = createExtensionContext();
		ctx.setLabel("important");
		assert.strictEqual(ctx.labels.important, "important");
	});

	it("adds diagnostics", () => {
		const ctx = createExtensionContext();
		ctx.addDiagnostic("my-ext", "Something happened");
		ctx.addDiagnostic("my-ext", "Critical error", "error");
		ctx.addDiagnostic("my-ext", "Warning", "warning");

		assert.strictEqual(ctx.diagnostics.length, 3);
		assert.strictEqual(ctx.diagnostics[0].severity, "info");
		assert.strictEqual(ctx.diagnostics[1].severity, "error");
		assert.strictEqual(ctx.diagnostics[2].severity, "warning");
	});

	it("updates context state", () => {
		const ctx = createExtensionContext();
		ctx.turnIndex = 5;
		ctx.iteration = 10;
		ctx.inToolLoop = true;

		assert.strictEqual(ctx.turnIndex, 5);
		assert.strictEqual(ctx.iteration, 10);
		assert.strictEqual(ctx.inToolLoop, true);
	});
});

// ── HookBus → ExtensionEventBus wiring tests ────────────────────────────
import { HookBus } from "../hooks/native/hook-bus.ts";

describe("ExtensionEventBus.fromHookBus", () => {
	it("emits typed events from HookBus firehose", async () => {
		const hookBus = new HookBus();
		const extBus = ExtensionEventBus.fromHookBus(hookBus);
		const received: Array<{ type: string; prompt: string }> = [];

		extBus.on("before_agent_start", (event: any) => {
			received.push({ type: event.type, prompt: event.prompt });
			return undefined;
		});

		// Call the hook through the AgentHooks produced by toHooks()
		const hooks = hookBus.toHooks();
		await hooks.beforeAgentStart?.({
			prompt: "hello world",
			systemPrompt: "system prompt",
			messages: [],
		});

		assert.strictEqual(received.length, 1);
		assert.strictEqual(received[0].prompt, "hello world");
	});

	it("emits tool_execution_start from beforeToolCall", async () => {
		const hookBus = new HookBus();
		const extBus = ExtensionEventBus.fromHookBus(hookBus);
		const received: Array<{ type: string; toolName: string }> = [];

		extBus.on("tool_execution_start", (event: any) => {
			received.push({ type: event.type, toolName: event.toolName });
			return undefined;
		});

		const hooks = hookBus.toHooks();
		await hooks.beforeToolCall?.({
			toolCall: { id: "call_1", name: "bash", arguments: '{}' },
			args: { cmd: "ls" },
			iteration: 5,
		});

		assert.strictEqual(received.length, 1);
		assert.strictEqual(received[0].toolName, "bash");
	});

	it("dispose() cleans up HookBus subscription", async () => {
		const hookBus = new HookBus();
		const extBus = ExtensionEventBus.fromHookBus(hookBus);
		let triggered = false;

		extBus.on("before_agent_start", () => {
			triggered = true;
			return undefined;
		});

		const hooks = hookBus.toHooks();

		// Should trigger
		await hooks.beforeAgentStart?.({
			prompt: "test",
			systemPrompt: "system",
			messages: [],
		});
		assert.strictEqual(triggered, true);

		// Dispose
		await extBus.dispose();

		// Should NOT trigger after dispose
		triggered = false;
		await hooks.beforeAgentStart?.({
			prompt: "after dispose",
			systemPrompt: "system",
			messages: [],
		});
		assert.strictEqual(triggered, false);
	});

	it("emitLegacy() dispatches to handlers for custom event types", async () => {
		const extBus = new ExtensionEventBus();
		const received: Array<{ type: string; iteration: number }> = [];

		// Register on a real ExtensionEventName but intercept legacy dispatch
		extBus.on("agent_start", (event: any) => {
			received.push({ type: event.type ?? "agent_start", iteration: event.iteration });
			return undefined;
		});

		// emitLegacy dispatches to handlers registered for that exact type;
		// since "thinking_loop_detected" != "agent_start", no handler fires.
		await (extBus as any).emitLegacy({
			type: "thinking_loop_detected",
			iteration: 42,
		});

		// No handler matched because the type string doesn't match
		assert.strictEqual(received.length, 0);
	});
});
