import { describe, expect, it, mock } from "bun:test";
import { PiAdapter } from "../../adapters/pi/index.ts";
import type { EventBus } from "../../core/extension/event-bus.ts";
import type {
	ExtensionAPI as LApi,
	RegisteredCommand as LCommand,
	ExtensionContext as LContext,
	ExtensionEvent as LEvent,
	ExtensionEventType as LEventType,
	ExtensionEventHandler as LHandler,
	RegisteredTool as LTool,
} from "../../core/extension/types.ts";

// Create mock logician API
function createMockLogicianApi(): LApi {
	const registeredTools: LTool[] = [];
	const registeredCommands: LCommand[] = [];
	const eventBus = { clear: mock(() => {}) } as unknown as EventBus;
	const handlers = new Map<string, LHandler[]>();

	return {
		on: mock((event: LEventType, handler: LHandler) => {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
			return () => {};
		}),
		registerTool: mock((tool: LTool) => {
			registeredTools.push(tool);
		}),
		registerCommand: mock((cmd: LCommand) => {
			registeredCommands.push(cmd);
		}),
		emit: mock(async (event: LEvent) => {
			const list = handlers.get(event.type) ?? [];
			for (const h of list) {
				await h(event, {} as LContext);
			}
		}),
		events: eventBus,
		info: { name: "mock", path: "/mock" },
	} as unknown as LApi;
}

function createMockLogicianContext(): LContext {
	return {
		ui: {
			notify: mock(() => {}),
			confirm: mock(async () => false),
			input: mock(async () => undefined),
			select: mock(async () => undefined),
		},
		state: {
			get: mock(async () => undefined),
			set: mock(async () => {}),
			delete: mock(async () => {}),
			keys: mock(async () => []),
		},
		cwd: "/test",
		sessionId: "test-session-1",
	};
}

describe("PiAdapter - input event", () => {
	it("should emit input event to registered handlers", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		let _receivedText = "";
		let _receivedSource = "";

		// Handler that only intercepts specific input
		piApi.onInput(async (text, _images, source, _hasUI, _ui) => {
			_receivedText = text;
			_receivedSource = source;
			if (text.includes("block")) {
				return { action: "handled" };
			}
			return { action: "transform", text: `transformed: ${text}` };
		});

		// Test handled case
		const handledResult = await adapter.emitInputEvent(
			"block this",
			[],
			"interactive",
		);
		expect(handledResult?.action).toBe("handled");

		// Test transform case
		const transformResult = await adapter.emitInputEvent(
			"hello",
			[],
			"interactive",
		);
		expect(transformResult?.action).toBe("transform");
		expect(transformResult?.text).toBe("transformed: hello");
	});
});

describe("PiAdapter - user_bash event", () => {
	it("should emit user_bash event to registered handlers", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		piApi.onUserBash(
			async (command, _excludeFromContext, _cwd, _hasUI, _ui) => {
				if (command.includes("blocked")) {
					return {
						action: "replace",
						result: { output: "intercepted", exitCode: 0, cancelled: false },
					};
				}
				return null; // continue
			},
		);

		// Test replace case
		const replaceResult = await adapter.emitUserBashEvent(
			"blocked command",
			false,
		);
		expect(replaceResult?.action).toBe("replace");
		expect(replaceResult?.result?.output).toBe("intercepted");

		// Test continue case
		const continueResult = await adapter.emitUserBashEvent(
			"normal command",
			true,
		);
		expect(continueResult).toBeNull();
	});
});

describe("PiAdapter - project_trust event", () => {
	it("should emit project_trust event to registered handlers", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		piApi.onProjectTrust(async (cwd, _hasUI, _ui) => {
			if (cwd.includes("trusted")) {
				return { trusted: "yes", remember: true };
			}
			if (cwd.includes("denied")) {
				return { trusted: "no" };
			}
			return { trusted: "undecided" }; // continue
		});

		// Test trust yes
		const trustYes = await adapter.emitProjectTrustEvent("/trusted/path");
		expect(trustYes?.trusted).toBe("yes");
		expect(trustYes?.remember).toBe(true);

		// Test trust no
		const trustNo = await adapter.emitProjectTrustEvent("/denied/path");
		expect(trustNo?.trusted).toBe("no");

		// Test undecided
		const undecided = await adapter.emitProjectTrustEvent("/neutral/path");
		expect(undecided?.trusted).toBe("undecided");
	});
});

describe("PiAdapter - handler chaining", () => {
	it("should chain multiple handlers and use first non-null result", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		let firstCalled = false;
		let secondCalled = false;

		// First handler returns null (continue)
		piApi.onInput(async _text => {
			firstCalled = true;
			return null; // continue to next handler
		});

		// Second handler returns transform
		piApi.onInput(async text => {
			secondCalled = true;
			return { action: "transform", text: `processed: ${text}` };
		});

		const result = await adapter.emitInputEvent("hello", [], "interactive");
		expect(firstCalled).toBe(true);
		expect(secondCalled).toBe(true);
		expect(result?.action).toBe("transform");
		expect(result?.text).toBe("processed: hello");
	});

	it("should stop at first handler that returns non-null", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		let firstCalled = false;
		let secondCalled = false;

		// First handler returns handled (short-circuit)
		piApi.onInput(async () => {
			firstCalled = true;
			return { action: "handled" };
		});

		// Second handler should NOT be called
		piApi.onInput(async () => {
			secondCalled = true;
			return { action: "transform" };
		});

		const result = await adapter.emitInputEvent("hello", [], "interactive");
		expect(firstCalled).toBe(true);
		expect(secondCalled).toBe(false);
		expect(result?.action).toBe("handled");
	});
});

describe("PiAdapter - error handling", () => {
	it("should swallow handler errors and continue to next handler", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		// First handler throws
		piApi.onInput(async () => {
			throw new Error("handler error");
		});

		// Second handler returns valid result
		piApi.onInput(async () => {
			return { action: "transform", text: "recovered" };
		});

		const result = await adapter.emitInputEvent("hello", [], "interactive");
		expect(result?.action).toBe("transform");
		expect(result?.text).toBe("recovered");
	});
});
