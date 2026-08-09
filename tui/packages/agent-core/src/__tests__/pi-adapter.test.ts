import { describe, expect, it, mock } from "bun:test";
import type { EventBus } from "../extensions/event-bus.ts";
import { PiAdapter } from "../extensions/pi-adapter.ts";
import type {
	ExtensionToolResult,
	ExtensionAPI as LApi,
	RegisteredCommand as LCommand,
	ExtensionContext as LContext,
	ExtensionEvent as LEvent,
	ExtensionEventType as LEventType,
	ExtensionEventHandler as LHandler,
	RegisteredTool as LTool,
} from "../extensions/types.ts";

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

describe("PiAdapter", () => {
	it("should convert TypeBox String schema to JSON Schema", () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const _adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		api.registerTool({
			name: "test_tool",
			description: "test",
			parameters: {
				type: "object",
				properties: { name: { type: "string" } },
			} as any,
			execute: async () =>
				({ content: "ok" }) as unknown as ExtensionToolResult,
		});

		// Verify tool was registered with Logician
		expect(api.registerTool.mock.calls.length).toBe(1);
	});

	it("should create Pi-compatible UI wrapper", () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		// UI methods should exist
		expect(piApi.registerTool).toBeDefined();
		expect(piApi.registerCommand).toBeDefined();
		expect(piApi.on).toBeDefined();
	});

	it("should forward Pi tool registration to Logician", () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();
		let _capturedTool: any = null;

		piApi.registerTool({
			name: "greet",
			label: "Greet",
			description: "Greet someone",
			parameters: { type: "object", properties: { name: { type: "string" } } },
			execute: async (toolCallId, params) => {
				_capturedTool = { toolCallId, params };
				return { content: [{ type: "text", text: `Hello, ${params.name}!` }] };
			},
		});

		expect(api.registerTool.mock.calls.length).toBe(1);
		const forwarded = api.registerTool.mock.calls[0][0] as LTool;
		expect(forwarded.name).toBe("greet");
		expect(forwarded.description).toBe("Greet someone");
	});

	it("should forward Pi command registration to Logician", () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		piApi.registerCommand("hello", {
			description: "Say hello",
			handler: async () => {},
		});

		expect(api.registerCommand.mock.calls.length).toBe(1);
	});

	it("should translate session_start event to Pi format", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		let receivedEvent: any = null;
		piApi.on("session_start", async (event, _ctx) => {
			receivedEvent = event;
		});

		// Emit from Logician
		await adapter.emitFromLogician({
			type: "session_start",
			context: { sessionId: "test", cwd: "/test" },
		});

		expect(receivedEvent).not.toBeNull();
		expect(receivedEvent.type).toBe("session_start");
	});

	it("should translate agent_start event to Pi format", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		let receivedEvent: any = null;
		piApi.on("agent_start", async (event, _ctx) => {
			receivedEvent = event;
		});

		await adapter.emitFromLogician({
			type: "agent_start",
			context: { sessionId: "test", cwd: "/test" },
		});

		expect(receivedEvent).not.toBeNull();
		expect(receivedEvent.type).toBe("agent_start");
	});

	it("should translate tool_call event for blocking", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		let receivedEvent: any = null;
		piApi.on("tool_execution_start", async (event, _ctx) => {
			receivedEvent = event;
			if (
				event.toolName === "bash" &&
				(event.input as any).command?.includes("rm -rf")
			) {
				return { block: true, reason: "Dangerous command" };
			}
		});

		await adapter.emitFromLogician({
			type: "tool_execution_start",
			context: {
				sessionId: "test",
				cwd: "/test",
				tool_name: "bash",
				toolInput: { command: "rm -rf /" },
				toolCallId: "tc-1",
			},
		});

		expect(receivedEvent).not.toBeNull();
		expect(receivedEvent.type).toBe("tool_execution_start");
		expect(receivedEvent.toolName).toBe("bash");
	});

	it("should translate turn events with indices", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		const startEvents: any[] = [];
		const endEvents: any[] = [];

		piApi.on("turn_start", async (event, _ctx) => {
			startEvents.push(event);
		});
		piApi.on("turn_end", async (event, _ctx) => {
			endEvents.push(event);
		});

		await adapter.emitFromLogician({
			type: "turn_start",
			context: { sessionId: "test", cwd: "/test", turnIndex: 5 },
		});
		await adapter.emitFromLogician({
			type: "turn_end",
			context: { sessionId: "test", cwd: "/test", turnIndex: 5 },
		});

		expect(startEvents.length).toBe(1);
		expect(startEvents[0].type).toBe("turn_start");
		expect(endEvents.length).toBe(1);
		expect(endEvents[0].type).toBe("turn_end");
	});

	it("should handle getFlag returning registered defaults", () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		piApi.registerFlag("my-flag", {
			description: "A flag",
			type: "boolean",
			default: true,
		});

		expect(piApi.getFlag("my-flag")).toBe(true);
		expect(piApi.getFlag("unknown-flag")).toBeUndefined();
	});

	it("should return registered tools in getAllTools", () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
		});

		const piApi = adapter.getApi();

		piApi.registerTool({
			name: "tool1",
			label: "Tool 1",
			description: "First tool",
			parameters: { type: "object", properties: {} },
			execute: async () => ({ content: [{ type: "text", text: "" }] }),
		});
		piApi.registerTool({
			name: "tool2",
			label: "Tool 2",
			description: "Second tool",
			parameters: { type: "object", properties: {} },
			execute: async () => ({ content: [{ type: "text", text: "" }] }),
		});

		const tools = piApi.getAllTools();
		expect(tools.length).toBe(2);
		expect(tools[0].name).toBe("tool1");
		expect(tools[1].name).toBe("tool2");
	});

	it("should delegate runtime controls through the Pi compatibility port", async () => {
		const api = createMockLogicianApi();
		const ctx = createMockLogicianContext();
		const sent: string[] = [];
		let selectedModel: unknown;
		let thinkingLevel: unknown = "medium";
		const adapter = new PiAdapter(api, ctx, {
			sessionId: "test",
			cwd: "/test",
			runtime: {
				sendUserMessage: content => sent.push(content),
				getActiveTools: () => ["read", "bash"],
				setModel: async model => {
					selectedModel = model;
					return true;
				},
				getThinkingLevel: () => thinkingLevel,
				setThinkingLevel: level => {
					thinkingLevel = level;
				},
			},
		});
		const piApi = adapter.getApi();

		piApi.sendUserMessage("continue");
		expect(sent).toEqual(["continue"]);
		expect(piApi.getActiveTools()).toEqual(["read", "bash"]);
		expect(await piApi.setModel({ id: "next-model" })).toBe(true);
		expect(selectedModel).toEqual({ id: "next-model" });
		piApi.setThinkingLevel("high");
		expect(piApi.getThinkingLevel()).toBe("high");
	});
});
