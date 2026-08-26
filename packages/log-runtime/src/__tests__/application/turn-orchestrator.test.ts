import { describe, expect, test } from "bun:test";
import type { RuntimeEvent } from "@logician/log-core/events";
import { TurnOrchestrator } from "../../runtime/bridge/application/turn-orchestrator.ts";

describe("TurnOrchestrator", () => {
	test("orders prerequisites while keeping MCP discovery non-blocking", async () => {
		const calls: string[] = [];
		let releaseMcp: () => void = () => {};
		const mcp = new Promise<void>(resolve => {
			releaseMcp = resolve;
		});
		const turns = new TurnOrchestrator({
			extensionsReady: async () => {
				calls.push("extensions");
			},
			hasSession: () => false,
			steer: () => {},
			emit: () => {},
			ensureStartup: async () => {
				calls.push("startup");
			},
			isMcpLoaded: () => false,
			loadMcp: () => {
				calls.push("mcp");
				return mcp;
			},
			reportMcpError: () => {},
			runTurn: async message => {
				calls.push(`turn:${message}`);
			},
		});

		await turns.submit("hello");
		expect(calls).toEqual(["extensions", "startup", "mcp", "turn:hello"]);
		releaseMcp();
	});

	test("routes a concurrent submission to steering", async () => {
		let releaseTurn: () => void = () => {};
		const activeTurn = new Promise<void>(resolve => {
			releaseTurn = resolve;
		});
		const steered: string[] = [];
		const events: RuntimeEvent[] = [];
		const turns = new TurnOrchestrator({
			extensionsReady: async () => {},
			hasSession: () => true,
			steer: message => steered.push(message),
			emit: event => events.push(event),
			ensureStartup: async () => {},
			isMcpLoaded: () => true,
			loadMcp: async () => {},
			reportMcpError: () => {},
			runTurn: () => activeTurn,
		});

		const first = turns.submit("first");
		await Promise.resolve();
		await turns.submit("change direction");
		expect(steered).toEqual(["change direction"]);
		expect(events).toContainEqual({
			type: "steered",
			message: "change direction",
		});
		releaseTurn();
		await first;
	});
});
