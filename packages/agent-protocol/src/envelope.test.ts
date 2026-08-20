import { describe, expect, test } from "bun:test";
import {
	AGENT_PROTOCOL_VERSION,
	createNotification,
	isAgentProtocolNotification,
} from "./envelope.ts";

describe("agent protocol envelope", () => {
	test("creates versioned ordered notifications", () => {
		const notification = createNotification(
			{ type: "phase", state: "thinking" },
			4,
			123,
		);
		expect(notification).toEqual({
			protocolVersion: AGENT_PROTOCOL_VERSION,
			sequence: 4,
			timestamp: 123,
			event: { type: "phase", state: "thinking" },
		});
		expect(isAgentProtocolNotification(notification)).toBe(true);
	});

	test("rejects incompatible protocol versions", () => {
		expect(
			isAgentProtocolNotification({
				protocolVersion: 99,
				sequence: 1,
				timestamp: 123,
				event: { type: "phase", state: "ready" },
			}),
		).toBe(false);
	});
});
