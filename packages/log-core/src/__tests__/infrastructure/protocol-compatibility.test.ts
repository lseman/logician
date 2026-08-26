import { describe, expect, test } from "bun:test";
import {
	createNotification,
	isAgentProtocolNotification,
} from "../../system/types/types-protocol.ts";

describe("agent protocol compatibility", () => {
	test("accepts version-one notifications produced before correlation metadata", () => {
		expect(
			isAgentProtocolNotification({
				protocolVersion: 1,
				sequence: 1,
				timestamp: 1,
				event: { type: "phase", state: "ready" },
			}),
		).toBe(true);
	});

	test("adds correlation without changing protocol version", () => {
		const notification = createNotification(
			{ type: "phase", state: "ready" },
			2,
			3,
			{ sessionId: "s", runId: "r", turnId: "t" },
		);

		expect(notification.protocolVersion).toBe(1);
		expect(notification.correlation).toEqual({
			sessionId: "s",
			runId: "r",
			turnId: "t",
		});
		expect(isAgentProtocolNotification(notification)).toBe(true);
	});
});
