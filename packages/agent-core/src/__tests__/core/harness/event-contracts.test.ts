import { describe, expect, test } from "bun:test";
import { isExtensionLifecycleEvent } from "../../../core/events/contracts.ts";
import type { AgentEvent } from "../../../core/types/types-messages.ts";

describe("event contracts", () => {
	test("keeps runtime-only deltas out of the extension lifecycle protocol", () => {
		expect(
			isExtensionLifecycleEvent({ type: "agent_start" } as AgentEvent),
		).toBe(true);
		expect(
			isExtensionLifecycleEvent({
				type: "text_delta",
				turnId: "turn-1",
				delta: "x",
			}),
		).toBe(false);
	});
});
