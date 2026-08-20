import { describe, expect, test } from "bun:test";
import {
	extensionHooks,
	runControlHooks,
} from "../../../core/hooks/contracts.ts";
import type { AgentHooks } from "../../../core/types/types-messages.ts";

describe("hook contracts", () => {
	test("separates extension interception from run control", () => {
		const beforeToolCall = async () => undefined;
		const shouldStopAfterTurn = async () => true;
		const hooks: AgentHooks = { beforeToolCall, shouldStopAfterTurn };

		expect(extensionHooks(hooks)).toEqual({
			beforeAgentStart: undefined,
			beforeToolCall,
			afterToolCall: undefined,
			transformContext: undefined,
			beforeProviderRequest: undefined,
			beforeProviderPayload: undefined,
			afterProviderResponse: undefined,
			beforeCompact: undefined,
		});
		expect(runControlHooks(hooks)).toEqual({
			prepareNextTurn: undefined,
			shouldStopAfterTurn,
			getSteeringMessages: undefined,
			getFollowUpMessages: undefined,
		});
	});
});
