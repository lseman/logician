import { describe, expect, test } from "bun:test";
import type { AgentEvent } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import { RuntimeActivity } from "../../runtime/bridge/application/runtime-activity.ts";

describe("RuntimeActivity", () => {
	test("owns context, retry, repair, and subagent state", () => {
		const events: RuntimeEvent[] = [];
		const activity = new RuntimeActivity({
			emit: event => events.push(event),
			runPhase: () => "running",
		});
		activity.handle({
			type: "context_update",
			tokens: 42,
			maxTokens: 100,
		} as AgentEvent);
		activity.handle({
			type: "agent_retry_start",
			attempt: 2,
			maxRetries: 3,
		} as AgentEvent);
		activity.handle({
			type: "subagent_start",
			agentId: "worker-1",
		} as AgentEvent);

		expect(activity.context()).toEqual({ tokens: 42, maxTokens: 100 });
		expect(events.at(-1)).toMatchObject({
			type: "runtime_status",
			retry: "2/3",
			activeSubagents: 1,
		});
	});

	test("does not emit status before a session has a run phase", () => {
		const events: RuntimeEvent[] = [];
		const activity = new RuntimeActivity({
			emit: event => events.push(event),
			runPhase: () => undefined,
		});
		activity.handle({ type: "agent_retry_end" } as AgentEvent);
		expect(events.some(event => event.type === "runtime_status")).toBe(false);
	});
});
