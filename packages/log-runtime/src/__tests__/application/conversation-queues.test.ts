import { describe, expect, test } from "bun:test";
import type { AgentSession } from "@logician/log-core/session";
import { ConversationQueues } from "../../runtime/bridge/application/conversation-queues.ts";

describe("ConversationQueues", () => {
	test("returns safe empty values without a live session", () => {
		const queues = new ConversationQueues(() => null);

		expect(queues.snapshot()).toEqual({
			steering: [],
			followUp: [],
			nextTurn: [],
		});
		expect(queues.clear()).toEqual({
			steering: [],
			followUp: [],
			nextTurn: [],
		});
		expect(queues.flushSteeringNow()).toBe(0);
		expect(queues.drop(0)).toBeUndefined();
	});

	test("delegates queue operations to the current session", () => {
		const calls: string[] = [];
		const session = {
			steer: (message: string) => calls.push(`steer:${message}`),
			followUp: (message: string) => calls.push(`followUp:${message}`),
			nextTurn: (message: string) => calls.push(`nextTurn:${message}`),
			getQueues: () => ({
				steering: ["now"],
				followUp: ["later"],
				nextTurn: ["next"],
			}),
			flushSteeringNow: () => 1,
			dropQueuedMessage: (index: number) => `dropped:${index}`,
		} as unknown as AgentSession;
		const queues = new ConversationQueues(() => session);

		queues.steer("a");
		queues.followUp("b");
		queues.nextTurn("c");

		expect(calls).toEqual(["steer:a", "followUp:b", "nextTurn:c"]);
		expect(queues.snapshot().nextTurn).toEqual(["next"]);
		expect(queues.flushSteeringNow()).toBe(1);
		expect(queues.drop(2)).toBe("dropped:2");
	});
});
