import { describe, expect, test } from "bun:test";
import type { AgentConfig } from "@logician/log-core";
import { ConversationIdentity } from "../../runtime/bridge/application/conversation-identity.ts";

describe("ConversationIdentity", () => {
	test("changes every session-aware adapter through one interface", () => {
		const config = {} as AgentConfig;
		const calls: string[] = [];
		const durable = { append: async () => {} } as never;
		const identity = new ConversationIdentity("provisional", {
			cwd: "/tmp/project",
			config: () => config,
			sessions: () => ({
				use: (id: string, store: unknown) =>
					calls.push(`session:${id}:${store === durable}`),
			}),
			events: () => ({
				setSessionId: (id: string) => calls.push(`events:${id}`),
			}),
		});

		expect(identity.use("conversation-1", durable)).toBe(true);
		expect(identity.id).toBe("conversation-1");
		expect(config.hookSessionId).toBe("conversation-1");
		expect(config.hookTranscriptPath).toBe(identity.transcript);
		expect(config.eventLogPath).toBe(
			identity.transcript.replace(/\.jsonl$/, ".events.jsonl"),
		);
		expect(calls).toEqual([
			"events:conversation-1",
			"session:conversation-1:true",
		]);
	});

	test("rejects an empty identity without changing adapters", () => {
		const calls: string[] = [];
		const identity = new ConversationIdentity("current", {
			cwd: "/tmp/project",
			config: () => undefined,
			sessions: () => undefined,
			events: () => ({ setSessionId: (id: string) => calls.push(id) }),
		});
		expect(identity.use("  ")).toBe(false);
		expect(identity.id).toBe("current");
		expect(calls).toEqual([]);
	});
});
