import { describe, expect, test } from "bun:test";
import type { AgentConfig } from "@logician/log-core";
import { OpenAIBackend } from "@logician/log-core";
import { ExtensionRegistry } from "../../capabilities/extensions/extensions.ts";
import { ConversationSession } from "../../runtime/bridge/application/conversation-session.ts";

function createConversationSession() {
	const cwd = process.cwd();
	const config = {
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		tools: [],
		cwd,
		maxIterations: 2,
	} as AgentConfig;
	const extensions = new ExtensionRegistry({
		sessionId: "session-1",
		cwd,
		projectTrusted: false,
		extensionDirs: { paths: [] },
	});
	let contextChanges = 0;
	const sessions = new ConversationSession(
		{
			config: () => config,
			backend: new OpenAIBackend({
				baseUrl: config.baseUrl,
				model: config.model,
			}),
			extensions: () => extensions,
			emit: () => {},
			contextChanged: () => contextChanges++,
			contextCompacted: () => {},
		},
		"session-1",
	);
	return { sessions, contextChanges: () => contextChanges };
}

describe("ConversationSession", () => {
	test("owns lazy construction and history restoration", () => {
		const state = createConversationSession();
		expect(state.sessions.current).toBeNull();
		expect(
			state.sessions.restoreHistory([{ role: "user", content: "restored" }]),
		).toBe(true);
		expect(state.sessions.current?.messages).toEqual([
			{ role: "user", content: "restored" },
		]);
		expect(state.contextChanges()).toBe(1);
	});

	test("owns queue mutation and clearing", () => {
		const { sessions } = createConversationSession();
		sessions.ensure();
		sessions.followUp("later");
		sessions.nextTurn("next");
		expect(sessions.queues().followUp).toEqual(["later"]);
		expect(sessions.queues().nextTurn).toEqual(["next"]);
		expect(sessions.clearQueues()).toEqual({
			steering: [],
			followUp: ["later"],
			nextTurn: ["next"],
		});
	});

	test("drops the owned session as one lifecycle operation", () => {
		const { sessions } = createConversationSession();
		sessions.ensure();
		expect(sessions.current).not.toBeNull();
		sessions.clearAndDrop();
		expect(sessions.current).toBeNull();
	});
});
