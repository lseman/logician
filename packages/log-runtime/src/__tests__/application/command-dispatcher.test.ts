import { describe, expect, test } from "bun:test";
import type { RuntimeEvent } from "@logician/log-core/events";
import { CommandDispatcher } from "../../runtime/bridge/application/command-dispatcher.ts";

function createDispatcher() {
	const messages: string[] = [];
	const events: RuntimeEvent[] = [];
	const errors: unknown[] = [];
	let reloads = 0;
	const dispatcher = new CommandDispatcher({
		session: () => null,
		skills: () => [
			{
				name: "review",
				displayName: "Review",
				description: "Review",
				content: "Review $ARGUMENTS",
				filePath: "/tmp/review/SKILL.md",
				baseDir: "/tmp/review",
				slashName: "review",
				disableModelInvocation: false,
				source: "path",
			},
		],
		prompts: () => [
			{
				name: "explain",
				description: "Explain",
				content: "Explain this",
				filePath: "/tmp/explain.md",
				slashName: "explain",
			},
		],
		sendMessage: async message => {
			messages.push(message);
		},
		reload: async () => {
			reloads++;
		},
		emit: event => events.push(event),
		reportError: error => errors.push(error),
	});
	return { dispatcher, messages, events, errors, reloads: () => reloads };
}

describe("CommandDispatcher", () => {
	test("routes reload separately from ordinary slash input", async () => {
		const state = createDispatcher();
		state.dispatcher.dispatchSlash(" /reload ");
		state.dispatcher.dispatchSlash("/status");
		await Promise.resolve();
		expect(state.reloads()).toBe(1);
		expect(state.messages).toEqual(["/status"]);
	});

	test("expands skills and prompts before sending", async () => {
		const state = createDispatcher();
		expect(state.dispatcher.invokeSkill("review", "src/index.ts")).toBe(true);
		expect(state.dispatcher.invokePrompt("explain", "briefly")).toBe(true);
		expect(state.dispatcher.invokePrompt("missing", "")).toBe(false);
		await Promise.resolve();
		expect(state.messages[0]).toContain("Review src/index.ts");
		expect(state.messages[1]).toBe("Explain this\n\nbriefly");
	});
});
