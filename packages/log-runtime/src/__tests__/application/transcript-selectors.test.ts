import { describe, expect, test } from "bun:test";
import type { SessionState, Turn } from "../../runtime/transcript/model.ts";
import {
	selectAssistantContent,
	selectAssistantThinking,
	selectAssistantTools,
	selectCurrentTurn,
	selectMessageCount,
	selectStreamingContent,
	selectStreamingThinking,
} from "../../runtime/transcript/selectors.ts";

const turn: Turn = {
	id: "turn-1",
	userMessage: { type: "user", content: "question" },
	assistantMessage: {
		type: "assistant",
		isComplete: false,
		chunks: [
			{ seq: 0, type: "thinking", contentText: "first", isComplete: true },
			{ seq: 1, type: "thinking", contentText: "second", isComplete: false },
			{ seq: 2, type: "content", contentText: "done ", isComplete: true },
			{ seq: 3, type: "content", contentText: "stream", isComplete: false },
			{
				seq: 4,
				type: "tool",
				tool: { tool_name: "read_file", isError: false, isComplete: true },
				isComplete: true,
			},
		],
	},
	isComplete: false,
};

describe("transcript selectors", () => {
	test("derive current and streaming projections without mutating state", () => {
		const state: SessionState = {
			turns: [turn],
			currentTurnId: turn.id,
			thinkingDisplayMode: "expanded",
			thinkingLevel: "off",
		};
		const snapshot = structuredClone(state);

		expect(selectCurrentTurn(state)).toBe(turn);
		expect(selectStreamingContent(turn)).toBe("stream");
		expect(selectStreamingThinking(turn)).toEqual(["first", "second"]);
		expect(state).toEqual(snapshot);
	});

	test("derive completed assistant projections and counts", () => {
		expect(selectAssistantThinking(turn)).toBe("first\n\nsecond");
		expect(selectAssistantContent(turn)).toBe("done stream");
		expect(selectAssistantTools(turn).map(tool => tool.tool_name)).toEqual([
			"read_file",
		]);
		expect(selectMessageCount([turn])).toBe(2);
	});

	test("return empty projections when no assistant exists", () => {
		const empty: Turn = {
			id: "empty",
			userMessage: null,
			assistantMessage: null,
			isComplete: true,
		};
		expect(selectStreamingContent(empty)).toBeNull();
		expect(selectAssistantThinking(empty)).toBeNull();
		expect(selectAssistantContent(empty)).toBeNull();
		expect(selectAssistantTools(empty)).toEqual([]);
	});
});
