import { describe, expect, test } from "bun:test";
import type { GroveSession, GroveState } from "./model.ts";
import {
	conversationParents,
	filterSessions,
	renderForest,
	renderTree,
} from "./render.ts";

const session: GroveSession = {
	id: "session-1",
	name: "Fix authentication",
	preview: "Investigate the login failure",
	cwd: "/work/project",
	lastActivity: Date.now(),
	messageCount: 3,
	branchCount: 1,
	entries: [
		{
			type: "message",
			id: "root",
			timestamp: 1,
			message: { role: "user", content: "Investigate login", timestamp: 1 },
		},
		{
			type: "message",
			id: "answer-a",
			parentId: "root",
			timestamp: 2,
			message: { role: "assistant", content: "First answer", timestamp: 2 },
		},
		{
			type: "message",
			id: "answer-b",
			parentId: "root",
			timestamp: 3,
			message: { role: "assistant", content: "Alternate answer", timestamp: 3 },
		},
	],
};

const state: GroveState = {
	screen: { kind: "forest" },
	selection: 0,
	scroll: 0,
	query: "",
};

describe("grove rendering", () => {
	test("filters by title and preview", () => {
		expect(filterSessions([session], "AUTH")).toHaveLength(1);
		expect(filterSessions([session], "login")).toHaveLength(1);
		expect(filterSessions([session], "docs")).toHaveLength(0);
	});

	test("renders forest metadata and controls", () => {
		const output = renderForest([session], state, 100, 15).join("\n");
		expect(output).toContain("Fix authentication");
		expect(output).toContain("1 forks");
		expect(output).toContain("Enter/a open Logician");
	});

	test("renders sibling entries as branches", () => {
		const output = renderTree(session, 100, 20).join("\n");
		expect(output).toContain("├─");
		expect(output).toContain("└─");
		expect(output).toContain("Alternate answer");
	});

	test("hides tool configuration events without disconnecting messages", () => {
		const entries: GroveSession["entries"] = [
			{
				type: "message",
				id: "root",
				timestamp: 1,
				message: { role: "user", content: "Hello", timestamp: 1 },
			},
			{
				type: "active_tools_change",
				id: "tools",
				parentId: "root",
				timestamp: 2,
				activeToolNames: ["read_file"],
			},
			{
				type: "message",
				id: "reply",
				parentId: "tools",
				timestamp: 3,
				message: { role: "assistant", content: "Hi", timestamp: 3 },
			},
		];
		const children = conversationParents(entries);
		expect(children.get("root")?.map(entry => entry.id)).toEqual(["reply"]);

		const output = renderTree({ ...session, entries }, 100, 20).join("\n");
		expect(output).not.toContain("active tools change");
		expect(output).toContain("Hi");
		expect(output).toContain("2 conversation entries");
	});
});
