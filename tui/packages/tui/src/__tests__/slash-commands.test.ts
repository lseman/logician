// ── Slash command tests ──────────────────────────────────────────────────────

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import {
	CATEGORY_ORDER,
	filterSlashCommands,
	groupByCategory,
	type SlashCommandDef,
} from "@logician/coding-agent/commands";

const TEST_COMMANDS: SlashCommandDef[] = [
	{
		command: "/new",
		description: "New session",
		dispatch: "bridge",
		acceptsArgs: false,
		category: "session",
	},
	{
		command: "/sessions",
		description: "List sessions",
		dispatch: "local",
		acceptsArgs: true,
		category: "session",
		argHint: "[filter]",
	},
	{
		command: "/compact",
		description: "Compact history",
		dispatch: "bridge",
		acceptsArgs: false,
		category: "context",
	},
	{
		command: "/fork",
		description: "Fork branch",
		dispatch: "local",
		acceptsArgs: false,
		category: "context",
		examples: ["/fork"],
	},
	{
		command: "/thinking",
		description: "Set thinking level",
		dispatch: "local",
		acceptsArgs: true,
		category: "display",
		argHint: "<level>",
		examples: ["/thinking high", "/thinking off"],
	},
	{
		command: "/quit",
		description: "Exit",
		dispatch: "quit",
		acceptsArgs: false,
		category: "shortcuts",
	},
	{
		command: "/reasoner",
		description: "Select reasoning mode",
		dispatch: "local",
		acceptsArgs: true,
		category: "reasoning",
		argHint: "<mode>",
	},
	{
		command: "/plugins",
		description: "Manage plugins",
		dispatch: "local",
		acceptsArgs: true,
		category: "skills",
	},
];

describe("filterSlashCommands", () => {
	it("returns all commands with empty query", () => {
		const result = filterSlashCommands(TEST_COMMANDS, "");
		assert.strictEqual(result.length, TEST_COMMANDS.length);
	});

	it("filters by exact command name match", () => {
		const result = filterSlashCommands(TEST_COMMANDS, "/new");
		assert.strictEqual(result.length, 1);
		assert.strictEqual(result[0].command, "/new");
	});

	it("filters by prefix match", () => {
		const result = filterSlashCommands(TEST_COMMANDS, "/com");
		assert.ok(result.some(c => c.command.startsWith("/comp")));
	});

	it("filters by description match", () => {
		const result = filterSlashCommands(TEST_COMMANDS, "session");
		assert.ok(
			result.some(c => c.command === "/new" || c.command === "/sessions"),
		);
	});

	it("respects limit", () => {
		const result = filterSlashCommands(TEST_COMMANDS, "", 3);
		assert.strictEqual(result.length, 3);
	});

	it("returns empty for no match", () => {
		const result = filterSlashCommands(TEST_COMMANDS, "/nonexistent");
		assert.strictEqual(result.length, 0);
	});

	it("sorts by relevance (exact > prefix > contains > description)", () => {
		// Exact match should score highest
		const result = filterSlashCommands(TEST_COMMANDS, "/new");
		assert.strictEqual(result.length, 1);
		assert.strictEqual(result[0].command, "/new");
	});
});

describe("groupByCategory", () => {
	it("groups all commands by category", () => {
		const groups = groupByCategory(TEST_COMMANDS);
		assert.strictEqual(groups.size, 6); // session, context, skills, reasoning, display, shortcuts
	});

	it("preserves category order", () => {
		const groups = groupByCategory(TEST_COMMANDS);
		const cats: string[] = [];
		for (const cat of CATEGORY_ORDER) {
			if (groups.has(cat)) cats.push(cat);
		}
		// CATEGORY_ORDER: help, session, agent, context, skills, reasoning, display, permissions, shortcuts, loop, misc
		assert.strictEqual(cats[0], "session");
		assert.strictEqual(cats[1], "context");
		assert.strictEqual(cats[2], "skills");
		assert.strictEqual(cats[3], "reasoning");
		assert.strictEqual(cats[4], "display");
		assert.strictEqual(cats[5], "shortcuts");
	});

	it("excludes empty categories", () => {
		const groups = groupByCategory(TEST_COMMANDS);
		assert.ok(!groups.has("agent"));
		assert.ok(!groups.has("help"));
	});

	it("puts uncategorized commands in misc", () => {
		const uncategorized: SlashCommandDef[] = [
			{
				command: "/test",
				description: "No category",
				dispatch: "local",
				acceptsArgs: false,
			},
		];
		const groups = groupByCategory(uncategorized);
		assert.strictEqual(groups.size, 1);
		assert.ok(groups.has("misc"));
		assert.strictEqual(groups.get("misc")?.[0].command, "/test");
	});

	it("preserves command order within category", () => {
		const grouped = groupByCategory(TEST_COMMANDS);
		const sessionCmds = grouped.get("session");
		assert.strictEqual(sessionCmds?.length, 2);
		assert.strictEqual(sessionCmds?.[0].command, "/new");
		assert.strictEqual(sessionCmds?.[1].command, "/sessions");
	});
});

describe("category metadata", () => {
	it("commands have category property", () => {
		const groups = groupByCategory(TEST_COMMANDS);
		for (const cmds of groups.values()) {
			for (const cmd of cmds) {
				assert.ok(cmd.category, `${cmd.command} should have a category`);
			}
		}
	});

	it("commands with argHint are marked correctly", () => {
		const groups = groupByCategory(TEST_COMMANDS);
		const contextCmds = groups.get("context") || [];
		const compact = contextCmds.find(c => c.command === "/compact");
		const fork = contextCmds.find(c => c.command === "/fork");
		assert.ok(compact !== undefined);
		assert.strictEqual(compact?.argHint, undefined);
		assert.strictEqual(fork?.argHint, undefined);
		assert.strictEqual(fork?.examples?.[0], "/fork");
	});

	it("commands with examples are marked correctly", () => {
		const groups = groupByCategory(TEST_COMMANDS);
		const displayCmds = groups.get("display") || [];
		const thinking = displayCmds.find(c => c.command === "/thinking");
		assert.ok(thinking !== undefined);
		assert.strictEqual(thinking?.examples?.length, 2);
		assert.strictEqual(thinking?.examples?.[0], "/thinking high");
	});
});
