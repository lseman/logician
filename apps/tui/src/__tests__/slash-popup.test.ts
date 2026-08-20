import { test } from "bun:test";
import assert from "node:assert/strict";
import type { SlashCommandDef } from "@logician/agent-core/commands";
import { SlashPopup } from "../overlays/slash-popup.ts";
import { initTheme, theme } from "../terminal/theme.ts";

initTheme("dark");

const commands: SlashCommandDef[] = [
	{
		command: "/help",
		description: "Show help",
		dispatch: "bridge",
		acceptsArgs: false,
		category: "help",
	},
	{
		command: "/settings",
		description: "Open settings",
		dispatch: "bridge",
		acceptsArgs: false,
		category: "help",
	},
];

test("SlashPopup renders the active command using the theme selected color", () => {
	const popup = new SlashPopup();
	popup.setCommands(commands);
	popup.setQuery("/");
	popup.show();

	const selectedColor = theme.fgRaw("selected");
	let output = popup.render(80).join("\n");
	assert.ok(output.includes(`${selectedColor}▸ \x1b[1m/help`));

	popup.moveSelection(1);
	output = popup.render(80).join("\n");
	assert.ok(output.includes(`${selectedColor}▸ \x1b[1m/settings`));
});

test("SlashPopup completes declared subcommands", () => {
	const popup = new SlashPopup();
	popup.setCommands([
		{
			command: "/mcp",
			description: "Manage MCP servers",
			dispatch: "local",
			acceptsArgs: true,
			subcommands: ["list", "add", "remove"],
		},
	]);

	popup.setQuery("/mcp li");
	popup.show();
	assert.equal(popup.hasMatches(), true);
	assert.equal(popup.currentCommand(), "/mcp list");
	assert.match(popup.render(80).join("\n"), /\/mcp list/);
});

test("SlashPopup returns to the best-ranked result when the query changes", () => {
	const popup = new SlashPopup();
	popup.setCommands(commands);
	popup.setQuery("/");
	popup.moveSelection(1);
	assert.equal(popup.currentCommand(), "/settings");

	popup.setQuery("/he");
	assert.equal(popup.currentCommand(), "/help");
});

test("SlashPopup emphasizes command characters responsible for fuzzy matches", () => {
	const popup = new SlashPopup();
	popup.setCommands([
		...commands,
		{
			command: "/sessions",
			description: "Browse conversations",
			dispatch: "local",
			acceptsArgs: false,
			category: "session",
		},
	]);
	popup.setQuery("/s");
	popup.show();
	const rendered = popup.render(80).join("\n");
	assert.equal(popup.hasMatches(), true);
	assert.match(rendered, new RegExp(theme.fgRaw("accent").replace("[", "\\[")));
});

test("submitRaw establishes the command turn before running the local handler", () => {
	const order: string[] = [];
	const popup = new SlashPopup();
	popup.setCommands([
		{
			command: "/spawn",
			description: "Spawn",
			dispatch: "local",
			acceptsArgs: true,
			handler: () => {
				order.push("handler");
				return undefined;
			},
		},
	]);
	popup.setOnSubmit((result, _dispatch, command) => {
		if (command?.trim()) order.push(`turn:${command}`);
		if (result) order.push(`result:${result}`);
	});
	assert.equal(popup.submitRaw("/spawn list files"), true);
	assert.deepEqual(order, ["turn:/spawn list files", "handler"]);
});

test("submitRaw delivers handler return text after the turn is established", () => {
	const order: string[] = [];
	const popup = new SlashPopup();
	popup.setCommands([
		{
			command: "/thinking",
			description: "Thinking",
			dispatch: "local",
			acceptsArgs: true,
			handler: () => {
				order.push("handler");
				return "Thinking level: high";
			},
		},
	]);
	popup.setOnSubmit((result, _dispatch, command) => {
		if (command?.trim()) order.push(`turn:${command}`);
		if (result) order.push(`result:${result}`);
	});
	assert.equal(popup.submitRaw("/thinking high"), true);
	assert.deepEqual(order, [
		"turn:/thinking high",
		"handler",
		"result:Thinking level: high",
	]);
});
