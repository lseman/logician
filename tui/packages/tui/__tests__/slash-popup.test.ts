import assert from "node:assert/strict";
import { test } from "node:test";
import { SlashPopup } from "../src/overlays/slash-popup.ts";
import { initTheme } from "../src/terminal/theme.ts";
import type { SlashCommandDef } from "@logician/coding-agent/commands";

initTheme("dark");

void test("slash popup renders every command for a bare slash", () => {
	const popup = new SlashPopup();
	const commands: SlashCommandDef[] = Array.from({ length: 15 }, (_, index) => ({
		command: `/command-${index}`,
		description: `Command ${index}`,
		dispatch: "local",
		acceptsArgs: false,
		category: index < 8 ? "session" : "context",
	}));
	popup.setCommands(commands);
	popup.setQuery("/");
	popup.show();
	let rendered = popup.render(120).join("\n");
	assert.match(rendered, /commands.*\(15\)/);
	assert.match(rendered, /more below/);
	for (let index = 0; index < commands.length - 1; index++) {
		popup.moveSelection(1);
	}
	rendered = popup.render(120).join("\n");
	assert.match(rendered, /more above/);
	assert.match(rendered, /\/command-14/);
});

void test("slash popup does not replay a stale previous result", () => {
	const popup = new SlashPopup();
	const submissions: Array<string | null> = [];
	popup.setCommands([
		{
			command: "/first",
			description: "first",
			dispatch: "local",
			acceptsArgs: false,
			handler: () => "first result",
		},
		{
			command: "/second",
			description: "second",
			dispatch: "local",
			acceptsArgs: false,
			handler: () => undefined,
		},
	]);
	popup.setOnSubmit((result) => submissions.push(result));
	popup.setQuery("/first");
	popup.show();
	popup.handleInput("\n");
	popup.setQuery("/second");
	popup.show();
	popup.handleInput("\n");
	assert.deepEqual(submissions, ["first result", null]);
});
