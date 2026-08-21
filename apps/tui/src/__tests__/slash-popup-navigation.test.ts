import { test } from "bun:test";
import assert from "node:assert/strict";
import type { SlashCommandDef } from "@logician/agent-runtime/commands";
import { SlashPopup } from "../overlays/slash-popup.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

void test("slash popup renders every command for a bare slash", () => {
	const popup = new SlashPopup();
	const commands: SlashCommandDef[] = Array.from(
		{ length: 15 },
		(_, index) => ({
			command: `/command-${index}`,
			description: `Command ${index}`,
			dispatch: "local",
			acceptsArgs: false,
			category: index < 8 ? "session" : "context",
		}),
	);
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
	const submissions: Array<[string | null, string | undefined]> = [];
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
	// Each submit fires a turn-establish call (command set, result null) before
	// any handler result call — see submitRaw's comment on turn ordering.
	popup.setOnSubmit((result, _dispatch, command) =>
		submissions.push([result, command]),
	);
	popup.setQuery("/first");
	popup.show();
	popup.handleInput("\n");
	popup.setQuery("/second");
	popup.show();
	popup.handleInput("\n");
	assert.deepEqual(submissions, [
		[null, "/first"],
		["first result", undefined],
		[null, "/second"],
	]);
});
