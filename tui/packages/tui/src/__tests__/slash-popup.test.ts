import { test } from "node:test";
import assert from "node:assert/strict";
import type { SlashCommandDef } from "@logician/coding-agent/slash-commands";
import { SlashPopup } from "../components/slash-popup.ts";
import { initTheme, theme } from "../layers/theme/theme.ts";

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
