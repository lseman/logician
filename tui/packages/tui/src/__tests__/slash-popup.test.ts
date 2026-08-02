import { test } from "node:test";
import assert from "node:assert/strict";
import type { SlashCommandDef } from "@logician/coding-agent/commands";
import { SlashPopup } from "../overlays/slash-popup.ts";
import { initTheme } from "../terminal/theme.ts";

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

test("SlashPopup exposes the active command to Ink", () => {
		const popup = new SlashPopup();
		popup.setCommands(commands);
		popup.setQuery("/");
		popup.show();

		let model = popup.getInkOverlayModel();
		assert.equal(model.items.find((item) => item.selected)?.label, "/help");

		popup.moveSelection(1);
		model = popup.getInkOverlayModel();
		assert.equal(model.items.find((item) => item.selected)?.label, "/settings");
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
