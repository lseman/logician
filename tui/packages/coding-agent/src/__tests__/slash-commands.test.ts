import assert from "node:assert/strict";
import { test } from "node:test";
import {
	createSlashCommands,
	filterSlashCommands,
	formatSlashHelp,
} from "../commands/slash-commands.ts";

const bridge = { sendSlash: () => {}, cancel: () => {}, reset: () => {} };

void test("bare slash returns the complete command catalog", () => {
	const commands = createSlashCommands(bridge, {});
	assert.ok(commands.length > 10);
	assert.equal(filterSlashCommands(commands, "/").length, commands.length);
});

void test("memory command is discoverable with management actions", () => {
	const commands = createSlashCommands(bridge, {});
	const memory = commands.find((command) => command.command === "/memory");
	assert.ok(memory);
	assert.equal(memory.acceptsArgs, true);
	assert.match(memory.argHint ?? "", /status.*list.*search.*show.*add.*drop.*clear/);
});

void test("steer-now forces the existing steering queue without accepting text", () => {
	const commands = createSlashCommands(bridge, {});
	const command = commands.find((item) => item.command === "/steer-now");
	assert.ok(command);
	assert.equal(command.dispatch, "bridge");
	assert.equal(command.acceptsArgs, false);
});

void test("queue management commands are discoverable", () => {
	const commands = createSlashCommands(bridge, {});
	assert.ok(commands.some((command) => command.command === "/queue"));
	assert.ok(commands.some((command) => command.command === "/queue-clear"));
	const drop = commands.find((command) => command.command === "/queue-drop");
	assert.equal(drop?.argHint, "<number>");
});

void test("file-backed EoH is discoverable", () => {
	const commands = createSlashCommands(bridge, {
		eoh: (args: unknown) => `eoh:${String(args)}`,
	});
	const command = commands.find((item) => item.command === "/eoh");
	assert.ok(command);
	assert.equal(command.handler?.("heuristic.py"), "eoh:heuristic.py");
	assert.match(command.argHint ?? "", /heuristic\.py/);
});

void test("help renders the live registry and supports topics", () => {
	const commands = createSlashCommands(bridge, {});
	const help = commands.find((command) => command.command === "/help");
	const alias = commands.find((command) => command.command === "/?");
	const full = help?.handler?.("") ?? "";
	assert.match(full, new RegExp(`Available commands \\(${commands.length}\\)`));
	assert.match(full, /\/context/);
	assert.match(full, /\/settings/);
	assert.match(alias?.handler?.("") ?? "", /Available commands/);

	const sessionHelp = formatSlashHelp(commands, "session");
	assert.match(sessionHelp, /SESSION/);
	assert.match(sessionHelp, /\/sessions/);
	assert.doesNotMatch(sessionHelp, /\/rag\s/);
});
