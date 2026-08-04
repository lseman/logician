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

void test("command search ranks compact and boundary-aware fuzzy matches", () => {
	const commands = [
		{ command: "/settings", description: "Configure Logician", category: "display" },
		{ command: "/sessions", description: "Browse saved conversations", category: "session" },
		{ command: "/status", description: "Show runtime details", category: "agent" },
	] as Parameters<typeof filterSlashCommands>[0];

	assert.equal(filterSlashCommands(commands, "/ssn")[0]?.command, "/sessions");
	assert.equal(filterSlashCommands(commands, "/saved")[0]?.command, "/sessions");
	assert.equal(filterSlashCommands(commands, "/display")[0]?.command, "/settings");
});

void test("memory command exposes the persistent memory handlers", () => {
	const commands = createSlashCommands(bridge, {});
	const memory = commands.find((command) => command.command === "/memory");
	assert.ok(memory);
	assert.equal(memory.dispatch, "local");
	assert.equal(memory.acceptsArgs, true);
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

void test("ask-user popup preview is discoverable and invokes its local handler", () => {
	let opened = false;
	const commands = createSlashCommands(bridge, {
		askPreview: () => {
			opened = true;
		},
	});
	const command = commands.find((item) => item.command === "/ask-preview");
	assert.equal(command?.dispatch, "local");
	assert.equal(command?.acceptsArgs, false);
	command?.handler?.("");
	assert.equal(opened, true);
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
