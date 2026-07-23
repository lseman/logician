import assert from "node:assert/strict";
import { cpSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import {
	runInPty,
	stripTerminalControls,
} from "../testing/pty-harness.ts";
import { normalizeKeyboardInput } from "../layers/core/tui-core.ts";
import { InputBar } from "../components/input-bar.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const tsx = path.join(tuiRoot, "node_modules", ".bin", "tsx");
const entry = path.join(tuiRoot, "packages", "tui", "src", "index.ts");

void test("TUI starts in a real terminal and Ctrl+M changes mode", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	cpSync(path.join(tuiRoot, "themes"), themeDir, { recursive: true });
	const result = await runInPty({
		command: tsx,
		args: [entry],
		cwd: tuiRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
		},
		actions: [{ afterMs: 400, send: "\x1b[109;5u" }],
		timeoutMs: 4_000,
		columns: 120,
		rows: 32,
	});
	const output = stripTerminalControls(result.output);
	assert.match(output, /MESSAGE/);
	assert.match(output, /mode: REASON|Inference mode: Instruct \(Reasoning\)/);
	assert.doesNotMatch(output, /TypeError|TUI render error/);
});

void test("Kitty Ctrl+O and Ctrl+C reports reach legacy TUI keybindings", () => {
	assert.equal(normalizeKeyboardInput("\x1b[111;5u"), "\x0f");
	assert.equal(normalizeKeyboardInput("\x1b[99;5u"), "\x03");
	assert.equal(normalizeKeyboardInput("\x1b[116;6u"), "\x14");
	assert.equal(
		normalizeKeyboardInput("\x1b[109;5u"),
		"\x1b[109;5u",
		"Ctrl+M must remain distinct from Enter",
	);

	const input = new InputBar();
	let cancelled = 0;
	input.onCancel = () => {
		cancelled++;
	};
	input.valueText = "draft message";
	input.handleInput(normalizeKeyboardInput("\x1b[99;5u"));
	assert.equal(input.valueText, "");
	assert.equal(cancelled, 1);
});

void test("Escape clears first and cancels the active turn on second press", () => {
	const input = new InputBar();
	let cancelled = 0;
	input.onCancel = () => {
		cancelled++;
	};
	input.valueText = "draft message";

	input.handleInput("\x1b");
	assert.equal(input.valueText, "");
	assert.equal(cancelled, 0);

	input.handleInput("\x1b");
	assert.equal(cancelled, 1);

	input.handleInput("x");
	input.handleInput("\x1b");
	input.handleInput("y");
	input.handleInput("\x1b");
	assert.equal(
		cancelled,
		1,
		"typing between Escape presses must disarm turn cancellation",
	);
});

void test("TUI expands tools from a Kitty Ctrl+O sequence", async () => {
	const home = mkdtempSync(path.join(tmpdir(), "logician-pty-home-"));
	const themeDir = path.join(home, ".logician", "themes");
	mkdirSync(themeDir, { recursive: true });
	cpSync(path.join(tuiRoot, "themes"), themeDir, { recursive: true });
	const result = await runInPty({
		command: tsx,
		args: [entry],
		cwd: tuiRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
		},
		actions: [{ afterMs: 400, send: "\x1b[111;5u" }],
		timeoutMs: 4_000,
		columns: 120,
		rows: 32,
	});
	const output = stripTerminalControls(result.output);
	assert.match(output, /TOOLS EXPANDED/);
	assert.doesNotMatch(output, /TUI render error/);
});
