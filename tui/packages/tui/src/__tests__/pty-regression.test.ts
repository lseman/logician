import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { InputBar } from "../input/input-bar.ts";
import {
	type Component,
	normalizeKeyboardInput,
	TUI,
} from "../terminal/core.ts";
import {
	runInPty,
	screenFromPtyResult,
} from "../testing/pty-harness.ts";
import { createPtyAppHome } from "../testing/pty-app-home.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const entry = path.join(tuiRoot, "packages", "tui", "src", "index.ts");

// Run with `bun run` (not `tsx`/node directly): @logician/memory imports
// `bun:sqlite`, which plain Node's ESM loader cannot resolve.
void test("TUI starts in a real terminal and Ctrl+M changes mode", async () => {
	const result = await runInPty({
		command: "bun",
		args: ["run", entry],
		cwd: tuiRoot,
		env: {
			HOME: createPtyAppHome(),
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
		},
		actions: [
			{ afterMs: 1200, send: "s\n" },
			{ afterMs: 500, send: "\x1b[109;5u" },
		],
		timeoutMs: 6_000,
		columns: 120,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 120, 32).text();
	assert.match(screen, /Enter commands/);
	assert.match(screen, /REASON|Inference mode: Instruct \(Reasoning\)/);
	assert.doesNotMatch(result.output, /TypeError|TUI render error/);
});

void test("Kitty Ctrl+O and Ctrl+C reports reach TUI keybindings", () => {
	assert.equal(normalizeKeyboardInput("\x1b[27u"), "\x1b");
	assert.equal(normalizeKeyboardInput("\x1b[27;1u"), "\x1b");
	assert.equal(normalizeKeyboardInput("\x1b[111;5u"), "\x0f");
	assert.equal(normalizeKeyboardInput("\x1b[99;5u"), "\x03");
	assert.equal(normalizeKeyboardInput("\x1b[116;6u"), "\x14");
	assert.equal(
		normalizeKeyboardInput("\x1b[105;5u"),
		"\x1b[105;5u",
		"Ctrl+I must remain distinct from Tab",
	);
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

void test("Ctrl+I changes and persists the execution profile", async () => {
	const home = createPtyAppHome();
	const result = await runInPty({
		command: "bun",
		args: ["run", entry],
		cwd: tuiRoot,
		env: {
			HOME: home,
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
		},
		actions: [
			{ afterMs: 1200, send: "s\n" },
			{ afterMs: 500, send: "\x1b[105;5u" },
		],
		timeoutMs: 6_000,
		columns: 140,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 140, 32).text();
	const settings = JSON.parse(
		readFileSync(path.join(home, ".logician", "settings.json"), "utf8"),
	) as { executionProfile?: string };

	assert.match(screen, /Execution policy: minimal|exec: minimal/);
	assert.equal(settings.executionProfile, "minimal");
	assert.doesNotMatch(result.output, /TUI render error/);
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

void test("Escape reaches the dialog owner before the core overlay fallback", () => {
	const tui = new TUI({} as NodeJS.WriteStream);
	const dialog: Component & {
		visible: boolean;
		handleInput(data: string): { type: "close" } | null;
	} = {
		visible: true,
		render: () => [],
		handleInput: (data) => (data === "\x1b" ? { type: "close" } : null),
	};
	let dismissed = false;
	tui.showOverlay(dialog);
	tui.addInputListener((data) => {
		const action = dialog.handleInput(data);
		if (action?.type === "close") {
			dismissed = true;
			tui.removeOverlay(dialog);
			return { consume: true };
		}
		return undefined;
	});

	(
		tui as unknown as {
			handleInput(data: string): void;
		}
	).handleInput(normalizeKeyboardInput("\x1b[27u"));

	assert.equal(dismissed, true);
	assert.equal(dialog.visible, false);
});

void test("TUI expands tools from a Kitty Ctrl+O sequence", async () => {
	const result = await runInPty({
		command: "bun",
		args: ["run", entry],
		cwd: tuiRoot,
		env: {
			HOME: createPtyAppHome(),
			TERM: "xterm-256color",
			LOGICIAN_TRUST: "always",
			LOGICIAN_MCP: "0",
			LOGICIAN_HOOKS: "0",
		},
		actions: [
			{ afterMs: 1200, send: "s\n" },
			{ afterMs: 500, send: "\x1b[111;5u" },
		],
		timeoutMs: 6_000,
		columns: 120,
		rows: 32,
	});
	const screen = screenFromPtyResult(result, 120, 32).text();
	assert.match(screen, /TOOLS EXPANDED/);
	assert.doesNotMatch(result.output, /TUI render error/);
});
