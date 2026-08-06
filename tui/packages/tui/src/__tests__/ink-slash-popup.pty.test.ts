import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { createPtyAppHome } from "../testing/pty-app-home.ts";
import { runInPty, screenFromPtyResult } from "../testing/pty-harness.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const entry = path.join(tuiRoot, "packages", "tui", "src", "index.ts");

// Regression test: the aboveInput selector must honor both stack-level
// visibility and SlashPopup.visible, which show()/hide() toggles.
void test("shows the slash command popup while typing a command", async () => {
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
			{ afterMs: 1200, send: "/mem" },
			{ afterMs: 800, send: "" },
		],
		timeoutMs: 6_000,
		columns: 100,
		rows: 30,
	});
	const screen = screenFromPtyResult(result, 100, 30).text();
	assert.match(
		screen,
		/commands \(\d+\)/,
		"slash popup header must be visible",
	);
	assert.match(screen, /\/memory/, "matching command must be listed");
	assert.match(
		result.output,
		/\x1b\[28;9H/,
		"hardware cursor must be parked on the input row",
	);
});
