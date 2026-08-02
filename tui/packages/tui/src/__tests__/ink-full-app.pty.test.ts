import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import {
	runInPty,
	screenFromPtyResult,
} from "../testing/pty-harness.ts";
import { createPtyAppHome } from "../testing/pty-app-home.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const entry = path.join(tuiRoot, "packages", "tui", "src", "index.ts");

// Boots the real LogicianTUI app (bridge, transcript, input bar, status bar,
// overlays) end to end. Run with `bun run` (not `tsx`/node directly):
// @logician/memory imports `bun:sqlite`, which plain Node's ESM loader
// cannot resolve -- a pre-existing constraint on how the CLI is invoked.
void test("LogicianTUI boots, accepts input, and starts a real turn", async () => {
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
			{ afterMs: 1500, send: "ping\n" },
			{ afterMs: 800, send: "\x03" },
		],
		timeoutMs: 8_000,
		columns: 100,
		rows: 30,
	});
	const screen = screenFromPtyResult(result, 100, 30).text();
	assert.match(screen, /YOU/, "submitted message must appear in the transcript");
	assert.match(screen, /ping/, "input bar text must reach the transcript");
	assert.doesNotMatch(
		result.output,
		/ERR_UNSUPPORTED_ESM_URL_SCHEME|TypeError|\[TUI render error\]/,
	);
});
