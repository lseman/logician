import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import {
	runInPty,
	screenFromPtyResult,
} from "../testing/pty-harness.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const entry = path.join(tuiRoot, "packages", "tui", "src", "index.ts");

// Boots the real LogicianTUI app (bridge, transcript, input bar, status bar,
// overlays) under the Ink-backed renderer (LOGICIAN_INK_RENDERER=1), which
// reuses TUI's own input routing / scroll / overlay logic via the
// onFrame/externalIO hooks and only replaces the paint layer. Run with `bun
// run` (not `tsx`/node directly): @logician/memory imports `bun:sqlite`,
// which plain Node's ESM loader cannot resolve regardless of renderer --
// this is a pre-existing constraint on how the CLI is invoked, not specific
// to the Ink renderer.
void test("LogicianTUI boots under the Ink renderer, accepts input, and starts a real turn", async () => {
	const result = await runInPty({
		command: "bun",
		args: ["run", entry],
		cwd: tuiRoot,
		env: {
			TERM: "xterm-256color",
			LOGICIAN_INK_RENDERER: "1",
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
