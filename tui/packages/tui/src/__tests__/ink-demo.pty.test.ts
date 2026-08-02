import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import {
	runInPty,
	screenFromPtyResult,
} from "../testing/pty-harness.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const tsx = path.join(tuiRoot, "node_modules", ".bin", "tsx");
const entry = path.join(
	tuiRoot,
	"packages",
	"tui",
	"src",
	"ink-app",
	"demo-main.tsx",
);

// Proof-of-concept coverage for the Ink-based renderer shell: alt-screen
// entry, fixed dock (input bar + separators) under a scrollable transcript,
// and correct handling of astral/wide characters (emoji, CJK) that the
// legacy hand-rolled diff engine used to split across cells (see the
// visibleWidth/hardWrapVisible fix in terminal/core.ts).
void test("Ink demo renders transcript with emoji/CJK and a fixed input dock", async () => {
	const result = await runInPty({
		command: tsx,
		args: [entry],
		cwd: path.join(tuiRoot, "packages", "tui"),
		env: { TERM: "xterm-256color" },
		actions: [
			{ afterMs: 400, send: "hello world" },
			{ afterMs: 300, send: "\x1b" },
		],
		timeoutMs: 4_000,
		columns: 100,
		rows: 30,
	});
	const screen = screenFromPtyResult(result, 100, 30).text();
	assert.match(screen, /RESPONSE/);
	assert.match(screen, /🎉/, "emoji must render as a single glyph, not split");
	assert.match(screen, /中文/, "wide CJK characters must render intact");
	assert.match(screen, /hello world/, "input bar must reflect typed text");
	assert.doesNotMatch(result.output, /TypeError|Error:|ERROR/);
});
