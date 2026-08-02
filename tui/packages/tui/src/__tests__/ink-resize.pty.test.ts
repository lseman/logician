import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import {
	runInPty,
	screenFromPtyResult,
} from "../testing/pty-harness.ts";

const tuiRoot = path.resolve(import.meta.dirname, "../../../..");
const entry = path.join(tuiRoot, "packages", "tui", "src", "index.ts");

// A mid-session resize is exactly the scenario that used to leave stale,
// wrong-width cells on screen under the legacy renderer (previousLines was
// never invalidated on resize until a fix earlier in this effort). Ink owns
// layout/diffing/resize itself, so this exercises that Ink's resize handling
// produces a frame sized to the NEW terminal dimensions, not a stale one.
void test("Ink renderer repaints correctly after a terminal resize", async () => {
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
			{ afterMs: 1200, resize: { columns: 60, rows: 20 } },
			{ afterMs: 500, send: "resized ok" },
			{ afterMs: 500, send: "\x03" },
		],
		timeoutMs: 8_000,
		columns: 100,
		rows: 30,
	});
	// After the resize the PTY itself is 60x20; render against that.
	const screen = screenFromPtyResult(result, 60, 20).text();
	assert.match(screen, /resized ok/, "input bar must reflect input typed post-resize");
	assert.doesNotMatch(
		result.output,
		/ERR_UNSUPPORTED_ESM_URL_SCHEME|TypeError|\[TUI render error\]/,
	);
	// A crude but effective floating-content check: no row in the final
	// screen should be wider than the new terminal width once ANSI is
	// stripped (would indicate a stale wider frame bleeding through).
	const lines = screenFromPtyResult(result, 60, 20).lines();
	for (const line of lines) {
		assert.ok(
			line.length <= 60,
			`row exceeds resized terminal width: ${JSON.stringify(line)}`,
		);
	}
});
