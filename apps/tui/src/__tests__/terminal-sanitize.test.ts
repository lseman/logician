import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	sanitizeTerminalText,
	sanitizeTerminalValue,
} from "../rendering/terminal-sanitize.ts";

void test("terminal sanitizer removes CSI and terminal string controls", () => {
	const untrusted =
		"before\x1b[2Jafter" +
		"\x1b]0;owned title\x07visible" +
		"\x1bPmalicious device command\x1b\\done" +
		"\x1b_hidden payload\x1b\\end";

	assert.equal(sanitizeTerminalText(untrusted), "beforeaftervisibledoneend");
});

void test("terminal sanitizer normalizes carriage returns and removes C0/C1 controls", () => {
	assert.equal(
		sanitizeTerminalText("progress 1\rprogress 2\r\nok\x00\x7f\x9b2Jdone"),
		"progress 1\nprogress 2\nokdone",
	);
});

void test("terminal sanitizer recursively protects nested tool data without mutation", () => {
	const original = {
		result: "safe\x1b[Htext",
		details: {
			chunks: [{ output: "\x1b]8;;https://evil.test\x07link\x1b]8;;\x07" }],
		},
	};

	const sanitized = sanitizeTerminalValue(original);

	assert.deepEqual(sanitized, {
		result: "safetext",
		details: { chunks: [{ output: "link" }] },
	});
	assert.match(original.result, /\x1b/);
});
