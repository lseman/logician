import { test } from "bun:test";
import assert from "node:assert/strict";
import { clampLineToWidth, visibleWidth } from "../terminal/core.ts";
import { renderTerminalScreen } from "../testing/terminal-screen.ts";

void test("terminal screen applies absolute cursor updates and erases", () => {
	const screen = renderTerminalScreen(
		"\x1b[2J\x1b[1;1Hhello world\x1b[1;7H\x1b[0mX\x1b[2;1Hfooter" +
			"\x1b[2;4H\x1b[K",
		12,
		3,
	);

	assert.equal(screen.line(0), "hello Xorld");
	assert.equal(screen.line(1), "foo");
});

void test("terminal screen preserves unchanged cells across differential frames", () => {
	const screen = renderTerminalScreen(
		"\x1b[2J\x1b[1;1Halpha bravo\x1b[1;7HX",
		20,
		2,
	);

	assert.equal(screen.line(0), "alpha Xravo");
});

void test("terminal screen tracks wide cells and cursor visibility", () => {
	const screen = renderTerminalScreen(
		"\x1b[1;1H界abc\x1b[1;4HX\x1b[?25l",
		10,
		2,
	);

	assert.equal(screen.line(0), "界aXc");
	assert.equal(screen.cursor().visible, false);
});

void test("width helpers keep emoji grapheme clusters intact", () => {
	assert.equal(visibleWidth("✅"), 2);
	assert.equal(visibleWidth("👨‍👩‍👧‍👦"), 2);
	assert.equal(clampLineToWidth("ab✅cd", 4), "ab✅");
	assert.equal(clampLineToWidth("ab👨‍👩‍👧‍👦cd", 3), "ab");
});
