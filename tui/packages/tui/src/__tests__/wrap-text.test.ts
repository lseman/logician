import assert from "node:assert/strict";
import test from "node:test";
import { wrapText } from "../rendering/transcript/layout.ts";
import { visibleWidth } from "../terminal/core.ts";

const RESET = "\x1b[0m";
const COLOR = "\x1b[38;5;8m";
const ACCENT = "\x1b[38;5;14m";
const BOLD = "\x1b[1m";

void test("wrapText preserves color across every wrapped line, not just the first", () => {
	const text = `${COLOR}This is a long notice body that should definitely wrap across multiple lines when the terminal width is narrow enough to force it${RESET}`;
	const lines = wrapText(text, 40);
	assert.ok(lines.length > 1, "text should actually wrap for this test to be meaningful");
	for (const line of lines) {
		assert.match(line, /^\x1b\[38;5;8m/, `every wrapped line should reopen the color: ${JSON.stringify(line)}`);
	}
});

void test("wrapText never splits a word mid-character when it fits on its own line", () => {
	const original = "This is a long notice body that should definitely wrap across multiple lines when narrow";
	const lines = wrapText(original, 40);
	assert.ok(lines.length > 1, "text should actually wrap for this test to be meaningful");
	// Rejoining every wrapped line's words must reproduce the exact original
	// word sequence — if any word got split (e.g. "definitely" -> "d" +
	// "efinitely"), the rejoined word list would differ from the original.
	const rejoinedWords = lines.join(" ").split(/\s+/);
	const originalWords = original.split(/\s+/);
	assert.deepEqual(rejoinedWords, originalWords);
});

void test("wrapText does not duplicate color codes when hard-wrapping a single long word", () => {
	const lines = wrapText(`${COLOR}${"x".repeat(60)}${RESET}`, 20);
	for (const line of lines) {
		const opens = line.match(/\x1b\[38;5;8m/g) ?? [];
		assert.equal(opens.length, 1, `line should open the color exactly once: ${JSON.stringify(line)}`);
	}
});

void test("wrapText carries color from a hard-wrapped word into the words that follow it", () => {
	const lines = wrapText(
		`${COLOR}${"z".repeat(50)} and then some more words follow after the long one${RESET}`,
		20,
	);
	for (const line of lines) {
		assert.match(line, /^\x1b\[38;5;8m/, `line should still be colored: ${JSON.stringify(line)}`);
	}
});

void test("wrapText preserves a mid-line color change (e.g. accent inside muted text)", () => {
	const lines = wrapText(
		`${COLOR}prefix text here ${ACCENT}[accent part]${RESET} ${COLOR}more muted text continues on and on to force a wrap${RESET}`,
		40,
	);
	const joined = lines.join("");
	assert.ok(joined.includes(ACCENT), "accent color should survive wrapping");
});

void test("wrapText carries color across explicit newlines", () => {
	const lines = wrapText(
		`${COLOR}first line is short${RESET}\n${COLOR}second line is quite a bit longer and needs to wrap across multiple output lines${RESET}`,
		30,
	);
	for (const line of lines) {
		assert.match(line, /^\x1b\[38;5;8m/);
	}
});

void test("wrapText combines multiple active codes (bold + color) without duplication", () => {
	const lines = wrapText(
		`${BOLD}${COLOR}Bold and colored text that is long enough to wrap onto a second visual line for sure${RESET}`,
		35,
	);
	for (const line of lines) {
		const boldOpens = line.match(/\x1b\[1m/g) ?? [];
		const colorOpens = line.match(/\x1b\[38;5;8m/g) ?? [];
		assert.equal(boldOpens.length, 1, `bold should appear once: ${JSON.stringify(line)}`);
		assert.equal(colorOpens.length, 1, `color should appear once: ${JSON.stringify(line)}`);
	}
});

void test("wrapText never produces a line wider than the requested width", () => {
	const cases: Array<[string, number]> = [
		[`${COLOR}This is a long notice body that should definitely wrap across multiple lines when the terminal width is narrow enough to force it${RESET}`, 40],
		["This is a long notice body that should definitely wrap across multiple lines when narrow", 40],
		[`${COLOR}${"x".repeat(60)}${RESET}`, 20],
		[`${BOLD}${COLOR}Bold and colored text that is long enough to wrap onto a second visual line for sure${RESET}`, 35],
	];
	for (const [text, width] of cases) {
		for (const line of wrapText(text, width)) {
			assert.ok(
				visibleWidth(line) <= width,
				`line exceeds width ${width}: ${JSON.stringify(line)} (${visibleWidth(line)})`,
			);
		}
	}
});

void test("wrapText leaves plain uncolored text unchanged in behavior", () => {
	const lines = wrapText("aaaaaaaaaa bbbbbbbbbb cccccccccc dddddddddd", 15);
	assert.deepEqual(lines, ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc", "dddddddddd"]);
});

void test("wrapText returns short text unchanged", () => {
	assert.deepEqual(wrapText("short", 40), ["short"]);
	assert.deepEqual(wrapText(`${COLOR}short${RESET}`, 40), [`${COLOR}short${RESET}`]);
});
