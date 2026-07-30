import assert from "node:assert/strict";
import test from "node:test";
import { diffTerminalLine } from "../terminal/core.ts";

void test("cell diff writes only the changed run on the second frame", () => {
	const update = diffTerminalLine("alpha bravo", "alpha Xravo", 0, 19);

	assert.ok(update.includes("\x1b[1;7H"));
	assert.ok(update.includes("\x1b[0mX"));
	assert.doesNotMatch(update, /alpha/);
	assert.doesNotMatch(update, /\r|\n/);
});

void test("cell diff addresses columns after a wide character correctly", () => {
	const update = diffTerminalLine("界abc", "界aXc", 0, 19);

	assert.ok(update.includes("\x1b[1;4H"));
	assert.ok(update.includes("\x1b[0mX"));
});
