import { test } from "bun:test";
import assert from "node:assert/strict";
import { isCtrlE } from "../app/input-controller.ts";

void test("Ctrl+E recognizes bare CAN and CSI-u encodings", () => {
	assert.equal(isCtrlE("\x05"), true);
	assert.equal(isCtrlE("\x1b[5;5u"), true);
	assert.equal(isCtrlE("\x1b[27;5;5~"), true);
	assert.equal(isCtrlE("\r"), false);
	assert.equal(isCtrlE("\n"), false);
});
