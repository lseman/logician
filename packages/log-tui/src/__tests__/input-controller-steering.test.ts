import { test } from "bun:test";
import assert from "node:assert/strict";
import { isCtrlEnter } from "../app/input-controller.ts";

void test("Ctrl+Enter recognizes CSI-u and xterm modifyOtherKeys encodings", () => {
	assert.equal(isCtrlEnter("\x1b[13;5u"), true);
	assert.equal(isCtrlEnter("\x1b[27;5;13~"), true);
	assert.equal(isCtrlEnter("\r"), false);
	assert.equal(isCtrlEnter("\n"), false);
});
