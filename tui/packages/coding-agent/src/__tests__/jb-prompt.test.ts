import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { findJbPrompt } from "../runtime/bridge.ts";

void test("findJbPrompt supports launching from the tui directory", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-jb-tui-"));
	writeFileSync(path.join(cwd, "jb.md"), "direct prompt", "utf8");
	assert.equal(findJbPrompt(cwd), "direct prompt");
});

void test("findJbPrompt supports launching from the repository root", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-jb-root-"));
	mkdirSync(path.join(cwd, "tui"));
	writeFileSync(path.join(cwd, "tui", "jb.md"), "nested prompt", "utf8");
	assert.equal(findJbPrompt(cwd), "nested prompt");
});

void test("findJbPrompt returns null when jb.md is unavailable", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-jb-missing-"));
	assert.equal(findJbPrompt(cwd), null);
});
