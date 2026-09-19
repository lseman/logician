import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { listProjectFiles } from "../../runtime/context/file-mentions.ts";

void test("listProjectFiles lists files recursively without directories", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-mentions-"));
	writeFileSync(join(cwd, "top.txt"), "", "utf8");
	mkdirSync(join(cwd, "sub"), { recursive: true });
	writeFileSync(join(cwd, "sub", "nested.txt"), "", "utf8");

	const files = await listProjectFiles(cwd);
	assert.ok(files.includes("top.txt"));
	assert.ok(files.includes("sub/nested.txt"));
	assert.ok(!files.some(f => f.endsWith("/")), "directories should not be listed");
});

void test("listProjectFiles respects .gitignore", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-mentions-"));
	mkdirSync(join(cwd, ".git"), { recursive: true });
	writeFileSync(join(cwd, ".gitignore"), "ignored.txt\n", "utf8");
	writeFileSync(join(cwd, "ignored.txt"), "", "utf8");
	writeFileSync(join(cwd, "kept.txt"), "", "utf8");

	const files = await listProjectFiles(cwd);
	assert.ok(files.includes("kept.txt"));
	assert.ok(!files.includes("ignored.txt"));
});
