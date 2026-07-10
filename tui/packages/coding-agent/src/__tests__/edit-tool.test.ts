import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { edit_file } from "../tools/edit-file.ts";

void test("edit_file rejects missing oldText instead of silently succeeding", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-edit-"));
	writeFileSync(join(cwd, "file.txt"), "alpha\nbeta\n", "utf8");

	await assert.rejects(
		edit_file.execute(
			{ path: "file.txt", edits: [{ oldText: "gamma", newText: "delta" }] },
			{ cwd },
		),
		/Could not find the exact text/,
	);
});

void test("edit_file rejects duplicate oldText matches", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-edit-"));
	writeFileSync(join(cwd, "file.txt"), "same\nsame\n", "utf8");

	await assert.rejects(
		edit_file.execute(
			{ path: "file.txt", edits: [{ oldText: "same", newText: "other" }] },
			{ cwd },
		),
		/Found 2 occurrences/,
	);
});

void test("edit_file preserves BOM and CRLF line endings", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-edit-"));
	const file = join(cwd, "file.txt");
	writeFileSync(file, "\uFEFFline1\r\nline2\r\n", "utf8");

	await edit_file.execute(
		{ path: "file.txt", edits: [{ oldText: "line2\n", newText: "line two\n" }] },
		{ cwd },
	);

	const content = readFileSync(file, "utf8");
	assert.equal(content, "\uFEFFline1\r\nline two\r\n");
});
