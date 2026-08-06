import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { grep } from "../tools/search.ts";
import { ensureTool } from "../tools/shared/tools-manager.ts";

void test("grep prepareArguments does not turn the search pattern into a glob", () => {
	const args = grep.prepareArguments?.({ pattern: "needle" }) ?? {};
	assert.equal(args.pattern, "needle");
	assert.equal(args.glob, undefined);
});

void test("grep finds plain prepared searches across normal filenames", async t => {
	if (!(await ensureTool("rg"))) {
		t.skip("ripgrep is not available");
		return;
	}

	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "notes.txt"), "alpha\nneedle\nomega\n", "utf8");

	const args = grep.prepareArguments?.({ pattern: "needle" }) ?? {
		pattern: "needle",
	};
	const result = await grep.execute(args, { cwd });
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /notes\.txt:2: needle/);
});

void test("grep reports ripgrep pattern errors instead of no matches", async t => {
	if (!(await ensureTool("rg"))) {
		t.skip("ripgrep is not available");
		return;
	}

	const cwd = mkdtempSync(join(tmpdir(), "logician-grep-"));
	writeFileSync(join(cwd, "notes.txt"), "alpha\n", "utf8");

	const result = await grep.execute({ pattern: "[" }, { cwd });
	const content = typeof result === "string" ? result : result.content;
	assert.match(content, /^Error: /);
	assert.notEqual(content, "No matches found.");
});
