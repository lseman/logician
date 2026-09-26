import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ast_edit } from "../../tools/ast-edit.ts";
import { ast_grep } from "../../tools/ast-grep-tool.ts";

// ast_grep/ast_edit run on @logician/log-natives (ast-grep-core via N-API),
// not an external CLI, so these always run — no environment probe needed.
const astGrepTest = test;

function sampleFile(): { cwd: string; relPath: string } {
	const cwd = mkdtempSync(join(tmpdir(), "logician-ast-grep-"));
	const content = [
		"function one() {",
		'  console.log("hello");',
		"}",
		"",
		"function two() {",
		'  console.log("world");',
		"}",
	].join("\n");
	writeFileSync(join(cwd, "sample.ts"), content, "utf8");
	return { cwd, relPath: "sample.ts" };
}

void test("ast_grep requires a pattern and at least one path", async () => {
	const noPattern = await ast_grep.execute({ paths: ["x.ts"] }, { cwd: "." });
	assert.match(noPattern as string, /Provide a `pattern`/);

	const noPaths = await ast_grep.execute({ pattern: "x" }, { cwd: "." });
	assert.match(noPaths as string, /Provide at least one path/);
});

void astGrepTest(
	"ast_grep finds structural matches with correct 1-based line numbers",
	async () => {
		const { cwd, relPath } = sampleFile();
		const result = await ast_grep.execute(
			{ pattern: "console.log($X)", paths: [relPath] },
			{ cwd },
		);
		assert.match(result as string, /sample\.ts:2/);
		assert.match(result as string, /sample\.ts:6/);
		assert.match(result as string, /2 matches\./);
	},
);

void astGrepTest(
	"ast_grep reports no matches cleanly instead of erroring",
	async () => {
		const { cwd, relPath } = sampleFile();
		const result = await ast_grep.execute(
			{ pattern: "process.exit($X)", paths: [relPath] },
			{ cwd },
		);
		assert.equal(result, "No matches found.");
	},
);

void astGrepTest(
	"ast_edit also reports no matches cleanly (regression: exit-code-1-on-no-matches used to surface as an error)",
	async () => {
		const { cwd, relPath } = sampleFile();
		const result = await ast_edit.execute(
			{ ops: [{ pat: "process.exit($X)", out: "bail($X)" }], paths: [relPath] },
			{ cwd },
		);
		assert.equal(result, "No matches found for the given patterns.");
	},
);

void astGrepTest(
	"ast_edit still stages rewrites correctly (regression: --no-ignore used to break every call)",
	async () => {
		const { cwd, relPath } = sampleFile();
		const result = await ast_edit.execute(
			{
				ops: [{ pat: "console.log($X)", out: "logger.info($X)" }],
				paths: [relPath],
			},
			{ cwd },
		);
		assert.match(result as string, /Total: 2 change\(s\) across 1 file/);
	},
);
