import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { glob } from "../../capabilities/tools/glob.ts";

function makeCwd(): string {
	return mkdtempSync(join(tmpdir(), "logician-glob-"));
}

async function run(args: Record<string, unknown>, cwd: string): Promise<string> {
	const prepared = glob.prepareArguments?.(args) ?? args;
	const result = await glob.execute(prepared, { cwd });
	return typeof result === "string" ? result : result.content;
}

void test("glob lists a bare directory recursively", async () => {
	const cwd = makeCwd();
	writeFileSync(join(cwd, "a.txt"), "a", "utf8");
	mkdirSync(join(cwd, "sub"), { recursive: true });
	writeFileSync(join(cwd, "sub", "b.txt"), "b", "utf8");

	const content = await run({ path: "." }, cwd);
	assert.match(content, /a\.txt/);
	assert.match(content, /sub\/b\.txt/);
	assert.match(content, /sub\//);
});

void test("glob short-circuits a bare file to its relative path", async () => {
	const cwd = makeCwd();
	writeFileSync(join(cwd, "only.txt"), "content", "utf8");

	const content = await run({ path: "only.txt" }, cwd);
	assert.equal(content, "only.txt");
});

void test("glob matches by extension pattern", async () => {
	const cwd = makeCwd();
	writeFileSync(join(cwd, "keep.ts"), "", "utf8");
	writeFileSync(join(cwd, "skip.js"), "", "utf8");
	mkdirSync(join(cwd, "nested"), { recursive: true });
	writeFileSync(join(cwd, "nested", "also.ts"), "", "utf8");

	const content = await run({ path: "*.ts" }, cwd);
	assert.match(content, /keep\.ts/);
	assert.match(content, /nested\/also\.ts/);
	assert.doesNotMatch(content, /skip\.js/);
});

void test("glob respects .gitignore", async () => {
	const cwd = makeCwd();
	mkdirSync(join(cwd, ".git"), { recursive: true });
	writeFileSync(join(cwd, ".gitignore"), "ignored.txt\n", "utf8");
	writeFileSync(join(cwd, "ignored.txt"), "", "utf8");
	writeFileSync(join(cwd, "kept.txt"), "", "utf8");

	const content = await run({ path: "*.txt" }, cwd);
	assert.match(content, /kept\.txt/);
	assert.doesNotMatch(content, /ignored\.txt/);
});

void test("glob reports an error for a missing path", async () => {
	const cwd = makeCwd();
	const content = await run({ path: "does-not-exist" }, cwd);
	assert.match(content, /^Error: Path not found/);
});

void test("glob truncates results and notes the limit", async () => {
	const cwd = makeCwd();
	for (let i = 0; i < 5; i++) {
		writeFileSync(join(cwd, `file${i}.txt`), "", "utf8");
	}

	const content = await run({ path: "*.txt", limit: 2 }, cwd);
	const fileLines = content.split("\n").filter(line => line.endsWith(".txt"));
	assert.equal(fileLines.length, 2);
	assert.match(content, /results limit reached/);
});

void test("glob prepareArguments accepts a bare string as path", () => {
	const args = glob.prepareArguments?.("src/**/*.ts") ?? {};
	assert.equal(args.path, "src/**/*.ts");
});
