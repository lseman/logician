import assert from "node:assert/strict";
import {
	chmodSync,
	lstatSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	statSync,
	symlinkSync,
	utimesSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { edit_file } from "../tools/edit-file.ts";
import { write_file } from "../tools/write-file.ts";
import { read_file } from "../tools/read-file.ts";
import { isStaleSinceRead, recordRead } from "../tools/read-tracker.ts";

function setup(name: string, content: string): { cwd: string; file: string } {
	const cwd = mkdtempSync(join(tmpdir(), `logician-${name}-`));
	const file = join(cwd, "file.txt");
	writeFileSync(file, content, "utf8");
	recordRead(file);
	return { cwd, file };
}

void test("edit_file rejects missing oldText instead of silently succeeding", async () => {
	const { cwd } = setup("edit", "alpha\nbeta\n");

	await assert.rejects(
		edit_file.execute(
			{ path: "file.txt", edits: [{ oldText: "gamma", newText: "delta" }] },
			{ cwd },
		),
		/Could not find the exact text/,
	);
});

void test("edit_file rejects duplicate oldText matches with line numbers", async () => {
	const { cwd } = setup("edit", "same\nother\nsame\n");

	await assert.rejects(
		edit_file.execute(
			{ path: "file.txt", edits: [{ oldText: "same", newText: "changed" }] },
			{ cwd },
		),
		/Found 2 occurrences.*lines 1, 3/s,
	);
});

void test("edit_file preserves BOM and CRLF line endings", async () => {
	const { cwd, file } = setup("edit", "\uFEFFline1\r\nline2\r\n");

	await edit_file.execute(
		{ path: "file.txt", edits: [{ oldText: "line2\n", newText: "line two\n" }] },
		{ cwd },
	);

	const content = readFileSync(file, "utf8");
	assert.equal(content, "\uFEFFline1\r\nline two\r\n");
});

void test("edit_file requires the file to have been read first", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-edit-"));
	writeFileSync(join(cwd, "file.txt"), "alpha\n", "utf8");

	const result = await edit_file.execute(
		{ path: "file.txt", edits: [{ oldText: "alpha", newText: "beta" }] },
		{ cwd },
	);
	assert.match(String(result), /has not been read yet/);
});

void test("edit_file replaceAll replaces every occurrence", async () => {
	const { cwd, file } = setup("edit", "foo(1)\nbar\nfoo(2)\nfoo(3)\n");

	await edit_file.execute(
		{
			path: "file.txt",
			edits: [{ oldText: "foo", newText: "qux", replaceAll: true }],
		},
		{ cwd },
	);

	assert.equal(readFileSync(file, "utf8"), "qux(1)\nbar\nqux(2)\nqux(3)\n");
});

void test("edit_file fuzzy match does not rewrite untouched regions", async () => {
	// Smart quote on line 1 must survive a fuzzy edit of line 3.
	const original = "const s = “hello”;\nmiddle\ntarget   \nend\n";
	const { cwd, file } = setup("edit", original);

	await edit_file.execute(
		{ path: "file.txt", edits: [{ oldText: "target\n", newText: "replaced\n" }] },
		{ cwd },
	);

	const content = readFileSync(file, "utf8");
	assert.ok(content.includes("“hello”"), "smart quotes must be preserved");
	assert.ok(content.includes("replaced"));
});

void test("edit_file line-trimmed match re-indents newText to the file", async () => {
	const original = "function f() {\n\t\tconst x = 1;\n\t\treturn x;\n}\n";
	const { cwd, file } = setup("edit", original);

	// Model got the indentation wrong (4 spaces instead of 2 tabs).
	await edit_file.execute(
		{
			path: "file.txt",
			edits: [
				{
					oldText: "    const x = 1;\n    return x;\n",
					newText: "    const x = 2;\n    return x * 2;\n",
				},
			],
		},
		{ cwd },
	);

	assert.equal(
		readFileSync(file, "utf8"),
		"function f() {\n\t\tconst x = 2;\n\t\treturn x * 2;\n}\n",
	);
});

void test("edit_file not-found error hints at the closest matching line", async () => {
	const { cwd } = setup("edit", "aaa\nunique line\nbbb\n");

	await assert.rejects(
		edit_file.execute(
			{
				path: "file.txt",
				edits: [{ oldText: "unique line\nwrong following line", newText: "x" }],
			},
			{ cwd },
		),
		/first 1 line\(s\) match starting at line 2.*oldText line 2 \("wrong following line"\) does not match file line 3 \("bbb"\)/s,
	);
});

void test("write_file refuses to overwrite an unread existing file", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-write-"));
	writeFileSync(join(cwd, "file.txt"), "original\n", "utf8");

	const result = await write_file.execute(
		{ path: "file.txt", content: "clobbered\n" },
		{ cwd },
	);
	assert.match(String(result), /has not been read/);
	assert.equal(readFileSync(join(cwd, "file.txt"), "utf8"), "original\n");
});

void test("write_file overwrites a read file and returns a diff", async () => {
	const { cwd, file } = setup("write", "old content\n");

	const result = await write_file.execute(
		{ path: "file.txt", content: "new content\n" },
		{ cwd },
	);
	assert.equal(readFileSync(file, "utf8"), "new content\n");
	const text =
		typeof result === "string" ? result : (result as { content: string }).content;
	// Strip ANSI escape codes from syntax-highlighted output
	const plain = text.replace(/\x1b\[[\d;]*m/g, "");
	assert.match(plain, /Wrote /);
	assert.match(plain, /new content/);
	assert.ok(!plain.includes("Created"));
});

void test("write_file creates new files without requiring a read", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-write-"));

	const result = await write_file.execute(
		{ path: "sub/new.txt", content: "hello\n" },
		{ cwd },
	);
	assert.match(String(result), /Created /);
	assert.equal(readFileSync(join(cwd, "sub", "new.txt"), "utf8"), "hello\n");
});

void test("stale-read detection catches same-size changes with restored mtime", () => {
	const { file } = setup("stale-hash", "alpha\n");
	const original = statSync(file);
	writeFileSync(file, "bravo\n", "utf8");
	utimesSync(file, original.atime, original.mtime);

	assert.equal(isStaleSinceRead(file), true);
});

void test("atomic overwrite preserves permissions and leaves no temporary file", async () => {
	const { cwd, file } = setup("atomic", "before\n");
	chmodSync(file, 0o640);

	await write_file.execute({ path: "file.txt", content: "after\n" }, { cwd });

	assert.equal(statSync(file).mode & 0o777, 0o640);
	assert.equal(readFileSync(file, "utf8"), "after\n");
	assert.deepEqual(readdirSync(cwd), ["file.txt"]);
});

void test("atomic edits reject symbolic-link targets", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-symlink-"));
	const target = join(cwd, "target.txt");
	const link = join(cwd, "link.txt");
	writeFileSync(target, "before\n", "utf8");
	symlinkSync("target.txt", link);
	recordRead(link);

	await assert.rejects(
		edit_file.execute(
			{ path: "link.txt", edits: [{ oldText: "before", newText: "after" }] },
			{ cwd },
		),
		/Refusing to replace symbolic link/,
	);
	assert.equal(lstatSync(link).isSymbolicLink(), true);
	assert.equal(readFileSync(target, "utf8"), "before\n");
});

void test("read_file rejects binary files", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-read-"));
	writeFileSync(join(cwd, "blob.bin"), Buffer.from([0x89, 0x50, 0x00, 0x47]));

	const result = await read_file.execute({ path: "blob.bin" }, { cwd });
	assert.match(String(result), /binary file/);
});
