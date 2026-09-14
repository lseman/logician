import { afterEach, expect, test } from "bun:test";
import { DatabaseSync } from "node:sqlite";
import { copyFileSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { ToolResult } from "@logician/log-core";
import { createReadTool } from "../../capabilities/tools/read-file.ts";
import { hasBeenRead } from "../../capabilities/tools/support/read-tracker.ts";
import { createWriteTool } from "../../capabilities/tools/write-file.ts";
import { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";

const dirs: string[] = [];
afterEach(() => {
	for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});
function temp(): string {
	const dir = mkdtempSync(path.join(tmpdir(), "archive-sqlite-tools-"));
	dirs.push(dir);
	return dir;
}
function result(value: string | ToolResult): ToolResult {
	return typeof value === "string" ? { content: value } : value;
}
function tools() {
	const urls = new InternalUrlRouter();
	return { read: createReadTool(urls), write: createWriteTool(undefined, urls) };
}

// ── zip family ──────────────────────────────────────────────────────────────

test("zip: write creates a member, read returns it, a second write to a new member preserves the first", async () => {
	const cwd = temp();
	const { read, write } = tools();

	const first = result(
		await write.execute({ path: "a.zip:foo.txt", content: "hello zip" }, { cwd }),
	);
	expect(first.content).toContain("Created a.zip:foo.txt");
	expect(
		result(await read.execute({ path: "a.zip:foo.txt" }, { cwd })).content,
	).toContain("1:hello zip");

	await read.execute({ path: "a.zip:foo.txt" }, { cwd }); // read-before-overwrite gate
	const second = result(
		await write.execute({ path: "a.zip:bar.txt", content: "second" }, { cwd }),
	);
	expect(second.content).toContain("Created a.zip:bar.txt");

	expect(
		result(await read.execute({ path: "a.zip:foo.txt" }, { cwd })).content,
	).toContain("1:hello zip");
	expect(
		result(await read.execute({ path: "a.zip:bar.txt" }, { cwd })).content,
	).toContain("1:second");
});

test("zip: bare path lists entries", async () => {
	const cwd = temp();
	const { read, write } = tools();
	await write.execute({ path: "a.zip:foo.txt", content: "x" }, { cwd });
	await read.execute({ path: "a.zip:foo.txt" }, { cwd });
	await write.execute({ path: "a.zip:bar.txt", content: "y" }, { cwd });
	const listing = result(await read.execute({ path: "a.zip" }, { cwd }));
	expect(listing.content).toContain("foo.txt");
	expect(listing.content).toContain("bar.txt");
});

// ── tar family ──────────────────────────────────────────────────────────────

test("tar and tar.gz: write creates a member, read returns it, listing paginates like a directory", async () => {
	const cwd = temp();
	const { read, write } = tools();
	for (const name of ["b.tar", "c.tar.gz"]) {
		await write.execute({ path: `${name}:foo.txt`, content: "hello tar" }, { cwd });
		expect(
			result(await read.execute({ path: `${name}:foo.txt` }, { cwd })).content,
		).toContain("1:hello tar");
		expect(
			result(await read.execute({ path: name }, { cwd })).content,
		).toContain("foo.txt");
	}
});

test("tar.gz: a second member write preserves the first member's content", async () => {
	const cwd = temp();
	const { read, write } = tools();
	await write.execute({ path: "b.tar.gz:foo.txt", content: "hello tar" }, { cwd });
	await read.execute({ path: "b.tar.gz:foo.txt" }, { cwd });
	await write.execute({ path: "b.tar.gz:bar.txt", content: "second" }, { cwd });
	expect(
		result(await read.execute({ path: "b.tar.gz:foo.txt" }, { cwd })).content,
	).toContain("1:hello tar");
	expect(
		result(await read.execute({ path: "b.tar.gz:bar.txt" }, { cwd })).content,
	).toContain("1:second");
});

test("a corrupt tar.gz is rejected cleanly instead of crashing the process", async () => {
	// Regression test: .pipe() does not forward source-stream errors, so a
	// naive pipe chain (gunzip -> tar-extract) previously left the gunzip
	// stream's 'error' event unhandled, crashing the whole process on any
	// malformed .tar.gz instead of rejecting the read.
	const cwd = temp();
	const { read } = tools();
	writeFileSync(path.join(cwd, "corrupt.tar.gz"), "not valid gzip data");
	const output = result(
		await read.execute({ path: "corrupt.tar.gz:foo.txt" }, { cwd }),
	);
	expect(output.isError).toBe(true);
});

// ── archive negative cases ──────────────────────────────────────────────────

test("archive writes reject path traversal in the member path", async () => {
	const cwd = temp();
	const { write } = tools();
	const output = result(
		await write.execute({ path: "a.zip:../escape.txt", content: "x" }, { cwd }),
	);
	expect(output.content).toContain("Error");
});

test("archive write to an existing container this session never touched is rejected (staleness gate)", async () => {
	const cwd = temp();
	const { write } = tools();
	await write.execute({ path: "a.zip:foo.txt", content: "x" }, { cwd });
	// Simulate a container that exists on disk but this write/read pair never
	// touched: writing a.zip already marks it "read" (same as any file write
	// refreshing its own read-tracking), so copy it to an untouched path instead.
	copyFileSync(path.join(cwd, "a.zip"), path.join(cwd, "untouched.zip"));
	const output = result(
		await write.execute({ path: "untouched.zip:bar.txt", content: "y" }, { cwd }),
	);
	expect(output.content).toContain("has not been read");
});

test("append is not supported for archive members", async () => {
	const cwd = temp();
	const { write } = tools();
	const output = result(
		await write.execute(
			{ path: "a.zip:foo.txt", content: "x", append: true },
			{ cwd },
		),
	);
	expect(output.content).toContain("append is not supported");
});

// ── sqlite ──────────────────────────────────────────────────────────────────

function createSqliteFixture(cwd: string): string {
	const file = path.join(cwd, "d.sqlite");
	const db = new DatabaseSync(file);
	db.exec("CREATE TABLE people (name TEXT, age INTEGER)");
	db.prepare("INSERT INTO people (name, age) VALUES (?, ?)").run("Ada", 30);
	db.close();
	return file;
}

test("sqlite: bare path lists tables, table path shows schema+rows, table:rowid shows one row", async () => {
	const cwd = temp();
	createSqliteFixture(cwd);
	const { read } = tools();

	const tableList = result(await read.execute({ path: "d.sqlite" }, { cwd }));
	expect(tableList.content).toContain("people");

	const tableRead = result(await read.execute({ path: "d.sqlite:people" }, { cwd }));
	expect(tableRead.content).toContain("CREATE TABLE");
	expect(tableRead.content).toContain("Ada");

	const rowRead = result(await read.execute({ path: "d.sqlite:people:1" }, { cwd }));
	expect(rowRead.content).toContain("Ada");
	expect(rowRead.content).toContain("30");
});

test("sqlite: insert creates a new file and table from JSON keys, update and delete work by rowid", async () => {
	const cwd = temp();
	const { read, write } = tools();

	const inserted = result(
		await write.execute(
			{ path: "e.sqlite:people", content: JSON.stringify({ name: "Ada", age: 30 }) },
			{ cwd },
		),
	);
	expect(inserted.content).toContain("Inserted into people");

	const updated = result(
		await write.execute(
			{ path: "e.sqlite:people:1", content: JSON.stringify({ age: 31 }) },
			{ cwd },
		),
	);
	expect(updated.content).toContain("Updated people:1");
	expect(
		result(await read.execute({ path: "e.sqlite:people:1" }, { cwd })).content,
	).toContain("31");

	const deleted = result(
		await write.execute({ path: "e.sqlite:people:1", content: "" }, { cwd }),
	);
	expect(deleted.content).toContain("Deleted people:1");
	const afterDelete = result(await read.execute({ path: "e.sqlite:people:1" }, { cwd }));
	expect(afterDelete.isError).toBe(true);
	expect(afterDelete.content).toContain("No such row");
});

test("sqlite write requires no prior read (no staleness gate, unlike archives)", async () => {
	const cwd = temp();
	createSqliteFixture(cwd);
	const { write } = tools();
	// No read() call first — should still succeed.
	const output = result(
		await write.execute(
			{ path: "d.sqlite:people", content: JSON.stringify({ name: "Grace", age: 25 }) },
			{ cwd },
		),
	);
	expect(output.content).toContain("Inserted into people");
});

test("sqlite update/delete against a missing table return a clear error rather than creating one", async () => {
	const cwd = temp();
	createSqliteFixture(cwd);
	const { write } = tools();
	const updated = result(
		await write.execute(
			{ path: "d.sqlite:missing:1", content: JSON.stringify({ x: 1 }) },
			{ cwd },
		),
	);
	expect(updated.content).toContain("No such table");

	const deleted = result(
		await write.execute({ path: "d.sqlite:missing:1", content: "" }, { cwd }),
	);
	expect(deleted.content).toContain("No such table");
});

test("sqlite insert/update reject JSON keys that aren't safe SQL identifiers", async () => {
	const cwd = temp();
	createSqliteFixture(cwd);
	const { write } = tools();

	const maliciousKey = 'x" ,(SELECT 1)-- ';
	const insertAttempt = result(
		await write.execute(
			{
				path: "d.sqlite:injected",
				content: JSON.stringify({ [maliciousKey]: 1 }),
			},
			{ cwd },
		),
	);
	expect(insertAttempt.content).toContain("Invalid column name");

	const updateAttempt = result(
		await write.execute(
			{
				path: "d.sqlite:people:1",
				content: JSON.stringify({ [maliciousKey]: 1 }),
			},
			{ cwd },
		),
	);
	expect(updateAttempt.content).toContain("Invalid column name");
});

test("append is not supported for SQLite rows", async () => {
	const cwd = temp();
	createSqliteFixture(cwd);
	const { write } = tools();
	const output = result(
		await write.execute(
			{
				path: "d.sqlite:people",
				content: JSON.stringify({ name: "x", age: 1 }),
				append: true,
			},
			{ cwd },
		),
	);
	expect(output.content).toContain("append is not supported");
});

test("reading a non-existent table or rowid returns a clear message, not a thrown internal error", async () => {
	const cwd = temp();
	createSqliteFixture(cwd);
	const { read } = tools();
	expect(
		result(await read.execute({ path: "d.sqlite:missing" }, { cwd })).content,
	).toContain("Error");
	expect(
		result(await read.execute({ path: "d.sqlite:people:999" }, { cwd })).content,
	).toContain("Error");
});

// ── read-tracking parity with plain files ───────────────────────────────────

test("hasBeenRead(container) becomes true after reading an archive member or sqlite table", async () => {
	const cwd = temp();
	const { read } = tools();

	// Built directly on disk (not via write.execute) so it starts untracked.
	const zipSource = temp();
	const { write: seedWrite } = tools();
	await seedWrite.execute({ path: "seed.zip:foo.txt", content: "x" }, { cwd: zipSource });
	copyFileSync(path.join(zipSource, "seed.zip"), path.join(cwd, "a.zip"));
	expect(hasBeenRead(path.join(cwd, "a.zip"))).toBe(false);
	await read.execute({ path: "a.zip:foo.txt" }, { cwd });
	expect(hasBeenRead(path.join(cwd, "a.zip"))).toBe(true);

	const sqlitePath = createSqliteFixture(cwd);
	expect(hasBeenRead(sqlitePath)).toBe(false);
	await read.execute({ path: "d.sqlite:people" }, { cwd });
	expect(hasBeenRead(sqlitePath)).toBe(true);
});
