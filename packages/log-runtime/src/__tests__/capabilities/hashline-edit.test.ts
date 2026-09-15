import { afterEach, expect, test } from "bun:test";
import {
	existsSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { ToolContext, ToolResult } from "@logician/log-core";
import { createPostEditDiagnosticHooks } from "../../capabilities/lsp/post-edit-diagnostics.ts";
import { edit } from "../../capabilities/tools/edit-file.ts";
import { read as readTool } from "../../capabilities/tools/read-file.ts";

const directories: string[] = [];
afterEach(() => {
	for (const dir of directories.splice(0))
		rmSync(dir, { recursive: true, force: true });
});
function fixture(content = "first\nlast\n", name = "file.txt") {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-hashline-test-"));
	directories.push(cwd);
	const file = path.join(cwd, name);
	writeFileSync(file, content);
	const ctx = { cwd } as ToolContext;
	return { cwd, file, name, ctx };
}
function text(result: string | ToolResult): string {
	return typeof result === "string" ? result : result.content;
}

/** Read the fixture through the real `read` tool and pull out its `[path#TAG]`
 * header — the native hashline engine validates TAG against a snapshot that
 * `read` records, not a hash recomputed from disk, so a header can't be
 * precomputed independently of an actual read. */
async function readHeader(f: ReturnType<typeof fixture>): Promise<string> {
	const result = await readTool.execute({ path: f.name }, f.ctx);
	const escaped = f.name.replace(/\./g, "\\.");
	const header = text(result).match(
		new RegExp(`\\[${escaped}#[a-fA-F0-9]{4}\\]`),
	)?.[0];
	if (!header)
		throw new Error(`no hashline header in read output: ${text(result)}`);
	return header;
}

test("read → hashline edit changes the file and runs post-edit diagnostics", async () => {
	// A multi-line file: the native engine's structural validation rejects a
	// PUT that replaces a single-line file's *entire* content with something
	// unparseable (no surrounding structure to anchor the edit against), so
	// the introduced syntax error has to live on an inner line instead.
	const f = fixture('{\n  "valid": true\n}\n', "file.json");
	const header = await readHeader(f);
	const args = { path: f.name, input: `${header}\nPUT 2.=2:\n+  {invalid` };
	const result = await edit.execute(args, f.ctx);
	expect(readFileSync(f.file, "utf8")).toBe("{\n  {invalid\n}\n");
	expect(text(result)).toStartWith(
		"Applied 1 line change(s) across 1 file(s).",
	);
	if (typeof result === "string")
		throw new Error("expected structured edit result");
	expect(result.details?.mutation).toMatchObject({
		kind: "mutation",
		applied: true,
		changed: true,
		paths: [f.file],
		filesAffected: 1,
	});
	const hook = createPostEditDiagnosticHooks(f.cwd).afterToolCall!;
	const diagnostic = await hook({
		toolCall: {
			id: "edit",
			name: "edit",
			arguments: JSON.stringify(args),
		},
		args,
		result: text(result),
		isError: false,
		iteration: 0,
	});
	expect(diagnostic?.content).toContain("<post_edit_diagnostics");
	expect(diagnostic?.content).toContain("file.json:");
});

test("multiple operations keep their target and preserve CRLF and BOM", async () => {
	const f = fixture("﻿first\r\nlast\r\n");
	const header = await readHeader(f);
	const result = await edit.execute(
		{
			path: f.name,
			input: `${header}\nPUT >1:\n+  middle  \nPUT 2.=2:\n+changed`,
		},
		f.ctx,
	);
	expect(text(result)).toStartWith("Applied 2 line change(s)");
	expect(readFileSync(f.file, "utf8")).toBe(
		"﻿first\r\n  middle  \r\nchanged\r\n",
	);
});

test("no-op is reported as informational, not an error", async () => {
	const f = fixture();
	const header = await readHeader(f);
	const result = await edit.execute(
		{ path: f.name, input: `${header}\nPUT 1.=1:\n+first` },
		f.ctx,
	);
	expect(text(result)).toStartWith("No changes made:");
	if (typeof result !== "string") expect(result.isError).toBeFalsy();
	expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
});

test("rejects a file changed on disk since it was read", async () => {
	const f = fixture();
	const header = await readHeader(f);
	writeFileSync(f.file, "someone else's work\n");
	const result = await edit.execute(
		{ path: f.name, input: `${header}\nPUT >1:\n+middle` },
		f.ctx,
	);
	expect(text(result)).toContain("has been modified since it was last read");
	expect(readFileSync(f.file, "utf8")).toBe("someone else's work\n");
});

test("requires a read and rejects a header targeting a different file", async () => {
	const f = fixture();
	const bareHeader = `[${f.name}#0000]`;
	expect(
		text(
			await edit.execute(
				{ path: f.name, input: `${bareHeader}\nPUT >1:\n+middle` },
				f.ctx,
			),
		),
	).toContain("has not been read");

	const header = await readHeader(f);
	const badHeader = header.replace(`${f.name}#`, "../outside.txt#");
	expect(
		text(
			await edit.execute(
				{ path: f.name, input: `${badHeader}\nPUT >1:\n+middle` },
				f.ctx,
			),
		),
	).toContain("must target");
	expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
});

test("rejects an out-of-bounds range and an invalid range without writing", async () => {
	const f = fixture();
	for (const operation of ["PUT >99:\n+oops", "PUT 2.=1:\n+oops"]) {
		const header = await readHeader(f);
		const result = await edit.execute(
			{ path: f.name, input: `${header}\n${operation}` },
			f.ctx,
		);
		expect(text(result)).toStartWith("Error:");
		expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
	}
});

test("REM must be the section's only operation", async () => {
	const f = fixture();
	const header = await readHeader(f);
	const result = await edit.execute(
		{ path: f.name, input: `${header}\nPUT >1:\n+middle\nREM` },
		f.ctx,
	);
	expect(text(result)).toStartWith("Error:");
	expect(existsSync(f.file)).toBe(true);
	expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
});

test("REM deletes the file; MV renames it, applying prior edits to the destination", async () => {
	{
		const f = fixture();
		const header = await readHeader(f);
		const result = await edit.execute(
			{ path: f.name, input: `${header}\nREM` },
			f.ctx,
		);
		expect(text(result)).toStartWith("Applied");
		expect(existsSync(f.file)).toBe(false);
	}
	{
		const f = fixture();
		const header = await readHeader(f);
		const result = await edit.execute(
			{ path: f.name, input: `${header}\nPUT >1:\n+middle\nMV renamed.txt` },
			f.ctx,
		);
		expect(text(result)).toStartWith(
			"Applied 1 line change(s) across 2 file(s).",
		);
		expect(existsSync(f.file)).toBe(false);
		expect(readFileSync(path.join(f.cwd, "renamed.txt"), "utf8")).toBe(
			"first\nmiddle\nlast\n",
		);
	}
});

test("CUT + PUT with a register moves a line within one call", async () => {
	const f = fixture("a\nb\nc\n");
	const header = await readHeader(f);
	const result = await edit.execute(
		{ path: f.name, input: `${header}\nCUT 2.=2 @x\nPUT >3 @x` },
		f.ctx,
	);
	expect(text(result)).toStartWith("Applied");
	expect(readFileSync(f.file, "utf8")).toBe("a\nc\nb\n");
});

test("concurrent proposals based on the same tag cannot overwrite one another", async () => {
	const f = fixture();
	const header = await readHeader(f);
	const results = await Promise.all(
		["alpha", "beta"].map(value =>
			edit.execute(
				{ path: f.name, input: `${header}\nPUT >1:\n+${value}` },
				f.ctx,
			),
		),
	);
	expect(
		results.filter(result => text(result).startsWith("Applied ")),
	).toHaveLength(1);
	expect(
		results.filter(result => text(result).includes("Error:")),
	).toHaveLength(1);
	expect(["first\nalpha\nlast\n", "first\nbeta\nlast\n"]).toContain(
		readFileSync(f.file, "utf8"),
	);
});

test("diagnostics exclude failed and zero-change edits", async () => {
	const f = fixture("{invalid", "file.json");
	const hook = createPostEditDiagnosticHooks(f.cwd).afterToolCall!;
	for (const [result, isError] of [
		["Applied 0 line change(s) across 0 file(s).", false],
		["Applied 1 line change(s) across 1 file(s).", true],
	] as const) {
		expect(
			await hook({
				toolCall: { id: "edit", name: "edit", arguments: "{}" },
				args: { path: f.name },
				result,
				isError,
				iteration: 0,
			}),
		).toBeUndefined();
	}
});
