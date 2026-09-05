import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { ToolContext, ToolResult } from "@logician/log-core";
import { edit_file } from "../../capabilities/tools/edit-file.ts";
import { read_file } from "../../capabilities/tools/read-file.ts";
import { createEditStore } from "../../capabilities/tools/support/edit-store.ts";
import { previewHashlineEdit } from "../../capabilities/tools/support/hashline-engine.ts";
import { hashlineHash } from "../../capabilities/tools/support/hashline.ts";
import { createPostEditDiagnosticHooks } from "../../capabilities/lsp/post-edit-diagnostics.ts";

const directories: string[] = [];
afterEach(() => {
	for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
});
function fixture(content = "first\nlast\n", name = "file.txt") {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-hashline-test-"));
	directories.push(cwd);
	const file = path.join(cwd, name);
	writeFileSync(file, content);
	const ctx = { cwd } as ToolContext;
	return { cwd, file, name, ctx, input: (operations: string) => `[${name}#${hashlineHash(content)}]\n${operations}` };
}
function text(result: string | ToolResult): string {
	return typeof result === "string" ? result : result.content;
}

test("read → hashline edit changes the file and runs post-edit diagnostics", async () => {
	const f = fixture('{"valid": true}\n', "file.json");
	const read = await read_file.execute({ path: f.name }, f.ctx);
	const header = text(read).match(/\[file\.json#[a-f0-9]{4}\]/)?.[0];
	expect(header).toBeDefined();
	const args = { path: f.name, input: `${header}\nPUT 1.=1:{invalid` };
	const result = await edit_file.execute(args, f.ctx);
	expect(readFileSync(f.file, "utf8")).toBe("{invalid\n");
	expect(text(result)).toStartWith("Applied 1 line change(s) across 1 file(s).");
	if (typeof result === "string") throw new Error("expected structured edit result");
	expect(result.details?.mutation).toMatchObject({
		kind: "mutation",
		applied: true,
		changed: true,
		paths: [f.file],
		filesAffected: 1,
	});
	const hook = createPostEditDiagnosticHooks(f.cwd).afterToolCall!;
	const diagnostic = await hook({ toolCall: { id: "edit", name: "edit_file", arguments: JSON.stringify(args) }, args, result: text(result), isError: false, iteration: 0 });
	expect(diagnostic?.content).toContain("<post_edit_diagnostics");
	expect(diagnostic?.content).toContain("file.json:");
});

test("multiple operations keep their target and preserve CRLF, BOM, and literal whitespace", async () => {
	const f = fixture("\uFEFFfirst\r\nlast\r\n");
	await read_file.execute({ path: f.name }, f.ctx);
	const result = await edit_file.execute({ path: f.name, input: f.input("PUT >1:  middle  \nPUT 3.=3:changed") }, f.ctx);
	expect(text(result)).toStartWith("Applied 2 line change(s)");
	expect(readFileSync(f.file, "utf8")).toBe("\uFEFFfirst\r\n  middle  \r\nchanged\r\n");
});

test("preview never writes and no-op is not reported as applied", async () => {
	const f = fixture();
	const preview = await previewHashlineEdit(f.input("PUT >1:middle"), createEditStore(), f.cwd);
	expect(preview.applied).toBe(false);
	expect(preview.diff).toContain("middle");
	expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
	await read_file.execute({ path: f.name }, f.ctx);
	const result = await edit_file.execute({ path: f.name, input: f.input("PUT 1.=1:first") }, f.ctx);
	expect(text(result)).toStartWith("No changes made:");
});

test("rejects stale hashes even after a new read", async () => {
	const f = fixture();
	writeFileSync(f.file, "someone else's work\n");
	await read_file.execute({ path: f.name }, f.ctx);
	const result = await edit_file.execute({ path: f.name, input: f.input("PUT >1:middle") }, f.ctx);
	expect(text(result)).toContain("Stale hashline anchor");
	expect(readFileSync(f.file, "utf8")).toBe("someone else's work\n");
});

test("requires a read and rejects a header targeting a different file", async () => {
	const f = fixture();
	const input = f.input("PUT >1:middle");
	expect(text(await edit_file.execute({ path: f.name, input }, f.ctx))).toContain("has not been read");
	await read_file.execute({ path: f.name }, f.ctx);
	expect(text(await edit_file.execute({ path: f.name, input: input.replace("file.txt#", "../outside.txt#") }, f.ctx))).toContain("must target");
	expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
});

test("validates the whole proposal before writing and reports unsupported operations", async () => {
	const f = fixture();
	await read_file.execute({ path: f.name }, f.ctx);
	for (const operation of ["PUT >99:oops", "MV other.txt", "REM", "unexpected content", "PUT 2.=1:oops"]) {
		const result = await edit_file.execute({ path: f.name, input: f.input(`PUT >1:middle\n${operation}`) }, f.ctx);
		expect(text(result)).toStartWith("Error:");
		expect(readFileSync(f.file, "utf8")).toBe("first\nlast\n");
	}
});

test("concurrent proposals based on the same hash cannot overwrite one another", async () => {
	const f = fixture();
	await read_file.execute({ path: f.name }, f.ctx);
	const results = await Promise.all(["alpha", "beta"].map(value => edit_file.execute({ path: f.name, input: f.input(`PUT >1:${value}`) }, f.ctx)));
	expect(results.filter(result => text(result).startsWith("Applied "))).toHaveLength(1);
	expect(results.filter(result => text(result).includes("Stale hashline anchor"))).toHaveLength(1);
	expect(["first\nalpha\nlast\n", "first\nbeta\nlast\n"]).toContain(readFileSync(f.file, "utf8"));
});

test("diagnostics exclude failed and zero-change edits", async () => {
	const f = fixture("{invalid", "file.json");
	const hook = createPostEditDiagnosticHooks(f.cwd).afterToolCall!;
	for (const [result, isError] of [["Applied 0 line change(s) across 0 file(s).", false], ["Applied 1 line change(s) across 1 file(s).", true]] as const) {
		expect(await hook({ toolCall: { id: "edit", name: "edit_file", arguments: "{}" }, args: { path: f.name }, result, isError, iteration: 0 })).toBeUndefined();
	}
});
