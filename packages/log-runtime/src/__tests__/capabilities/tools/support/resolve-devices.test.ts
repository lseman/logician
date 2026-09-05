// ── resolve-devices tests ─────────────────────────────────────────────────────

import { test, describe, beforeEach } from "bun:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { tmpdir } from "node:os";
import { mkdtempSync } from "node:fs";
import { handleResolve } from "../../../../capabilities/tools/support/resolve-devices.ts";

describe("handleResolve", () => {
	let tmp: string;

	beforeEach(() => {
		tmp = mkdtempSync(path.join(tmpdir(), "resolve-test-"));
	});

	function makeArgs(files: Array<{ path: string; content: string }>) {
		return {
			reason: "test",
			sourceToolName: "test",
			label: "test",
			files,
		};
	}

	test("writes new file and reports affected", async () => {
		const filePath = path.join(tmp, "new.txt");
		const result = await handleResolve(makeArgs([{ path: "new.txt", content: "hello world\n" }]), tmp);
		assert.ok(result.success);
		assert.equal(result.filesAffected, 1);
		assert.equal(result.linesChanged, 2);
		assert.equal(fs.readFileSync(filePath, "utf8"), "hello world\n");
	});

	test("skips unchanged files", async () => {
		const filePath = path.join(tmp, "unchanged.txt");
		fs.writeFileSync(filePath, "same content\n");

		const result = await handleResolve(makeArgs([{ path: "unchanged.txt", content: "same content\n" }]), tmp);
		assert.ok(result.success);
		assert.equal(result.filesAffected, 0);
		assert.equal(result.linesChanged, 0);
	});

	test("writes changed files and skips unchanged in mixed batch", async () => {
		const unchangedFile = path.join(tmp, "keep.txt");
		const changedFile = path.join(tmp, "modify.txt");
		fs.writeFileSync(unchangedFile, "original\n");
		fs.writeFileSync(changedFile, "original\n");

		const result = await handleResolve(
			makeArgs([
				{ path: "keep.txt", content: "original\n" },
				{ path: "modify.txt", content: "changed\n" },
			]),
			tmp,
		);
		assert.ok(result.success);
		assert.equal(result.filesAffected, 1);
		assert.equal(fs.readFileSync(changedFile, "utf8"), "changed\n");
		// Unchanged file should still have original content
		assert.equal(fs.readFileSync(unchangedFile, "utf8"), "original\n");
	});

	test("returns failure message on write error", async () => {
		const result = await handleResolve(
			makeArgs([{ path: "nonexistent-dir/file.txt", content: "data\n" }]),
			"/root", // Invalid cwd to trigger error
		);
		assert.ok(!result.success);
		assert.ok(result.message.includes("Failed to write"));
	});
});
