import assert from "node:assert";
import { existsSync as existsSyncSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { EOH_DIR, ensureParentDir, sessionFilePath } from "../src/paths.ts";

test("sessionFilePath returns correct path for log", () => {
	const dir = "/tmp/test-eoh";
	assert.equal(
		sessionFilePath(dir, "log"),
		path.join(dir, EOH_DIR, "log.jsonl"),
	);
	assert.equal(
		sessionFilePath(dir, "problem"),
		path.join(dir, EOH_DIR, "problem.json"),
	);
	assert.equal(
		sessionFilePath(dir, "config"),
		path.join(dir, EOH_DIR, "config.json"),
	);
	assert.equal(
		sessionFilePath(dir, "prompt"),
		path.join(dir, EOH_DIR, "prompt.md"),
	);
});

test("ensureParentDir creates directory", async () => {
	const tmpDir = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-test-"));
	try {
		const filePath = path.join(tmpDir, "nested", "deep", "file.jsonl");
		ensureParentDir(filePath);
		assert.ok(existsSyncSync(path.dirname(filePath)));
	} finally {
		await rm(tmpDir, { recursive: true, force: true });
	}
});

test("sessionFilePath always uses .eoh/ layout (no legacy)", () => {
	// EoH has no legacy flat files — always .eoh/
	assert.equal(sessionFilePath("/some/dir", "log").includes(".eoh"), true);
});
