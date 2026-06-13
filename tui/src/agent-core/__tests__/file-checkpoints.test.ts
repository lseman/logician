import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
	beginFileFrame,
	clearFileFrames,
	currentFrameSize,
	recordFileBeforeWrite,
	restoreFileFrame,
} from "../file-checkpoints.ts";

void test("restore rewrites a modified file and deletes a created one", () => {
	clearFileFrames();
	const dir = mkdtempSync(join(tmpdir(), "fcp-"));
	const existing = join(dir, "a.txt");
	const created = join(dir, "new.txt");
	writeFileSync(existing, "before", "utf8");

	beginFileFrame();
	// Agent is about to write both files: record pre-states, then "write".
	recordFileBeforeWrite(existing, dir);
	recordFileBeforeWrite(created, dir);
	writeFileSync(existing, "after", "utf8");
	writeFileSync(created, "fresh", "utf8");
	assert.equal(currentFrameSize(), 2);

	const restored = restoreFileFrame();
	assert.equal(restored, 2);
	assert.equal(readFileSync(existing, "utf8"), "before");
	assert.equal(existsSync(created), false);
});

void test("only the first write per path is recorded in a frame", () => {
	clearFileFrames();
	const dir = mkdtempSync(join(tmpdir(), "fcp-"));
	const file = join(dir, "a.txt");
	writeFileSync(file, "v1", "utf8");

	beginFileFrame();
	recordFileBeforeWrite(file, dir);
	writeFileSync(file, "v2", "utf8");
	recordFileBeforeWrite(file, dir); // second write same turn — ignored
	writeFileSync(file, "v3", "utf8");

	restoreFileFrame();
	assert.equal(readFileSync(file, "utf8"), "v1");
});

void test("restore with no frame returns null; recording without a frame is a no-op", () => {
	clearFileFrames();
	assert.equal(restoreFileFrame(), null);
	recordFileBeforeWrite("/tmp/whatever.txt");
	assert.equal(currentFrameSize(), 0);
});
