import { test } from "bun:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
	existsSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	beginFileFrame,
	clearFileFrames,
	currentFrameSize,
	recordBashMutations,
	recordFileBeforeWrite,
	restoreFileFrame,
	snapshotBeforeBash,
} from "../../../core/session/file-checkpoints.ts";

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

void test("bash mutations are captured via git snapshots and restored", () => {
	clearFileFrames();
	const dir = mkdtempSync(join(tmpdir(), "fcp-git-"));
	execFileSync("git", ["init", "-q"], { cwd: dir });
	const tracked = join(dir, "tracked.txt");
	const doomed = join(dir, "doomed.txt");
	writeFileSync(tracked, "original\n", "utf8");
	writeFileSync(doomed, "delete me\n", "utf8");

	beginFileFrame();
	const before = snapshotBeforeBash(dir);
	assert.ok(before, "snapshot must work inside a git repo");

	// Simulate what a bash command did: modify, create, delete.
	writeFileSync(tracked, "mutated by bash\n", "utf8");
	writeFileSync(join(dir, "created.txt"), "new file\n", "utf8");
	rmSync(doomed);

	recordBashMutations(before);
	assert.equal(currentFrameSize(), 3);

	const restored = restoreFileFrame();
	assert.equal(restored, 3);
	assert.equal(readFileSync(tracked, "utf8"), "original\n");
	assert.equal(existsSync(join(dir, "created.txt")), false);
	assert.equal(readFileSync(doomed, "utf8"), "delete me\n");
});

void test("bash capture is a silent no-op outside git repositories", () => {
	clearFileFrames();
	const dir = mkdtempSync(join(tmpdir(), "fcp-nogit-"));
	// Guard against tmpdir being inside a repo (it is not on typical systems).
	beginFileFrame();
	const before = snapshotBeforeBash(dir);
	if (before !== null) return; // environment has a repo above tmpdir; skip
	recordBashMutations(before);
	assert.equal(currentFrameSize(), 0);
	restoreFileFrame();
});

void test("bash capture never overrides an earlier write-tool record", () => {
	clearFileFrames();
	const dir = mkdtempSync(join(tmpdir(), "fcp-git2-"));
	execFileSync("git", ["init", "-q"], { cwd: dir });
	const file = join(dir, "a.txt");
	writeFileSync(file, "v1\n", "utf8");

	beginFileFrame();
	// write tool touches it first (records v1), then bash mutates it again.
	recordFileBeforeWrite(file, dir);
	writeFileSync(file, "v2\n", "utf8");
	const before = snapshotBeforeBash(dir);
	writeFileSync(file, "v3\n", "utf8");
	recordBashMutations(before);

	restoreFileFrame();
	assert.equal(readFileSync(file, "utf8"), "v1\n");
});
