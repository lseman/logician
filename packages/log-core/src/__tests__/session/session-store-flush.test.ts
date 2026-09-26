import { test, vi } from "bun:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SessionStore } from "../../session/session-store.ts";

const msg = (content: string) => ({
	role: "user" as const,
	content,
	timestamp: Date.now(),
});

const NEAR_NEVER_MS = 60_000;

test("journal writes are immediate; metadata coalesces until flush", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	const store = new SessionStore("flush-meta", {
		baseDir: dir,
		flushDelayMs: NEAR_NEVER_MS,
	});
	store.append(msg("one"));
	store.append(msg("two"));
	// Metadata is coalesced: the projection is stale until a flush.
	assert.equal(store.getMeta().messageCount, 0);
	// The journal write is synchronous, so a fresh instance sees the data
	// without waiting for the coalesced flush.
	const reloaded = new SessionStore("flush-meta", { baseDir: dir });
	assert.deepEqual(
		reloaded.load().map(m => m.content),
		["one", "two"],
	);
	store.flush();
	assert.equal(store.getMeta().messageCount, 2);
});

test("large appends flush immediately via the pending-bytes threshold", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	const store = new SessionStore("flush-bytes", {
		baseDir: dir,
		flushDelayMs: NEAR_NEVER_MS,
		maxPendingBytes: 100,
	});
	store.append({
		role: "user",
		content: "x".repeat(200),
		timestamp: Date.now(),
	});
	assert.equal(store.getMeta().messageCount, 1);
});

test("coalesced flush fires after the delay window", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	vi.useFakeTimers();
	try {
		const store = new SessionStore("flush-timer", {
			baseDir: dir,
			flushDelayMs: 20,
		});
		store.append(msg("timer"));
		vi.advanceTimersByTime(30);
		assert.equal(store.getMeta().messageCount, 1);
	} finally {
		vi.useRealTimers();
	}
});

test("clear cancels the pending flush without resurrecting the directory", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	vi.useFakeTimers();
	try {
		const store = new SessionStore("flush-clear", {
			baseDir: dir,
			flushDelayMs: 10,
		});
		store.append(msg("doomed"));
		store.clear();
		// If the timer were still armed, the flush would recreate the
		// directory with a stale metadata file.
		vi.advanceTimersByTime(100);
		assert.equal(existsSync(store.dirPath), false);
		store.flush();
		assert.equal(existsSync(store.dirPath), false);
	} finally {
		vi.useRealTimers();
	}
});

test("truncate settles coalesced state before rewriting", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	const store = new SessionStore("flush-trunc", {
		baseDir: dir,
		flushDelayMs: NEAR_NEVER_MS,
	});
	for (let i = 0; i < 5; i++) store.append(msg(`m${i}`));
	store.truncate(2);
	assert.deepEqual(
		store.load().map(m => m.content),
		["m3", "m4"],
	);
	assert.equal(store.getMeta().messageCount, 2);
});

test("appends after truncate land in the rewritten journal file", () => {
	// truncate() replaces the journal via an atomic rename, which swaps the
	// file's inode. A cached append handle would keep writing to the old
	// (now unlinked) inode, silently losing entries — so truncate must drop
	// the handle and the next append must open the new file.
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	const store = new SessionStore("flush-trunc-append", {
		baseDir: dir,
		flushDelayMs: NEAR_NEVER_MS,
	});
	for (let i = 0; i < 5; i++) store.append(msg(`m${i}`));
	store.truncate(2);
	store.append(msg("after-truncate"));
	store.flush();
	assert.deepEqual(
		store.load().map(m => m.content),
		["m3", "m4", "after-truncate"],
	);
	// A fresh instance reads the on-disk file, not the in-memory tree.
	const reloaded = new SessionStore("flush-trunc-append", { baseDir: dir });
	assert.deepEqual(
		reloaded.load().map(m => m.content),
		["m3", "m4", "after-truncate"],
	);
});

test("append recreates the journal when its directory was removed", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	const store = new SessionStore("flush-rmdir", {
		baseDir: dir,
		flushDelayMs: NEAR_NEVER_MS,
	});
	store.append(msg("before"));
	store.flush();
	rmSync(store.dirPath, { recursive: true, force: true });
	// The cached handle is stale; the next append must rebuild the directory
	// and file (ENOENT retry path) instead of throwing or writing nowhere.
	store.append(msg("after"));
	store.flush();
	const reloaded = new SessionStore("flush-rmdir", { baseDir: dir });
	assert.deepEqual(
		reloaded.load().map(m => m.content),
		["after"],
	);
});

test("close flushes pending bytes so a reloaded store sees them", () => {
	const dir = mkdtempSync(join(tmpdir(), "logician-flush-"));
	const store = new SessionStore("flush-close", {
		baseDir: dir,
		flushDelayMs: NEAR_NEVER_MS,
	});
	store.append(msg("a"));
	store.append(msg("b"));
	store.close();
	const reloaded = new SessionStore("flush-close", { baseDir: dir });
	assert.deepEqual(
		reloaded.load().map(m => m.content),
		["a", "b"],
	);
	assert.equal(reloaded.getMeta().messageCount, 2);
});
