// ── Read tracker ─────────────────────────────────────────────────────────────────
// Records when the model last read each file (by realpath + mtime), so the edit
// tool can detect a file that changed underneath it since it was read and refuse
// to clobber the change. Inspired by openclaude's FILE_UNEXPECTEDLY_MODIFIED_ERROR.

import { realpathSync, statSync } from "node:fs";

interface ReadSnapshot {
	mtimeMs: number;
	size: number;
}

const lastReadSnapshot = new Map<string, ReadSnapshot>();

function keyFor(absolutePath: string): string {
	try {
		return realpathSync(absolutePath);
	} catch (_e: unknown) {
		return absolutePath;
	}
}

/** Record that the model just read this file at its current mtime. */
export function recordRead(absolutePath: string): void {
	try {
		const info = statSync(absolutePath);
		lastReadSnapshot.set(keyFor(absolutePath), {
			mtimeMs: info.mtimeMs,
			size: info.size,
		});
	} catch (_e: unknown) {
		// File vanished between read and stat; nothing to record.
	}
}

/** Returns true if the model has read (or written) this file this session. */
export function hasBeenRead(absolutePath: string): boolean {
	return lastReadSnapshot.has(keyFor(absolutePath));
}

/**
 * Returns true if the file changed on disk since it was last read by the model.
 * Returns false when the file was never read (no read-before-edit requirement)
 * or when its mtime still matches the recorded read.
 *
 * Uses stat-based comparison only (mtime + size). No SHA-256 — file content
 * is not re-read on every write, which would be O(file-size) on every call.
 */
export function isStaleSinceRead(absolutePath: string): boolean {
	const key = keyFor(absolutePath);
	const recorded = lastReadSnapshot.get(key);
	if (recorded === undefined) return false;
	try {
		const info = statSync(absolutePath);
		return (
			info.mtimeMs !== recorded.mtimeMs || info.size !== recorded.size
		);
	} catch (_e: unknown) {
		return false;
	}
}

/** Refresh the recorded mtime after a tool mutates the file itself. */
export function refreshAfterWrite(absolutePath: string): void {
	recordRead(absolutePath);
}
