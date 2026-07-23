// ── Read tracker ─────────────────────────────────────────────────────────────────
// Records when the model last read each file (by realpath + mtime), so the edit
// tool can detect a file that changed underneath it since it was read and refuse
// to clobber the change. Inspired by openclaude's FILE_UNEXPECTEDLY_MODIFIED_ERROR.

import { createHash } from "node:crypto";
import { readFileSync, realpathSync, statSync } from "node:fs";

interface ReadSnapshot {
	mtimeMs: number;
	size: number;
	sha256: string;
}

const lastReadSnapshot = new Map<string, ReadSnapshot>();

function keyFor(absolutePath: string): string {
	try {
		return realpathSync(absolutePath);
	} catch (e: unknown) {
		return absolutePath;
	}
}

/** Record that the model just read this file at its current mtime. */
export function recordRead(absolutePath: string): void {
	try {
		const info = statSync(absolutePath);
		const sha256 = createHash("sha256").update(readFileSync(absolutePath)).digest("hex");
		lastReadSnapshot.set(keyFor(absolutePath), {
			mtimeMs: info.mtimeMs,
			size: info.size,
			sha256,
		});
	} catch (e: unknown) {
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
 */
export function isStaleSinceRead(absolutePath: string): boolean {
	const key = keyFor(absolutePath);
	const recorded = lastReadSnapshot.get(key);
	if (recorded === undefined) return false;
	try {
		const info = statSync(absolutePath);
		if (info.mtimeMs !== recorded.mtimeMs || info.size !== recorded.size) return true;
		const sha256 = createHash("sha256").update(readFileSync(absolutePath)).digest("hex");
		return sha256 !== recorded.sha256;
	} catch (e: unknown) {
		return false;
	}
}

/** Refresh the recorded mtime after a tool mutates the file itself. */
export function refreshAfterWrite(absolutePath: string): void {
	recordRead(absolutePath);
}
