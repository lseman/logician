// ── Read tracker ─────────────────────────────────────────────────────────────────
// Records when the model last read each file (by realpath + mtime), so the edit
// tool can detect a file that changed underneath it since it was read and refuse
// to clobber the change. Inspired by openclaude's FILE_UNEXPECTEDLY_MODIFIED_ERROR.

import { realpathSync, statSync } from "node:fs";

const lastReadMtime = new Map<string, number>();

function keyFor(absolutePath: string): string {
    try {
        return realpathSync(absolutePath);
    } catch {
        return absolutePath;
    }
}

/** Record that the model just read this file at its current mtime. */
export function recordRead(absolutePath: string): void {
    try {
        lastReadMtime.set(keyFor(absolutePath), statSync(absolutePath).mtimeMs);
    } catch {
        // File vanished between read and stat; nothing to record.
    }
}

/**
 * Returns true if the file changed on disk since it was last read by the model.
 * Returns false when the file was never read (no read-before-edit requirement)
 * or when its mtime still matches the recorded read.
 */
export function isStaleSinceRead(absolutePath: string): boolean {
    const key = keyFor(absolutePath);
    const recorded = lastReadMtime.get(key);
    if (recorded === undefined) return false;
    try {
        return statSync(absolutePath).mtimeMs > recorded;
    } catch {
        return false;
    }
}

/** Refresh the recorded mtime after a tool mutates the file itself. */
export function refreshAfterWrite(absolutePath: string): void {
    recordRead(absolutePath);
}
