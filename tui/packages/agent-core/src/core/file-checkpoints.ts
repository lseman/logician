// ── File checkpoints ─────────────────────────────────────────────────────────
// Snapshot files before the agent's own write tools touch them, so a /rewind
// restores the workspace alongside the conversation. One frame per prompt
// (the harness opens a frame before each turn and pops it on rewind); within
// a frame, only the FIRST write to a path records its pre-state — that is the
// state the frame restores to.
//
// Scope: covers write_file / edit_file (recorded via the builtin
// beforeToolCall hook). Mutations made through bash are NOT captured.
// Module-level singleton, matching the todo / task-status pattern.

import { readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

// Don't snapshot files larger than this — restoring a generated artifact is
// not worth holding it in memory.
const MAX_SNAPSHOT_BYTES = 1024 * 1024;
const MAX_FRAMES = 20;

interface Frame {
	// path → content before the frame's first write; null = file did not exist.
	files: Map<string, string | null>;
}

const frames: Frame[] = [];

/** Open a new checkpoint frame (call before each prompt). */
export function beginFileFrame(): void {
	frames.push({ files: new Map() });
	if (frames.length > MAX_FRAMES) frames.shift();
}

/** Drop all frames (new session / history reset). */
export function clearFileFrames(): void {
	frames.length = 0;
}

/**
 * Record a file's pre-write state into the current frame. No-op without an
 * open frame, for already-recorded paths, and for oversized files.
 */
export function recordFileBeforeWrite(path: string, cwd?: string): void {
	const frame = frames.at(-1);
	if (!frame) return;
	const absolute = resolve(cwd ?? process.cwd(), path);
	if (frame.files.has(absolute)) return;
	try {
		const content = readFileSync(absolute, "utf8");
		if (Buffer.byteLength(content, "utf8") > MAX_SNAPSHOT_BYTES) return;
		frame.files.set(absolute, content);
	} catch {
		// File doesn't exist yet — record null so rewind deletes it.
		frame.files.set(absolute, null);
	}
}

/**
 * Pop the most recent frame and restore every recorded file to its pre-frame
 * state (rewrite, or delete files that did not exist). Returns the number of
 * files restored, or null when no frame exists. Restore failures are counted
 * as not-restored but never throw.
 */
export function restoreFileFrame(): number | null {
	const frame = frames.pop();
	if (!frame) return null;
	let restored = 0;
	for (const [path, content] of frame.files) {
		try {
			if (content === null) unlinkSync(path);
			else writeFileSync(path, content, "utf8");
			restored++;
		} catch {
			// Best-effort: a vanished directory or permission change must not
			// abort the remaining restores.
		}
	}
	return restored;
}

/** Number of files recorded in the current frame (for tests/UI). */
export function currentFrameSize(): number {
	return frames.at(-1)?.files.size ?? 0;
}
