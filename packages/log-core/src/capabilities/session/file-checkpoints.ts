// ── File checkpoints ─────────────────────────────────────────────────────────
// Snapshot files before the agent's own tools touch them, so a /rewind
// restores the workspace alongside the conversation. One frame per prompt
// (the harness opens a frame before each turn and pops it on rewind); within
// a frame, only the FIRST write to a path records its pre-state — that is the
// state the frame restores to.
//
// Two capture layers, both recorded via the builtin tool-call hooks:
// 1. write_file / edit_file: the path's content is read before the write.
// 2. bash: the working tree is snapshotted to a git tree object (via a
//    temporary index — worktree and real index untouched) before and after
//    the command; the diff yields the paths bash mutated, and their pre-call
//    contents are recorded from the "before" tree. Restores stay per-file, so
//    a rewind never reverts edits the user made outside the agent's tools.
//    Requires a git repository; gitignored and >1MB files are not captured.
//
// All git failures are silent — bash mutations are then simply not captured.
// Module-level singleton, matching the todo / task-status pattern.

import { execFileSync } from "node:child_process";
import { readFileSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

// Don't snapshot files larger than this — restoring a generated artifact is
// not worth holding it in memory.
const MAX_SNAPSHOT_BYTES = 1024 * 1024;
const MAX_FRAMES = 20;
const GIT_TIMEOUT_MS = 15_000;

interface Frame {
	// path → content before the frame's first write; null = file did not exist.
	files: Map<string, string | null>;
}

/** Opaque token from snapshotBeforeBash, consumed by recordBashMutations. */
export interface WorkspaceSnapshot {
	root: string;
	tree: string;
}

const frames: Frame[] = [];
let indexSeq = 0;

function git(root: string, args: string[], env?: NodeJS.ProcessEnv): string {
	return execFileSync("git", args, {
		cwd: root,
		env: env ?? process.env,
		timeout: GIT_TIMEOUT_MS,
		stdio: ["ignore", "pipe", "ignore"],
		maxBuffer: 64 * 1024 * 1024,
	})
		.toString("utf8")
		.trim();
}

/**
 * Write the current working tree (tracked + untracked, gitignore respected)
 * to a tree object using a temporary index. Returns null when cwd is not in a
 * git repository or any git step fails.
 */
function gitSnapshotTree(cwd: string): WorkspaceSnapshot | null {
	try {
		const root = git(cwd, ["rev-parse", "--show-toplevel"]);
		const indexFile = join(
			tmpdir(),
			`logician-ckpt-${process.pid}-${++indexSeq}`,
		);
		const env = { ...process.env, GIT_INDEX_FILE: indexFile };
		try {
			git(root, ["add", "-A"], env);
			const tree = git(root, ["write-tree"], env);
			return { root, tree };
		} finally {
			try {
				unlinkSync(indexFile);
			} catch (_e: unknown) {
				// Temp index may not exist when add failed early.
			}
		}
	} catch (_e: unknown) {
		return null;
	}
}

/** Read one file's content out of a snapshot tree. Null when absent/oversized. */
function readBlobFromTree(
	snapshot: WorkspaceSnapshot,
	relPath: string,
): string | null {
	try {
		const size = Number(
			git(snapshot.root, ["cat-file", "-s", `${snapshot.tree}:${relPath}`]),
		);
		if (!Number.isFinite(size) || size > MAX_SNAPSHOT_BYTES) return null;
		// No trim: blob content must be byte-exact (trailing newlines matter).
		return execFileSync("git", ["show", `${snapshot.tree}:${relPath}`], {
			cwd: snapshot.root,
			timeout: GIT_TIMEOUT_MS,
			stdio: ["ignore", "pipe", "ignore"],
			maxBuffer: 64 * 1024 * 1024,
		}).toString("utf8");
	} catch (_e: unknown) {
		return null;
	}
}

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
	} catch (_e: unknown) {
		// File doesn't exist yet — record null so rewind deletes it.
		frame.files.set(absolute, null);
	}
}

/**
 * Snapshot the working tree before a bash call. Returns null (making the
 * capture a no-op) outside git repositories or without an open frame.
 */
export function snapshotBeforeBash(cwd?: string): WorkspaceSnapshot | null {
	if (!frames.at(-1)) return null;
	return gitSnapshotTree(cwd ?? process.cwd());
}

/**
 * Diff the working tree against the pre-bash snapshot and record the pre-call
 * state of every path the command mutated (first write per path wins, same as
 * the write-tool layer).
 */
export function recordBashMutations(before: WorkspaceSnapshot | null): void {
	const frame = frames.at(-1);
	if (!frame || !before) return;
	try {
		const current = gitSnapshotTree(before.root);
		if (!current || current.tree === before.tree) return;

		// -z output: <status>\0<path>\0... Direction before→current, so "A"
		// means bash created the path (pre-state: did not exist).
		const raw = git(before.root, [
			"diff-tree",
			"-r",
			"--no-renames",
			"--name-status",
			"-z",
			before.tree,
			current.tree,
		]);
		if (!raw) return;

		const parts = raw.split("\0").filter(p => p.length > 0);
		for (let i = 0; i + 1 < parts.length; i += 2) {
			const status = parts[i];
			const relPath = parts[i + 1];
			const absolute = resolve(before.root, relPath);
			if (frame.files.has(absolute)) continue;
			if (status === "A") {
				frame.files.set(absolute, null);
			} else {
				const content = readBlobFromTree(before, relPath);
				if (content !== null) frame.files.set(absolute, content);
			}
		}
	} catch (_e: unknown) {
		// Best-effort: bash capture must never break tool execution.
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
			if (content === null) rmSync(path, { force: true });
			else writeFileSync(path, content, "utf8");
			restored++;
		} catch (_e: unknown) {
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
