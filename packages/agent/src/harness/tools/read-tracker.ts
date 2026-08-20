// ── Read tracker ─────────────────────────────────────────────────────────
// Records when the model last read each file (by canonical path + mtime), so
// the edit/write tools can detect a file that changed underneath them since
// it was read and refuse to clobber the change. Ported from coding-agent's
// tools/read-tracker.ts, adapted to go through ExecutionEnv instead of raw
// node:fs so it works against any FileSystem implementation.

import type { ExecutionEnv } from "../../env/execution-env.ts";

interface ReadSnapshot {
	mtimeMs: number;
	size: number;
	sha256: string;
}

const lastReadSnapshot = new Map<string, ReadSnapshot>();

async function sha256Hex(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<string | undefined> {
	const bytes = await env.readBinaryFile(absolutePath);
	if (!bytes.ok) return undefined;
	const digest = await crypto.subtle.digest(
		"SHA-256",
		bytes.value.buffer as ArrayBuffer,
	);
	return Array.from(new Uint8Array(digest))
		.map(b => b.toString(16).padStart(2, "0"))
		.join("");
}

async function keyFor(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<string> {
	const canonical = await env.canonicalPath(absolutePath);
	return canonical.ok ? canonical.value : absolutePath;
}

/** Record that the model just read this file at its current mtime. */
export async function recordRead(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<void> {
	const info = await env.fileInfo(absolutePath);
	if (!info.ok) return;
	const sha256 = await sha256Hex(env, absolutePath);
	if (sha256 === undefined) return;
	const key = await keyFor(env, absolutePath);
	lastReadSnapshot.set(key, {
		mtimeMs: info.value.mtimeMs,
		size: info.value.size,
		sha256,
	});
}

/** Returns true if the model has read (or written) this file this session. */
export async function hasBeenRead(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<boolean> {
	return lastReadSnapshot.has(await keyFor(env, absolutePath));
}

/**
 * Returns true if the file changed on disk since it was last read by the model.
 * Returns false when the file was never read (no read-before-edit requirement)
 * or when its mtime still matches the recorded read.
 */
export async function isStaleSinceRead(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<boolean> {
	const key = await keyFor(env, absolutePath);
	const recorded = lastReadSnapshot.get(key);
	if (recorded === undefined) return false;
	const info = await env.fileInfo(absolutePath);
	if (!info.ok) return false;
	if (
		info.value.mtimeMs !== recorded.mtimeMs ||
		info.value.size !== recorded.size
	)
		return true;
	const sha256 = await sha256Hex(env, absolutePath);
	if (sha256 === undefined) return false;
	return sha256 !== recorded.sha256;
}

/** Refresh the recorded mtime after a tool mutates the file itself. */
export async function refreshAfterWrite(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<void> {
	await recordRead(env, absolutePath);
}
