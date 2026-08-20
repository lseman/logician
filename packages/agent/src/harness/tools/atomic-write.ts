// ── Atomic file write ─────────────────────────────────────────────────────
// Replace a file through a same-directory temporary file and rename, with an
// optional precondition check against content the caller observed before
// computing the replacement — guards against a concurrent writer racing the
// same path between read and write. Ported from coding-agent's
// tools/shared/atomic-write.ts, rebuilt on ExecutionEnv (writeFile +
// renameFile) instead of raw node:fs. Symlink-rejection and file-mode
// preservation are dropped: ExecutionEnv doesn't expose lstat/chmod, and a
// temp-file-plus-rename already can't silently follow a symlink the way an
// in-place write could.

import type { ExecutionEnv } from "../../env/execution-env.ts";

export interface AtomicWriteOptions {
	/** Exact content observed before computing the replacement. */
	expectedContent?: string;
	/** Fail if another process creates the target before the rename. */
	expectedMissing?: boolean;
}

/** First 1-indexed line where two texts diverge, or null if identical. */
function firstDifferingLine(expected: string, actual: string): number | null {
	const a = expected.split("\n");
	const b = actual.split("\n");
	const max = Math.max(a.length, b.length);
	for (let i = 0; i < max; i++) {
		if (a[i] !== b[i]) return i + 1;
	}
	return null;
}

/** Replace a file through a same-directory temporary file and rename. */
export async function atomicWriteFile(
	env: ExecutionEnv,
	filePath: string,
	content: string,
	options: AtomicWriteOptions = {},
): Promise<void> {
	const tempPath = `${filePath}.tmp-${crypto.randomUUID()}`;
	try {
		const write = await env.writeFile(tempPath, content);
		if (!write.ok)
			throw new Error(
				`Failed to stage write for ${filePath}: ${write.error.message}`,
			);

		if (options.expectedContent !== undefined) {
			const current = await env.readTextFile(filePath);
			if (current.ok && current.value !== options.expectedContent) {
				const line = firstDifferingLine(options.expectedContent, current.value);
				const where = line !== null ? ` First difference at line ${line}.` : "";
				throw new Error(
					`${filePath} changed on disk after it was read but before this write landed — likely another edit or an external process ran concurrently.${where} Read it again before editing.`,
				);
			}
		}
		if (options.expectedMissing) {
			const exists = await env.exists(filePath);
			if (exists.ok && exists.value) {
				throw new Error(
					`${filePath} was created while the write was being prepared. Read it before overwriting.`,
				);
			}
		}

		const rename = await env.renameFile(tempPath, filePath);
		if (!rename.ok)
			throw new Error(
				`Failed to publish write for ${filePath}: ${rename.error.message}`,
			);
	} catch (error) {
		await env.remove(tempPath, { force: true });
		throw error;
	}
}

/**
 * Append content to a file, creating it (and parent directories) if missing.
 * Not atomic across the whole file the way atomicWriteFile is — intended for
 * incrementally streaming a large file across multiple tool calls, each call
 * appending the byte-length it actually wrote to `expectedSizeBefore` so a
 * concurrent writer racing the same path is still detected.
 */
export async function appendToFile(
	env: ExecutionEnv,
	filePath: string,
	chunk: string,
	options: { expectedSizeBefore?: number } = {},
): Promise<{ newSize: number }> {
	if (options.expectedSizeBefore !== undefined) {
		const info = await env.fileInfo(filePath);
		const currentSize = info.ok ? info.value.size : 0;
		if (currentSize !== options.expectedSizeBefore) {
			throw new Error(
				`${filePath} size changed on disk (expected ${options.expectedSizeBefore} bytes, found ${currentSize}) — another writer touched this file. Re-read it before continuing to append.`,
			);
		}
	}
	const append = await env.appendFile(filePath, chunk);
	if (!append.ok)
		throw new Error(`Failed to append to ${filePath}: ${append.error.message}`);
	const after = await env.fileInfo(filePath);
	return { newSize: after.ok ? after.value.size : 0 };
}
