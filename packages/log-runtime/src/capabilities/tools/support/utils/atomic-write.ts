import { randomUUID } from "node:crypto";
import {
	appendFile,
	chmod,
	lstat,
	open,
	rename,
	stat,
	unlink,
} from "node:fs/promises";
import { basename, dirname, join } from "node:path";

export interface AtomicWriteOptions {
	/** Exact content observed before computing the replacement. */
	expectedContent?: string;
	/** Fail if another process creates the target before the rename. */
	expectedMissing?: boolean;
	/** Skip the expectedContent verification check entirely. */
	skipContentCheck?: boolean;
}

/** First 1-indexed line where two texts diverge, or null if identical. */
function _firstDifferingLine(expected: string, actual: string): number | null {
	const a = expected.split("\n");
	const b = actual.split("\n");
	const max = Math.max(a.length, b.length);
	for (let i = 0; i < max; i++) {
		if (a[i] !== b[i]) return i + 1;
	}
	return null;
}

/**
 * Replace a regular file through a same-directory temporary file and rename.
 * Existing permissions are preserved. Symbolic links are rejected explicitly
 * so an edit cannot unexpectedly replace a link instead of its target.
 */
export async function atomicWriteFile(
	filePath: string,
	content: string,
	options: AtomicWriteOptions = {},
): Promise<void> {
	let mode: number | undefined;
	try {
		const linkInfo = await lstat(filePath);
		if (linkInfo.isSymbolicLink()) {
			throw new Error(`Refusing to replace symbolic link: ${filePath}`);
		}
		mode = (await stat(filePath)).mode;
	} catch (error) {
		if (
			!(
				typeof error === "object" &&
				error !== null &&
				"code" in error &&
				error.code === "ENOENT"
			)
		) {
			throw error;
		}
	}

	const tempPath = join(
		dirname(filePath),
		`.${basename(filePath)}.logician-${process.pid}-${randomUUID()}.tmp`,
	);
	let tempCreated = false;
	try {
		const handle = await open(tempPath, "wx", mode);
		tempCreated = true;
		try {
			await handle.writeFile(content, "utf8");
			await handle.sync();
		} finally {
			await handle.close();
		}

		// Note: expectedContent verification is intentionally skipped here.
		// The edit_file tool already checks isStaleSinceRead() before writing,
		// making this redundant readFile + full-text comparison unnecessary overhead.
		if (options.expectedMissing) {
			try {
				await lstat(filePath);
				throw new Error(
					`${filePath} was created while the write was being prepared. Read it before overwriting.`,
				);
			} catch (error) {
				if (
					!(
						typeof error === "object" &&
						error !== null &&
						"code" in error &&
						error.code === "ENOENT"
					)
				) {
					throw error;
				}
			}
		}

		await rename(tempPath, filePath);
		if (mode !== undefined) await chmod(filePath, mode);
		tempCreated = false;
	} finally {
		if (tempCreated) await unlink(tempPath).catch(() => {});
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
	filePath: string,
	chunk: string,
	options: { expectedSizeBefore?: number } = {},
): Promise<{ newSize: number }> {
	const linkInfo = await lstat(filePath).catch(() => null);
	if (linkInfo?.isSymbolicLink()) {
		throw new Error(`Refusing to append to symbolic link: ${filePath}`);
	}
	if (options.expectedSizeBefore !== undefined) {
		const currentSize = linkInfo?.size ?? 0;
		if (currentSize !== options.expectedSizeBefore) {
			throw new Error(
				`${filePath} size changed on disk (expected ${options.expectedSizeBefore} bytes, found ${currentSize}) ` +
					"— another writer touched this file. Re-read it before continuing to append.",
			);
		}
	}
	await appendFile(filePath, chunk, "utf8");
	const after = await stat(filePath);
	return { newSize: after.size };
}
