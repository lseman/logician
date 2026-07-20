import { randomUUID } from "node:crypto";
import { lstat, open, readFile, rename, stat, chmod, unlink } from "node:fs/promises";
import { basename, dirname, join } from "node:path";

export interface AtomicWriteOptions {
	/** Exact content observed before computing the replacement. */
	expectedContent?: string;
	/** Fail if another process creates the target before the rename. */
	expectedMissing?: boolean;
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
		if (!(typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT")) {
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

		if (options.expectedContent !== undefined) {
			const current = await readFile(filePath, "utf8");
			if (current !== options.expectedContent) {
				throw new Error(
					`${filePath} changed while the edit was being prepared. Read it again before editing.`,
				);
			}
		}
		if (options.expectedMissing) {
			try {
				await lstat(filePath);
				throw new Error(
					`${filePath} was created while the write was being prepared. Read it before overwriting.`,
				);
			} catch (error) {
				if (!(typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT")) {
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
