// ── Path Utilities ─────────────────────────────────────────────────────────────
// Path resolution and CWD safety checks.
// Extracted from helpers.ts to reduce its size.

import * as path from "node:path";
import * as fs from "node:fs";
import { execFile } from "node:child_process";

/** Resolve a (possibly relative) file path to an absolute path. */
export function resolvePath(cwd: string | undefined, filePath: string): string {
	if (path.isAbsolute(filePath)) return filePath;
	return path.resolve(cwd ?? process.cwd(), filePath);
}

/** Ensure a resolved path is inside the CWD. Throws if outside. */
export function ensureInsideCwd(
	cwd: string | undefined,
	resolvedPath: string,
): void {
	const resolvedCwd = path.resolve(cwd ?? process.cwd());
	if (!resolvedPath.startsWith(resolvedCwd)) {
		throw new Error(
			`Path is outside CWD: ${resolvedPath} (CWD: ${resolvedCwd})`,
		);
	}
}

/** Read a file as UTF-8 if it exists, null otherwise. */
export function readUtf8IfExists(filePath: string): string | null {
	try {
		return fs.readFileSync(filePath, "utf-8");
	} catch {
		return null;
	}
}

/**
 * Mark a directory so cloud sync services (iCloud, Dropbox, OneDrive) skip it.
 * - macOS: sets com.apple.metadata:com_apple_backup_excludeItem xattr + .noindex
 * - Linux: creates .noindex marker file
 * Ported from pi packages/coding-agent/src/utils/paths.ts.
 * Errors are silently swallowed — this is best-effort.
 */
export function markPathIgnoredByCloudSync(dirPath: string): void {
	try {
		// macOS xattr — prevents iCloud Drive and Time Machine backup
		if (process.platform === "darwin") {
			execFile("xattr", [
				"-w",
				"com.apple.metadata:com_apple_backup_excludeItem",
				"com.apple.backupd",
				dirPath,
			]);
			// .noindex tells Spotlight and some sync clients to skip
			const noindex = path.join(dirPath, ".noindex");
			if (!fs.existsSync(noindex)) fs.writeFileSync(noindex, "");
		} else {
			// Linux: .noindex is respected by some cloud clients (Dropbox, etc.)
			const noindex = path.join(dirPath, ".noindex");
			if (!fs.existsSync(noindex)) fs.writeFileSync(noindex, "");
		}
	} catch {
		// best-effort
	}
}

