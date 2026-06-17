// ── Path Utilities ─────────────────────────────────────────────────────────────
// Path resolution and CWD safety checks.
// Extracted from helpers.ts to reduce its size.

import * as path from "node:path";
import * as fs from "node:fs";

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
