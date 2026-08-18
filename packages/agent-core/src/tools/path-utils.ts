// ── Path Utilities ─────────────────────────────────────────────────────────────
// Path resolution and CWD safety checks.
// Extracted from helpers.ts to reduce its size.

import { execFile } from "node:child_process";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const UNICODE_SPACES = /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g;
const NARROW_NO_BREAK_SPACE = "\u202F";

export interface PathInputOptions {
	trim?: boolean;
	expandTilde?: boolean;
	homeDir?: string;
	stripAtPrefix?: boolean;
	normalizeUnicodeSpaces?: boolean;
}

export function normalizePath(
	input: string,
	options: PathInputOptions = {},
): string {
	let normalized = options.trim ? input.trim() : input;
	if (options.normalizeUnicodeSpaces) {
		normalized = normalized.replace(UNICODE_SPACES, " ");
	}
	if (options.stripAtPrefix && normalized.startsWith("@")) {
		normalized = normalized.slice(1);
	}

	if (options.expandTilde ?? true) {
		const home = options.homeDir ?? homedir();
		if (normalized === "~") return home;
		if (
			normalized.startsWith("~/") ||
			(process.platform === "win32" && normalized.startsWith("~\\"))
		) {
			return path.join(home, normalized.slice(2));
		}
	}

	if (/^file:\/\//.test(normalized)) {
		return fileURLToPath(normalized);
	}

	return normalized;
}

/** Resolve a (possibly relative) file path to an absolute path. */
export function resolvePath(cwd: string | undefined, filePath: string): string {
	const normalized = normalizePath(filePath, {
		normalizeUnicodeSpaces: true,
		stripAtPrefix: true,
	});
	if (path.isAbsolute(normalized)) return path.resolve(normalized);
	// If cwd is "/" (root) or empty, it's almost certainly a config bug —
	// fall back to process.cwd() so relative paths resolve inside the
	// actual working directory instead of the filesystem root.
	const effectiveCwd = cwd && cwd !== "/" ? cwd : process.cwd();
	return path.resolve(effectiveCwd, normalized);
}

export function resolveToCwd(filePath: string, cwd: string): string {
	return resolvePath(cwd, filePath);
}

/**
 * Resolve symlinks in the existing portion of a path while preserving any
 * missing suffix. This keeps containment checks meaningful for both reads and
 * writes to files that do not exist yet.
 */
function canonicalizeForContainment(filePath: string): string {
	const resolved = path.resolve(filePath);
	const missing: string[] = [];
	let current = resolved;

	while (true) {
		try {
			const canonical = fs.realpathSync.native(current);
			return path.resolve(canonical, ...missing.reverse());
		} catch (error: unknown) {
			const code =
				error && typeof error === "object" && "code" in error
					? String((error as { code: unknown }).code)
					: "";
			if (code !== "ENOENT" && code !== "ENOTDIR") throw error;
			const parent = path.dirname(current);
			if (parent === current) return resolved;
			missing.push(path.basename(current));
			current = parent;
		}
	}
}

function isWithin(root: string, candidate: string): boolean {
	const relative = path.relative(root, candidate);
	return (
		relative === "" ||
		(relative !== ".." &&
			!relative.startsWith(`..${path.sep}`) &&
			!path.isAbsolute(relative))
	);
}

/** Ensure a resolved path is inside the CWD or an allowed path. Throws if outside. */
export function ensureInsideCwd(
	cwd: string | undefined,
	resolvedPath: string,
	allowedPaths?: string[],
	allowAllPaths?: boolean,
): void {
	if (allowAllPaths) return;

	const resolvedCwd = canonicalizeForContainment(cwd ?? process.cwd());
	const resolved = canonicalizeForContainment(resolvedPath);

	// Check CWD first.
	if (isWithin(resolvedCwd, resolved)) return;

	// Check allowed paths.
	if (allowedPaths) {
		for (const ap of allowedPaths) {
			const resolvedAp = canonicalizeForContainment(ap);
			if (isWithin(resolvedAp, resolved)) return;
		}
	}

	throw new Error(`Path is outside CWD: ${resolvedPath} (CWD: ${resolvedCwd})`);
}

function fileExists(filePath: string): boolean {
	try {
		fs.accessSync(filePath, fs.constants.F_OK);
		return true;
	} catch (_e: unknown) {
		return false;
	}
}

function tryMacOSScreenshotPath(filePath: string): string {
	return filePath.replace(/ (AM|PM)\./gi, `${NARROW_NO_BREAK_SPACE}$1.`);
}

function tryNFDVariant(filePath: string): string {
	return filePath.normalize("NFD");
}

function tryCurlyQuoteVariant(filePath: string): string {
	return filePath.replace(/'/g, "\u2019");
}

function readPathVariants(resolved: string): string[] {
	const variants = new Set<string>();
	const transforms = [
		tryMacOSScreenshotPath,
		tryNFDVariant,
		tryCurlyQuoteVariant,
	];
	for (const first of transforms) {
		const firstVariant = first(resolved);
		variants.add(firstVariant);
		for (const second of transforms) {
			variants.add(second(firstVariant));
		}
	}
	return [...variants];
}

export function resolveReadPath(filePath: string, cwd: string): string {
	const resolved = resolveToCwd(filePath, cwd);
	if (fileExists(resolved)) return resolved;

	for (const variant of readPathVariants(resolved)) {
		if (variant !== resolved && fileExists(variant)) return variant;
	}

	return resolved;
}

/** Read a file as UTF-8 if it exists, null otherwise. */
export function readUtf8IfExists(filePath: string): string | null {
	try {
		return fs.readFileSync(filePath, "utf-8");
	} catch (_e: unknown) {
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
	} catch (_e: unknown) {
		// best-effort
	}
}
