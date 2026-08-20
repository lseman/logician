// ── Path utilities ────────────────────────────────────────────────────────
// Path resolution and CWD containment checks for the file tools. Ported from
// coding-agent's tools/path-utils.ts, rebuilt on ExecutionEnv's
// absolutePath/canonicalPath instead of raw node:fs so containment checks
// work against any FileSystem implementation. Drops the macOS
// cloud-sync-exclusion marker and read-path OS-quirk variant guessing
// (curly-quote/NFD/macOS-screenshot-filename heuristics) as out of scope for
// this core port.

import type { ExecutionEnv } from "../../env/execution-env.ts";

/** Resolve a (possibly relative) path against the environment's cwd. */
export async function resolveToCwd(
	env: ExecutionEnv,
	filePath: string,
): Promise<string> {
	const result = await env.absolutePath(filePath);
	if (!result.ok) throw new Error(`Failed to resolve path: ${filePath}`);
	return result.value;
}

/**
 * Resolve symlinks in the existing portion of a path while preserving any
 * missing suffix, so containment checks are meaningful for both reads and
 * writes to files that do not exist yet.
 */
async function canonicalizeForContainment(
	env: ExecutionEnv,
	absolutePath: string,
): Promise<string> {
	const canonical = await env.canonicalPath(absolutePath);
	if (canonical.ok) return canonical.value;
	// Path (or a suffix of it) doesn't exist yet — walk up to the nearest
	// existing ancestor, canonicalize that, then reattach the missing suffix.
	const missing: string[] = [];
	let current = absolutePath;
	for (;;) {
		const parentResult = await env.joinPath([current, ".."]);
		if (!parentResult.ok) return absolutePath;
		const parent = await env.absolutePath(parentResult.value);
		if (!parent.ok || parent.value === current) return absolutePath;
		const parentCanonical = await env.canonicalPath(parent.value);
		const segment = current.slice(parent.value.length).replace(/^[/\\]+/, "");
		missing.unshift(segment);
		if (parentCanonical.ok) {
			const joined = await env.joinPath([parentCanonical.value, ...missing]);
			return joined.ok ? joined.value : absolutePath;
		}
		current = parent.value;
	}
}

function isWithin(root: string, candidate: string): boolean {
	if (candidate === root) return true;
	const normalizedRoot = root.endsWith("/") ? root : `${root}/`;
	return candidate.startsWith(normalizedRoot);
}

/** Ensure a resolved path is inside the CWD or an allowed path. Throws if outside. */
export async function ensureInsideCwd(
	env: ExecutionEnv,
	resolvedPath: string,
	allowedPaths?: string[],
	allowAllPaths?: boolean,
): Promise<void> {
	if (allowAllPaths) return;

	const resolvedCwd = await canonicalizeForContainment(env, env.cwd);
	const resolved = await canonicalizeForContainment(env, resolvedPath);

	if (isWithin(resolvedCwd, resolved)) return;

	if (allowedPaths) {
		for (const allowedPath of allowedPaths) {
			const resolvedAllowed = await canonicalizeForContainment(
				env,
				allowedPath,
			);
			if (isWithin(resolvedAllowed, resolved)) return;
		}
	}

	throw new Error(`Path is outside CWD: ${resolvedPath} (CWD: ${resolvedCwd})`);
}

/** Resolve a path for reading. Currently a thin alias over resolveToCwd (no OS-quirk fallback variants). */
export async function resolveReadPath(
	env: ExecutionEnv,
	filePath: string,
): Promise<string> {
	return resolveToCwd(env, filePath);
}
