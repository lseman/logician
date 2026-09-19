// ── File mention listing ────────────────────────────────────────────────────
// Lists project files for @-mention autocomplete in the TUI input bar.
// Uses @logician/log-natives' native glob() engine (pi-walker-backed) instead
// of shelling out to fd/rg; returns a plain array capped at a small limit
// since callers filter client-side.

import { loadNative } from "../../capabilities/tools/support/native-addon.ts";

const DEFAULT_LIMIT = 5000;

let cache: { cwd: string; files: string[]; expiresAt: number } | null = null;
const CACHE_TTL_MS = 15000;

/**
 * List project files under cwd for autocomplete, respecting .gitignore.
 * Results are cached briefly per cwd since the popup queries on every
 * keystroke while filtering client-side.
 */
export async function listProjectFiles(
	cwd: string,
	limit: number = DEFAULT_LIMIT,
): Promise<string[]> {
	const now = Date.now();
	if (cache && cache.cwd === cwd && cache.expiresAt > now) {
		return cache.files;
	}

	const files = await fetchFiles(cwd, limit);
	cache = { cwd, files, expiresAt: now + CACHE_TTL_MS };
	return files;
}

async function fetchFiles(cwd: string, limit: number): Promise<string[]> {
	try {
		const native = await loadNative();
		const result = await native.glob(
			{
				pattern: "**/*",
				path: cwd,
				fileType: native.FileType.File,
				hidden: true,
				gitignore: true,
				maxResults: limit,
			},
			null,
		);
		return result.matches.map(match => match.path);
	} catch {
		return [];
	}
}
