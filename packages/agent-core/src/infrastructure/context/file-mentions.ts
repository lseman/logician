// ── File mention listing ────────────────────────────────────────────────────
// Lists project files for @-mention autocomplete in the TUI input bar.
// Uses fd (falls back to rg --files); same tools as the find tool, but returns
// a plain array capped at a small limit since callers filter client-side.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { ensureTool } from "../tools/external-tools.ts";

const execFileAsync = promisify(execFile);

const DEFAULT_LIMIT = 5000;
const EXEC_TIMEOUT_MS = 3000;

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
	const fdPath = await ensureTool("fd");
	if (fdPath) {
		try {
			const { stdout } = await execFileAsync(
				fdPath,
				[
					"--type",
					"f",
					"--color=never",
					"--hidden",
					"--no-require-git",
					"--max-results",
					String(limit),
				],
				{ cwd, timeout: EXEC_TIMEOUT_MS, maxBuffer: 4 * 1024 * 1024 },
			);
			return stdout.split("\n").filter(Boolean);
		} catch {
			// fall through to rg
		}
	}

	const rgPath = await ensureTool("rg");
	if (rgPath) {
		try {
			const { stdout } = await execFileAsync(rgPath, ["--files", "--hidden"], {
				cwd,
				timeout: EXEC_TIMEOUT_MS,
				maxBuffer: 4 * 1024 * 1024,
			});
			return stdout.split("\n").filter(Boolean).slice(0, limit);
		} catch {
			return [];
		}
	}

	return [];
}
