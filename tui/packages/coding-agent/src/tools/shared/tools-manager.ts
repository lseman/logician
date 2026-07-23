// ── Tools manager ─────────────────────────────────────────────────────────────
// Locate required CLI tools (fd, rg) and return their path.
// Returns null when not found — callers fall back gracefully.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { existsSync } from "node:fs";

const execFileAsync = promisify(execFile);

// Cached results per tool name.
const toolPathCache = new Map<string, string | null>();

// Well-known installation prefixes to probe when `which` fails.
const SEARCH_PATHS: Record<string, string[]> = {
	fd: [
		"/usr/bin/fd",
		"/usr/local/bin/fd",
		"/opt/homebrew/bin/fd",
		"/home/linuxbrew/.linuxbrew/bin/fd",
	],
	rg: [
		"/usr/bin/rg",
		"/usr/local/bin/rg",
		"/opt/homebrew/bin/rg",
		"/home/linuxbrew/.linuxbrew/bin/rg",
	],
};

/**
 * Find a CLI tool by name.
 * Checks `which <tool>` first, then probes well-known paths.
 * Returns the absolute path or null if not found.
 */
export async function getToolPath(tool: string): Promise<string | null> {
	const cached = toolPathCache.get(tool);
	if (cached !== undefined) return cached;

	// Try which/where
	try {
		const cmd = process.platform === "win32" ? "where" : "which";
		const { stdout } = await execFileAsync(cmd, [tool], { timeout: 3000 });
		const found = stdout.trim().split("\n")[0].trim();
		if (found) {
			toolPathCache.set(tool, found);
			return found;
		}
	} catch (e: unknown) {
		// not on PATH
	}

	// Probe well-known paths
	for (const candidate of SEARCH_PATHS[tool] ?? []) {
		if (existsSync(candidate)) {
			toolPathCache.set(tool, candidate);
			return candidate;
		}
	}

	toolPathCache.set(tool, null);
	return null;
}

/**
 * Ensure a tool is available. Returns its path or null.
 * Pass `required = true` to get a thrown error instead of null.
 */
export async function ensureTool(tool: "fd" | "rg", required?: false): Promise<string | null>;
export async function ensureTool(tool: "fd" | "rg", required: true): Promise<string>;
export async function ensureTool(tool: "fd" | "rg", required = false): Promise<string | null> {
	const path = await getToolPath(tool);
	if (!path && required) {
		throw new Error(
			`Required tool '${tool}' is not installed. Install it with your package manager (e.g. apt install ${tool}, brew install ${tool}).`,
		);
	}
	return path;
}
