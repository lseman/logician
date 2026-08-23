/**
 * Session file path resolution for EoH.
 *
 * All EoH session files live under a single `.eoh/` subfolder
 * (one folder to preserve across reverts, gitignore, and clean up).
 */

import * as fs from "node:fs";
import * as path from "node:path";

export const EOH_DIR = ".eoh";

export type SessionFileKind =
	| "log"
	| "problem"
	| "config"
	| "prompt"
	| "check"
	| "evaluate";

const SESSION_FILE_NAMES: Record<SessionFileKind, string> = {
	log: "log.jsonl",
	problem: "problem.json",
	config: "config.json",
	prompt: "prompt.md",
	check: "check.sh",
	evaluate: "evaluate.sh",
};

function sessionPath(dir: string, kind: SessionFileKind): string {
	return path.join(dir, EOH_DIR, SESSION_FILE_NAMES[kind]);
}

function currentLayoutExists(dir: string): boolean {
	for (const kind of Object.keys(SESSION_FILE_NAMES) as SessionFileKind[]) {
		if (fs.existsSync(sessionPath(dir, kind))) return true;
	}
	return false;
}

/** Effective path for a session file. */
export function sessionFilePath(dir: string, kind: SessionFileKind): string {
	const basePath = sessionPath(dir, kind);
	if (currentLayoutExists(dir)) return basePath;
	// No legacy flat files for EoH — always use .eoh/
	return basePath;
}

/** Ensure the parent directory for a session file exists before writing. */
export function ensureParentDir(filePath: string): void {
	fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

/** Get the .eoh directory path for a working directory. */
export function eohDir(workDir: string): string {
	return path.join(workDir, EOH_DIR);
}
