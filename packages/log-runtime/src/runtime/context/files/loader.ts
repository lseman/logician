// ── Context files loader ──────────────────────────────────────────────────────
// Loads AGENTS.md / CLAUDE.md from multiple directories and concatenates.
// Also handles SYSTEM.md override and APPEND_SYSTEM.md append.
//
// Discovery order (later context files can refine earlier ones):
//   1. ~/.logician/AGENTS.md (global)
//   2. ~/.logician/SYSTEM.md (system prompt override)
//   3. ~/.logician/APPEND_SYSTEM.md (appended to system prompt)
//   4. LOGICIAN_AGENTS_FILE entries
//   5. AGENTS.md/CLAUDE.md from root through cwd
//   6. Trusted .logician/AGENTS.md files from root through cwd
//   7. Nearest trusted .logician/SYSTEM.md
//   8. Nearest trusted .logician/APPEND_SYSTEM.md

import { existsSync, readFileSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";

const CONTEXT_FILENAMES = ["AGENTS.md", "AGENTS.MD", "CLAUDE.md"];

function findContextFile(startDir: string): string | null {
	for (const name of CONTEXT_FILENAMES) {
		const path = join(startDir, name);
		if (existsSync(path)) return path;
	}
	return null;
}

function findSystemFile(startDir: string, filename: string): string | null {
	const path = join(startDir, filename);
	return existsSync(path) ? path : null;
}

export interface ContextFile {
	path: string;
	content: string;
	source: "global" | "explicit" | "project" | "parent";
}

/**
 * Load all context files from all discovery locations.
 * Returns files in concatenation order (oldest first).
 */
export function loadContextFiles(options: {
	agentDir: string;
	cwd: string;
	loadProjectContext?: boolean;
}): {
	contextFiles: ContextFile[];
	systemFile: ContextFile | null;
	appendSystemFile: ContextFile | null;
} {
	const contextFiles: ContextFile[] = [];
	const seenContextPaths = new Set<string>();
	let systemFile: ContextFile | null = null;
	let appendSystemFile: ContextFile | null = null;
	const addContextFile = (
		filePath: string,
		source: ContextFile["source"],
	): void => {
		const resolvedPath = resolve(filePath);
		if (seenContextPaths.has(resolvedPath) || !existsSync(resolvedPath)) return;
		seenContextPaths.add(resolvedPath);
		contextFiles.push({
			path: resolvedPath,
			content: readFileSync(resolvedPath, "utf-8"),
			source,
		});
	};

	// 1. Global context file
	addContextFile(join(options.agentDir, "AGENTS.md"), "global");

	// Explicit instruction files follow the global file and precede project files.
	const explicitFiles = process.env.LOGICIAN_AGENTS_FILE;
	if (explicitFiles) {
		for (const filePath of explicitFiles.split(delimiter)) {
			if (filePath.trim()) addContextFile(filePath.trim(), "explicit");
		}
	}

	// 2. Global system files
	const globalSystem = findSystemFile(options.agentDir, "SYSTEM.md");
	if (globalSystem) {
		systemFile = {
			path: globalSystem,
			content: readFileSync(globalSystem, "utf-8"),
			source: "global",
		};
	}
	const globalAppend = findSystemFile(options.agentDir, "APPEND_SYSTEM.md");
	if (globalAppend) {
		appendSystemFile = {
			path: globalAppend,
			content: readFileSync(globalAppend, "utf-8"),
			source: "global",
		};
	}

	// 3. Walk from the filesystem root toward cwd so nearer instructions are
	// appended later and can refine broader repository instructions.
	const ancestorDirs: string[] = [];
	let currentDir = resolve(options.cwd);
	while (true) {
		ancestorDirs.push(currentDir);
		const parentDir = dirname(currentDir);
		if (parentDir === currentDir) break;
		currentDir = parentDir;
	}
	for (const ancestorDir of ancestorDirs.reverse()) {
		const contextPath = findContextFile(ancestorDir);
		if (contextPath) addContextFile(contextPath, "parent");

		if (options.loadProjectContext !== false) {
			const projectDir = join(ancestorDir, ".logician");
			const projectContextPath = findContextFile(projectDir);
			if (projectContextPath) {
				addContextFile(projectContextPath, "project");
			}

			// Nearer project system files override broader/global files.
			const projectSystem = findSystemFile(projectDir, "SYSTEM.md");
			if (projectSystem) {
				systemFile = {
					path: projectSystem,
					content: readFileSync(projectSystem, "utf-8"),
					source: "project",
				};
			}
			const projectAppend = findSystemFile(projectDir, "APPEND_SYSTEM.md");
			if (projectAppend) {
				appendSystemFile = {
					path: projectAppend,
					content: readFileSync(projectAppend, "utf-8"),
					source: "project",
				};
			}
		}
	}

	return { contextFiles, systemFile, appendSystemFile };
}

/**
 * Concatenate context files into a single string.
 */
export function concatContextFiles(files: ContextFile[]): string {
	if (files.length === 0) return "";
	return files.map(f => `<!-- ${f.path} -->\n${f.content}`).join("\n\n");
}

/**
 * Build the full context file content: system override + appended + concatenated context files.
 */
export function buildContextContent(options: {
	contextFiles: ContextFile[];
	systemFile: ContextFile | null;
	appendSystemFile: ContextFile | null;
}): string {
	const { contextFiles, systemFile, appendSystemFile } = options;

	const parts: string[] = [];

	// System override (replaces default)
	if (systemFile) {
		parts.push(
			`<!-- SYSTEM.md (${systemFile.path}) -->\n${systemFile.content}`,
		);
	}

	// Appended system content
	if (appendSystemFile) {
		parts.push(
			`<!-- APPEND_SYSTEM.md (${appendSystemFile.path}) -->\n${appendSystemFile.content}`,
		);
	}

	// Concatenated context files
	const contextContent = concatContextFiles(contextFiles);
	if (contextContent) {
		parts.push(contextContent);
	}

	return parts.join("\n\n");
}
