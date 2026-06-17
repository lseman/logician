// ── Context files loader ──────────────────────────────────────────────────────
// Loads AGENTS.md / CLAUDE.md from multiple directories and concatenates.
// Also handles SYSTEM.md override and APPEND_SYSTEM.md append.
//
// Discovery order (last wins, all concatenated):
//   1. ~/.logician/AGENTS.md (global)
//   2. ~/.logician/SYSTEM.md (system prompt override)
//   3. ~/.logician/APPEND_SYSTEM.md (appended to system prompt)
//   4. <cwd>/.logician/AGENTS.md (project-local)
//   5. <cwd>/.logician/SYSTEM.md
//   6. <cwd>/.logician/APPEND_SYSTEM.md
//   7. Parent directories walking up to root

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

const CONTEXT_FILENAMES = ["AGENTS.md", "CLAUDE.md"];
const SYSTEM_FILENAMES = ["SYSTEM.md", "APPEND_SYSTEM.md"];

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

interface ContextFile {
	path: string;
	content: string;
	source: "global" | "project" | "parent";
}

/**
 * Load all context files from all discovery locations.
 * Returns files in concatenation order (oldest first).
 */
export function loadContextFiles(options: {
	agentDir: string;
	cwd: string;
}): {
	contextFiles: ContextFile[];
	systemFile: ContextFile | null;
	appendSystemFile: ContextFile | null;
} {
	const contextFiles: ContextFile[] = [];
	let systemFile: ContextFile | null = null;
	let appendSystemFile: ContextFile | null = null;

	// 1. Global context file
	const globalPath = join(options.agentDir, "AGENTS.md");
	if (existsSync(globalPath)) {
		contextFiles.push({ path: globalPath, content: readFileSync(globalPath, "utf-8"), source: "global" });
	}

	// 2. Global system files
	const globalSystem = findSystemFile(options.agentDir, "SYSTEM.md");
	if (globalSystem) {
		systemFile = { path: globalSystem, content: readFileSync(globalSystem, "utf-8"), source: "global" };
	}
	const globalAppend = findSystemFile(options.agentDir, "APPEND_SYSTEM.md");
	if (globalAppend) {
		appendSystemFile = { path: globalAppend, content: readFileSync(globalAppend, "utf-8"), source: "global" };
	}

	// 3. Project context file
	const projectDir = join(options.cwd, ".logician");
	if (existsSync(projectDir)) {
		const projectContext = findContextFile(projectDir);
		if (projectContext) {
			contextFiles.push({ path: projectContext, content: readFileSync(projectContext, "utf-8"), source: "project" });
		}

		// Project system files (override global)
		const projectSystem = findSystemFile(projectDir, "SYSTEM.md");
		if (projectSystem) {
			systemFile = { path: projectSystem, content: readFileSync(projectSystem, "utf-8"), source: "project" };
		}
		const projectAppend = findSystemFile(projectDir, "APPEND_SYSTEM.md");
		if (projectAppend) {
			appendSystemFile = { path: projectAppend, content: readFileSync(projectAppend, "utf-8"), source: "project" };
		}
	}

	// 4. Walk up parent directories for context files
	let currentDir = dirname(options.cwd);
	const root = "/";
	while (currentDir !== root) {
		const parentContext = findContextFile(currentDir);
		if (parentContext) {
			contextFiles.push({ path: parentContext, content: readFileSync(parentContext, "utf-8"), source: "parent" });
			// Stop at first parent match (like pi)
			break;
		}
		currentDir = dirname(currentDir);
	}

	return { contextFiles, systemFile, appendSystemFile };
}

/**
 * Concatenate context files into a single string.
 */
export function concatContextFiles(files: ContextFile[]): string {
	if (files.length === 0) return "";
	return files.map((f) => `<!-- ${f.path} -->\n${f.content}`).join("\n\n");
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
		parts.push(`<!-- SYSTEM.md (${systemFile.path}) -->\n${systemFile.content}`);
	}

	// Appended system content
	if (appendSystemFile) {
		parts.push(`<!-- APPEND_SYSTEM.md (${appendSystemFile.path}) -->\n${appendSystemFile.content}`);
	}

	// Concatenated context files
	const contextContent = concatContextFiles(contextFiles);
	if (contextContent) {
		parts.push(contextContent);
	}

	return parts.join("\n\n");
}
