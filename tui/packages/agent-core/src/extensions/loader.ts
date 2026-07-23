// ── Extension loader ──────────────────────────────────────────────────────────
// Discovers .ts and .js files in configured directories, loads them as modules,
// and returns the extension definitions with any diagnostics.

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import ignore from "ignore";
import type { Diagnostic, ExtensionDefinition, LoadExtensionsResult } from "./types.ts";

function addIgnoreRules(dir: string, matcher: ReturnType<typeof ignore>, rootDir: string): void {
	const relativeDir = relative(rootDir, dir);
	const prefix = relativeDir ? `${relativeDir}/` : "";

	for (const filename of [".gitignore", ".ignore"]) {
		const ignorePath = join(dir, filename);
		if (!existsSync(ignorePath)) continue;
		try {
			const content = readFileSync(ignorePath, "utf-8");
			const patterns = content
				.split(/\r?\n/)
				.map((line) => {
					const trimmed = line.trim();
					if (!trimmed || trimmed.startsWith("#")) return null;
					let pattern = trimmed.startsWith("!") ? trimmed.slice(1) : trimmed;
					pattern = pattern.startsWith("/") ? pattern.slice(1) : pattern;
					const prefixed = prefix ? `${prefix}${pattern}` : pattern;
					return trimmed.startsWith("!") ? `!${prefixed}` : prefixed;
				})
				.filter((p): p is string => Boolean(p));
			if (patterns.length > 0) matcher.add(patterns);
		} catch (e: unknown) {
			// ignore parse errors
		}
	}
}

function relative(from: string, to: string): string {
	// Simplified relative path computation
	if (to.startsWith(from)) {
		const rest = to.slice(from.length).replace(/^\/+/, "");
		return rest || ".";
	}
	return to;
}

function discoverFiles(dir: string, rootDir: string, matcher: ReturnType<typeof ignore>, source: "user" | "project" | "path"): Array<{ path: string; source: "user" | "project" | "path" }> {
	const results: Array<{ path: string; source: "user" | "project" | "path" }> = [];

	if (!existsSync(dir) || !statSync(dir).isDirectory()) {
		return results;
	}

	const entries = readdirSync(dir, { withFileTypes: true });
	for (const entry of entries) {
		if (entry.name.startsWith(".")) continue;
		const fullPath = join(dir, entry.name);
		const relPath = relative(rootDir, fullPath);

		if (matcher.ignores(relPath)) continue;

		let isFile = entry.isFile();
		let isDir = entry.isDirectory();
		if (entry.isSymbolicLink()) {
			try {
				const s = statSync(fullPath);
				isFile = s.isFile();
				isDir = s.isDirectory();
			} catch (e: unknown) {
				continue;
			}
		}

		if (isDir) {
			results.push(...discoverFiles(fullPath, rootDir, matcher, source));
		} else if (isFile && /\.(ts|js|mjs)$/.test(entry.name)) {
			results.push({ path: fullPath, source });
		}
	}

	return results;
}

export function loadExtensionsFromDir(
	dir: string,
	source: "user" | "project" | "path" = "path",
): LoadExtensionsResult {
	const definitions: ExtensionDefinition[] = [];
	const diagnostics: Diagnostic[] = [];
	const realPaths = new Set<string>();

	if (!existsSync(dir) || !statSync(dir).isDirectory()) {
		return { extensions: definitions, diagnostics };
	}

	const matcher = createIgnore();
	addIgnoreRules(dir, matcher, dir);

	const files = discoverFiles(dir, dir, matcher, source);

	for (const file of files) {
		const realPath = resolve(file.path);
		if (realPaths.has(realPath)) continue;
		realPaths.add(realPath);

		const name = basename(file.path, extname(file.path));
		definitions.push({
			path: file.path,
			name,
			source: file.source,
		});
	}

	return { extensions: definitions, diagnostics };
}

function extname(path: string): string {
	const idx = path.lastIndexOf(".");
	return idx > 0 ? path.slice(idx) : "";
}

function createIgnore(): ReturnType<typeof ignore> {
	return ignore();
}

export function loadExtensions(options: {
	userDir?: string;
	projectDir?: string;
	agentDir?: string;
	explicitPaths?: string[];
}): LoadExtensionsResult {
	const { userDir, projectDir, explicitPaths } = options;
	const allDefinitions: ExtensionDefinition[] = [];
	const allDiagnostics: Diagnostic[] = [];

	// Load from user (global) extensions directory
	if (userDir) {
		const result = loadExtensionsFromDir(userDir, "user");
		allDefinitions.push(...result.extensions);
		allDiagnostics.push(...result.diagnostics);
	}

	// Load from project extensions directory
	if (projectDir) {
		const projectExtDir = join(projectDir, ".logician", "extensions");
		if (existsSync(projectExtDir)) {
			const result = loadExtensionsFromDir(projectExtDir, "project");
			allDefinitions.push(...result.extensions);
			allDiagnostics.push(...result.diagnostics);
		}
	}

	// Load from explicit paths
	if (explicitPaths) {
		for (const p of explicitPaths) {
			const resolved = resolve(p);
			if (!existsSync(resolved)) {
				allDiagnostics.push({ type: "warning", message: `extension path does not exist: ${p}`, path: p });
				continue;
			}
			const result = loadExtensionsFromDir(resolved, "path");
			allDefinitions.push(...result.extensions);
			allDiagnostics.push(...result.diagnostics);
		}
	}

	return { extensions: allDefinitions, diagnostics: allDiagnostics };
}
