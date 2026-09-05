// ── TTSR Rule Loader ─────────────────────────────────────────────────────────
// Loads TTSR rules from .logician/rules/*.md files with YAML frontmatter.
import fs from "node:fs/promises";
import path from "node:path";
import type { Dirent } from "node:fs";
import type { TtsrRule, TtsrScope } from "@logician/log-core";


// ── YAML Frontmatter Parser ─────────────────────────────────────────────────

interface FrontmatterEntry {
	frontmatter: Record<string, string | string[]>;
	content: string;
}

/** Extract YAML frontmatter from a markdown file string. */
export function parseFrontmatter(content: string): FrontmatterEntry | null {
	const frontmatterRegex = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/;
	const match = content.match(frontmatterRegex);
	if (!match) return null;

	const fmRaw = match[1];
	const body = match[2];
	const fm: Record<string, string | string[]> = {};

	for (const line of fmRaw.split("\n")) {
		const colonIdx = line.indexOf(":");
		if (colonIdx === -1) continue;

		const key = line.slice(0, colonIdx).trim();
		let value = line.slice(colonIdx + 1).trim();

		if (value.startsWith("[") && value.endsWith("]")) {
			const items = value
				.slice(1, -1)
				.split(",")
				.map(s => s.trim().replace(/^["']|["']$/g, ""))
				.filter(Boolean);
			fm[key] = items;
		} else {
			fm[key] = value.replace(/^["']|["']$/g, "");
		}
	}

	return { frontmatter: fm, content: body };
}

/** Validate a scope string against known TTSR scopes. */
function isValidScope(value: string): value is TtsrScope {
	if (value === "text" || value === "thinking" || value === "tool") return true;
	if (value.startsWith("tool:")) return true;
	return false;
}

/** Convert frontmatter to a TtsrRule. */
export function frontmatterToRule(fm: Record<string, string | string[]>, filePath: string): TtsrRule | null {
	const name = typeof fm.name === "string" ? fm.name : null;
	if (!name) return null;

	const description = typeof fm.description === "string" ? fm.description : "";
	const content = typeof fm.content === "string" ? fm.content : "";
	const conditions = Array.isArray(fm.conditions)
		? (fm.conditions as string[])
		: typeof fm.conditions === "string"
			? [fm.conditions]
			: [];

	const rawScope = Array.isArray(fm.scope)
		? (fm.scope as string[])
		: typeof fm.scope === "string"
			? [fm.scope]
			: ["text"];
	const scope = rawScope.filter((s: string): s is TtsrScope => isValidScope(s));
	if (scope.length === 0) scope.push("text" as TtsrScope);

	const globs = Array.isArray(fm.globs)
		? (fm.globs as string[])
		: undefined;

	const astConditions = Array.isArray(fm.astConditions)
		? (fm.astConditions as string[])
		: undefined;

	const interruptMode = (typeof fm.interruptMode === "string"
		? fm.interruptMode
		: "always") as TtsrRule["interruptMode"];

	if (!content && conditions.length === 0) return null;

	return {
		name,
		path: filePath,
		description,
		content,
		conditions,
		astConditions,
		scope,
		globs,
		interruptMode,
	};
}

export interface LoadedRule {
	rule: TtsrRule;
	warnings: string[];
}

/**
 * Load TTSR rules from a directory.
 * Scans for *.md files, parses YAML frontmatter, converts to TtsrRule.
 */
export async function loadRules(dir: string): Promise<LoadedRule[]> {
	const rules: LoadedRule[] = [];
	let entries: Dirent[];
	try {
		entries = await fs.readdir(dir, { withFileTypes: true });
	} catch {
		// Directory doesn't exist or isn't readable — no rules
		return [];
	}

	for (const entry of entries) {
		if (!entry.isFile()) continue;
		if (!entry.name.endsWith(".md")) continue;
		if (entry.name.startsWith(".")) continue;

		const filePath = path.join(dir, entry.name);
		let content: string;
		try {
			content = await fs.readFile(filePath, "utf-8");
		} catch {
			rules.push({
				rule: {
					name: entry.name.replace(".md", ""),
					path: filePath,
					description: "",
					content: "",
					conditions: [],
					interruptMode: "never",
					scope: [],
				},
				warnings: [`Failed to read file: ${filePath}`],
			});
			continue;
		}

		const parsed = parseFrontmatter(content);
		if (!parsed) continue;

		const rule = frontmatterToRule(parsed.frontmatter, filePath);
		if (!rule) {
			rules.push({
				rule: {
					name: entry.name.replace(".md", ""),
					path: filePath,
					description: "",
					content: "",
					conditions: [],
					interruptMode: "never",
					scope: [],
				},
				warnings: ["Missing name field or empty content/conditions"],
			});
			continue;
		}

		rule.builtin = false;
		rules.push({ rule, warnings: [] });
	}

	return rules;
}
