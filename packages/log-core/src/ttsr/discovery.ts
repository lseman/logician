// ── TTSR Rule Discovery ───────────────────────────────────────────────────────
// Discovers TTSR rules from external config formats (Cursor, Cline, Windsurf,
// GitHub Copilot, .agent/ dirs) and normalizes them into TtsrRule objects.
//
// Priority-sorted dedup: native > cursor > windsurf > cline > github > agents
// First-wins semantics for duplicate rule names.

import * as fs from "node:fs/promises";
import { isTtsrScope, type TtsrRule } from "../types/ttsr.ts";

// ── Discovery sources ─────────────────────────────────────────────────────────

interface DiscoverySource {
	name: string;
	priority: number;
	load: (cwd: string) => Promise<TtsrRule[]>;
}

// ── Frontmatter parsing ───────────────────────────────────────────────────────

function parseFrontmatter(content: string): {
	metadata: Record<string, unknown>;
	body: string;
} {
	const trimmed = content.trimStart();
	if (!trimmed.startsWith("---")) {
		return { metadata: {}, body: trimmed };
	}

	const endMarker = trimmed.indexOf("\n---", 3);
	if (endMarker === -1) {
		return { metadata: {}, body: trimmed };
	}

	const fmText = trimmed.slice(3, endMarker).trim();
	const body = trimmed.slice(endMarker + 4).trimStart();

	// Simple YAML parsing (fallback to line-by-line if full parse fails)
	const metadata: Record<string, unknown> = {};
	try {
		const lines = fmText.split("\n");
		for (const line of lines) {
			const colonIndex = line.indexOf(":");
			if (colonIndex === -1) continue;
			const key = line.slice(0, colonIndex).trim();
			const value = line.slice(colonIndex + 1).trim();

			// Parse array values: [item1, item2]
			if (value.startsWith("[") && value.endsWith("]")) {
				const inner = value.slice(1, -1);
				metadata[key] = inner
					.split(",")
					.map((s: string) => s.trim().replace(/^["']|["']$/g, ""))
					.filter((s: string) => s.length > 0);
			}
			// Parse boolean
			else if (value === "true") {
				metadata[key] = true;
			} else if (value === "false") {
				metadata[key] = false;
			}
			// Parse number
			else if (!isNaN(Number(value))) {
				metadata[key] = Number(value);
			}
			// Parse string (remove quotes)
			else {
				metadata[key] = value.replace(/^["']|["']$/g, "");
			}
		}
	} catch {
		// Fallback: empty metadata
	}

	return { metadata, body };
}

// ── Rule normalization ────────────────────────────────────────────────────────

function normalizeRule(
	source: string,
	name: string,
	content: string,
	metadata: Record<string, unknown>,
): TtsrRule | null {
	// Parse scope
	const rawScope = metadata.scope;
	const scope: TtsrRule["scope"] = [];
	const scopeTokens: unknown[] =
		typeof rawScope === "string"
			? rawScope.split(",").map((s: string) => s.trim())
			: Array.isArray(rawScope)
				? rawScope
				: [];
	for (const token of scopeTokens) {
		if (typeof token === "string" && isTtsrScope(token)) scope.push(token);
	}
	if (scope.length === 0) scope.push("text", "tool");

	// Parse interrupt mode
	const rawInterrupt = metadata.interruptMode;
	const interruptMode: TtsrRule["interruptMode"] =
		typeof rawInterrupt === "string"
			? (rawInterrupt.toLowerCase() as TtsrRule["interruptMode"])
			: "always";

	// Parse conditions (legacy ttsr_trigger support)
	const conditions: string[] = [];
	const rawConditions = (metadata.condition ?? metadata.ttsr_trigger) as
		| string
		| string[]
		| undefined;
	if (typeof rawConditions === "string") {
		conditions.push(rawConditions);
	} else if (Array.isArray(rawConditions)) {
		conditions.push(...rawConditions);
	}

	// Parse astConditions
	const rawAstConditions = metadata.astConditions as
		| string
		| string[]
		| undefined;
	const astConditions: string[] = [];
	if (typeof rawAstConditions === "string") {
		astConditions.push(rawAstConditions);
	} else if (Array.isArray(rawAstConditions)) {
		astConditions.push(...rawAstConditions);
	}

	// Parse question (for judged rules)
	const question =
		typeof metadata.question === "string" ? metadata.question : undefined;

	// Parse agents
	const rawAgents = metadata.agents as string[] | undefined;
	const agents = Array.isArray(rawAgents) ? rawAgents : undefined;

	// Parse globs
	const rawGlobs = metadata.globs as string[] | undefined;
	const globs = Array.isArray(rawGlobs) ? rawGlobs : undefined;

	if (conditions.length === 0 && astConditions.length === 0 && !question) {
		return null; // No trigger conditions
	}

	const rule: TtsrRule = {
		name,
		path: source,
		description: (metadata.description as string) ?? `Discovered rule: ${name}`,
		content,
		conditions,
		...(astConditions.length > 0 && { astConditions }),
		...(question && { question }),
		...(agents && { agents }),
		scope,
		...(globs && { globs }),
		interruptMode,
	};

	return rule;
}

// ── Source loaders ────────────────────────────────────────────────────────────

async function loadNativeRules(cwd: string): Promise<TtsrRule[]> {
	const rules: TtsrRule[] = [];
	const home = process.env.HOME || process.env.USERPROFILE;

	// Project rules first (first-wins dedup), then user-level rules; .omp/rules
	// stays readable for rules shared with oh-my-pi.
	const dirs = [
		`${cwd}/.logician/rules`,
		`${cwd}/.omp/rules`,
		...(home ? [`${home}/.logician/rules`] : []),
	];
	for (const dir of dirs) {
		try {
			const entries = await fs.readdir(dir);
			for (const entry of entries) {
				if (!entry.endsWith(".md") && !entry.endsWith(".mdc")) continue;
				const filePath = `${dir}/${entry}`;
				const content = await fs.readFile(filePath, "utf-8");
				const { metadata, body } = parseFrontmatter(content);
				const rule = normalizeRule(
					filePath,
					entry.replace(/\.(md|mdc)$/, ""),
					body,
					metadata,
				);
				if (rule) rules.push(rule);
			}
		} catch {
			// Directory doesn't exist
		}
	}

	// Try sticky RULES.md
	try {
		const nearestOmp = await findNearestOmpDir(cwd);
		if (nearestOmp) {
			const rulesPath = `${nearestOmp}/RULES.md`;
			const content = await fs.readFile(rulesPath, "utf-8");
			const { metadata, body } = parseFrontmatter(content);
			const rule = normalizeRule(rulesPath, "RULES", body, {
				...metadata,
				alwaysApply: true,
			});
			if (rule) rules.push(rule);
		}
	} catch {
		// No sticky rules
	}

	return rules;
}

async function loadCursorRules(cwd: string): Promise<TtsrRule[]> {
	const rules: TtsrRule[] = [];
	const dirs = [
		`${cwd}/.cursor/rules`,
		`${process.env.HOME || process.env.USERPROFILE}/.cursor/rules`,
	];

	for (const dir of dirs) {
		try {
			const entries = await fs.readdir(dir);
			for (const entry of entries) {
				if (!entry.endsWith(".md") && !entry.endsWith(".mdc")) continue;
				const filePath = `${dir}/${entry}`;
				const content = await fs.readFile(filePath, "utf-8");
				const { metadata, body } = parseFrontmatter(content);
				const rule = normalizeRule(
					filePath,
					entry.replace(/\.(md|mdc)$/, ""),
					body,
					metadata,
				);
				if (rule) rules.push(rule);
			}
		} catch {
			// Directory doesn't exist
		}
	}

	return rules;
}

async function loadWindsurfRules(cwd: string): Promise<TtsrRule[]> {
	const rules: TtsrRule[] = [];
	const dirs = [
		`${cwd}/.windsurf/rules`,
		`${process.env.HOME || process.env.USERPROFILE}/.codeium/windsurf/memories`,
	];

	for (const dir of dirs) {
		try {
			const entries = await fs.readdir(dir);
			for (const entry of entries) {
				if (!entry.endsWith(".md")) continue;
				const filePath = `${dir}/${entry}`;
				const content = await fs.readFile(filePath, "utf-8");
				const { metadata, body } = parseFrontmatter(content);
				const rule = normalizeRule(
					filePath,
					dir.includes("memories")
						? "global_rules"
						: entry.replace(/\.md$/, ""),
					body,
					metadata,
				);
				if (rule) rules.push(rule);
			}
		} catch {
			// Directory doesn't exist
		}
	}

	return rules;
}

async function loadClineRules(cwd: string): Promise<TtsrRule[]> {
	const rules: TtsrRule[] = [];

	// Walk upward from cwd for .clinerules
	let dir = cwd;
	while (dir.length > 1) {
		try {
			const stats = await fs.stat(`${dir}/.clinerules`);
			if (stats.isDirectory()) {
				// Directory: load all .md files
				const entries = await fs.readdir(`${dir}/.clinerules`);
				for (const entry of entries) {
					if (!entry.endsWith(".md")) continue;
					const filePath = `${dir}/.clinerules/${entry}`;
					const content = await fs.readFile(filePath, "utf-8");
					const { metadata, body } = parseFrontmatter(content);
					const rule = normalizeRule(
						filePath,
						entry.replace(/\.md$/, ""),
						body,
						metadata,
					);
					if (rule) rules.push(rule);
				}
			} else {
				// File: load as single rule
				const filePath = `${dir}/.clinerules`;
				const content = await fs.readFile(filePath, "utf-8");
				const { metadata, body } = parseFrontmatter(content);
				const rule = normalizeRule(filePath, "clinerules", body, metadata);
				if (rule) rules.push(rule);
			}
			break; // Found it, stop walking
		} catch {
			// Not found, walk up
			dir = dir.slice(0, dir.lastIndexOf("/"));
		}
	}

	// Also check user home
	try {
		const userPath = `${process.env.HOME || process.env.USERPROFILE}/.clinerules`;
		const content = await fs.readFile(userPath, "utf-8");
		const { metadata, body } = parseFrontmatter(content);
		const rule = normalizeRule(userPath, "clinerules", body, metadata);
		if (rule) rules.push(rule);
	} catch {
		// No user rules
	}

	return rules;
}

async function loadGitHubCopilotRules(cwd: string): Promise<TtsrRule[]> {
	const rules: TtsrRule[] = [];

	// Project rules
	const projectDir = `${cwd}/.github/instructions`;
	try {
		const entries = await fs.readdir(projectDir, { recursive: true });
		for (const entry of entries as string[]) {
			if (!entry.endsWith(".instructions.md")) continue;
			const filePath = `${projectDir}/${entry}`;
			const content = await fs.readFile(filePath, "utf-8");
			const { metadata, body } = parseFrontmatter(content);

			// Parse applyTo for glob scoping
			const applyTo = metadata.applyTo as string | undefined;
			let globs: string[] | undefined;

			if (typeof applyTo === "string") {
				const parts = applyTo.split(",").map((s: string) => s.trim());
				if (
					parts.some((p: string) => p === "*" || p === "**" || p === "**/*")
				) {
					globs = undefined;
				} else {
					globs = parts.filter((p: string) => p.length > 0);
				}
			}

			const ruleName = entry.replace(/\.instructions\.md$/, "");
			const rule = normalizeRule(filePath, ruleName, body, {
				...metadata,
				...(globs && { globs }),
			});
			if (rule) rules.push(rule);
		}
	} catch {
		// No project instructions
	}

	// User rules from COPILOT_CUSTOM_INSTRUCTIONS_DIRS
	const customDirs = (process.env.COPILOT_CUSTOM_INSTRUCTIONS_DIRS || "")
		.split(",")
		.map((d: string) => d.trim())
		.filter((d: string) => d.length > 0);

	for (const dir of customDirs) {
		try {
			const entries = await fs.readdir(dir, { recursive: true });
			for (const entry of entries as string[]) {
				if (!entry.endsWith(".instructions.md")) continue;
				const filePath = `${dir}/${entry}`;
				const content = await fs.readFile(filePath, "utf-8");
				const { metadata, body } = parseFrontmatter(content);
				const ruleName = entry.replace(/\.instructions\.md$/, "");
				const rule = normalizeRule(filePath, ruleName, body, metadata);
				if (rule) rules.push(rule);
			}
		} catch {
			// Directory doesn't exist
		}
	}

	return rules;
}

async function loadAgentRules(cwd: string): Promise<TtsrRule[]> {
	const rules: TtsrRule[] = [];

	// Walk upward from cwd for .agent/ and .agents/
	let dir = cwd;
	while (dir.length > 1) {
		for (const agentDir of [".agent", ".agents"]) {
			try {
				const rulesDir = `${dir}/${agentDir}/rules`;
				const entries = await fs.readdir(rulesDir);
				for (const entry of entries) {
					if (!entry.endsWith(".md") && !entry.endsWith(".mdc")) continue;
					const filePath = `${rulesDir}/${entry}`;
					const content = await fs.readFile(filePath, "utf-8");
					const { metadata, body } = parseFrontmatter(content);
					const rule = normalizeRule(
						filePath,
						entry.replace(/\.(md|mdc)$/, ""),
						body,
						metadata,
					);
					if (rule) rules.push(rule);
				}
			} catch {
				// Directory doesn't exist
			}
		}
		dir = dir.slice(0, dir.lastIndexOf("/"));
	}

	// User home
	const homeDirs = [
		`${process.env.HOME || process.env.USERPROFILE}/.agent/rules`,
		`${process.env.HOME || process.env.USERPROFILE}/.agents/rules`,
	];

	for (const dir of homeDirs) {
		try {
			const entries = await fs.readdir(dir);
			for (const entry of entries) {
				if (!entry.endsWith(".md") && !entry.endsWith(".mdc")) continue;
				const filePath = `${dir}/${entry}`;
				const content = await fs.readFile(filePath, "utf-8");
				const { metadata, body } = parseFrontmatter(content);
				const rule = normalizeRule(
					filePath,
					entry.replace(/\.(md|mdc)$/, ""),
					body,
					metadata,
				);
				if (rule) rules.push(rule);
			}
		} catch {
			// Directory doesn't exist
		}
	}

	return rules;
}

// ── Helper: find nearest .omp/ directory ──────────────────────────────────────

async function findNearestOmpDir(cwd: string): Promise<string | null> {
	let dir = cwd;
	while (dir.length > 1) {
		try {
			const stats = await fs.stat(`${dir}/.omp`);
			if (stats.isDirectory()) return dir;
		} catch {
			// Continue walking
		}
		dir = dir.slice(0, dir.lastIndexOf("/"));
	}
	return null;
}

// ── Discovery pipeline ────────────────────────────────────────────────────────

const SOURCES: DiscoverySource[] = [
	{ name: "native", priority: 100, load: loadNativeRules },
	{ name: "cursor", priority: 50, load: loadCursorRules },
	{ name: "windsurf", priority: 50, load: loadWindsurfRules },
	{ name: "cline", priority: 40, load: loadClineRules },
	{ name: "github", priority: 30, load: loadGitHubCopilotRules },
	{ name: "agents", priority: 70, load: loadAgentRules },
];

export interface DiscoveryResult {
	rules: TtsrRule[];
	warnings: string[];
}

/**
 * Discover TTSR rules from all configured sources.
 * Returns deduplicated rules (first-wins for duplicate names).
 */
export async function discoverRules(cwd: string): Promise<DiscoveryResult> {
	const allRules: TtsrRule[] = [];
	const warnings: string[] = [];

	// Sort sources by priority (descending)
	const sortedSources = [...SOURCES].sort((a, b) => b.priority - a.priority);

	// Load from each source
	for (const source of sortedSources) {
		try {
			const rules = await source.load(cwd);
			allRules.push(...rules);
		} catch (error) {
			warnings.push(`Failed to load rules from ${source.name}: ${error}`);
		}
	}

	// Deduplicate by rule name (first-wins)
	const seen = new Set<string>();
	const deduped: TtsrRule[] = [];
	for (const rule of allRules) {
		if (seen.has(rule.name)) {
			warnings.push(
				`Duplicate rule name "${rule.name}" - skipping (first-wins)`,
			);
			continue;
		}
		seen.add(rule.name);
		deduped.push(rule);
	}

	return { rules: deduped, warnings };
}

/**
 * Load discovered rules into a TtsrManager.
 * Returns the number of rules loaded.
 */
export async function loadDiscoveredRules(
	manager: import("./ttsr-manager.ts").TtsrManager,
	cwd: string,
	settings?: Partial<import("../types/ttsr.ts").TtsrSettings>,
): Promise<number> {
	const { rules, warnings } = await discoverRules(cwd);

	let loaded = 0;
	for (const rule of rules) {
		if (settings?.disabledRules?.includes(rule.name)) continue;
		if (manager.addRule(rule)) loaded++;
	}

	// Log warnings
	for (const warning of warnings) {
		console.warn(`[TTSR] ${warning}`);
	}

	return loaded;
}
