// ── Skill loading system ──────────────────────────────────────────────────────
// Mirrors Pi's harness/skills.ts: recursive SKILL.md discovery, YAML frontmatter
// parsing, validation, ignore-file support, and system-prompt injection.

import { readFileSync } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import { dirname, join, relative } from "node:path";
import ignore from "ignore";
import { parse } from "yaml";

// ── Types ────────────────────────────────────────────────────────────────────

export interface Skill {
	name: string;
	description: string;
	content: string;
	filePath: string;
	disableModelInvocation: boolean;
}

export type SkillDiagnosticCode =
	| "file_info_failed"
	| "list_failed"
	| "read_failed"
	| "parse_failed"
	| "invalid_metadata";

export interface SkillDiagnostic {
	type: "warning";
	code: SkillDiagnosticCode;
	message: string;
	path: string;
}

interface SkillFrontmatter {
	name?: string;
	description?: string;
	"disable-model-invocation"?: boolean;
	[key: string]: unknown;
}

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_NAME_LENGTH = 64;
const MAX_DESCRIPTION_LENGTH = 1024;
const IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"];

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Load skills from one or more directories.
 *
 * Traverses directories recursively, loads SKILL.md files, honors ignore files,
 * and returns diagnostics for invalid skill files. Missing input directories
 * are skipped silently.
 */
export async function loadSkills(
	dirs: string | string[],
): Promise<{ skills: Skill[]; diagnostics: SkillDiagnostic[] }> {
	const skills: Skill[] = [];
	const diagnostics: SkillDiagnostic[] = [];

	for (const dir of Array.isArray(dirs) ? dirs : [dirs]) {
		const rootInfo = await safeStat(dir);
		if (!rootInfo || rootInfo.isDirectory() === false) {
			if (rootInfo?.isDirectory() === false) {
				diagnostics.push({
					type: "warning",
					code: "file_info_failed",
					message: `Not a directory: ${dir}`,
					path: dir,
				});
			}
			continue;
		}
		const result = await loadSkillsFromDir(dir, true, ignore(), dir);
		skills.push(...result.skills);
		diagnostics.push(...result.diagnostics);
	}

	return { skills, diagnostics };
}

/**
 * Format a skill as an XML invocation block for system-prompt injection.
 *
 * Optionally appends additional user instructions after the skill block.
 */
export function formatSkillInvocation(
	skill: Skill,
	additionalInstructions?: string,
): string {
	const skillBlock = `<skill name="${escapeXml(skill.name)}" location="${escapeXml(skill.filePath)}">\n${skill.content}\n</skill>`;
	return additionalInstructions
		? `${skillBlock}\n\n${additionalInstructions}`
		: skillBlock;
}

// ── Internal ─────────────────────────────────────────────────────────────────

async function loadSkillsFromDir(
	dir: string,
	includeRootFiles: boolean,
	ignoreMatcher: ReturnType<typeof ignore>,
	rootDir: string,
): Promise<{ skills: Skill[]; diagnostics: SkillDiagnostic[] }> {
	const skills: Skill[] = [];
	const diagnostics: SkillDiagnostic[] = [];

	const dirInfo = await safeStat(dir);
	if (!dirInfo || dirInfo.isDirectory() === false)
		return { skills, diagnostics };

	// Load ignore rules from this directory
	const ignorePatterns = loadIgnorePatterns(dir, rootDir, diagnostics);
	if (ignorePatterns.length > 0) ignoreMatcher.add(ignorePatterns);

	const entries = await safeReaddir(dir);
	if (!entries) {
		diagnostics.push({
			type: "warning",
			code: "list_failed",
			message: `Cannot read directory: ${dir}`,
			path: dir,
		});
		return { skills, diagnostics };
	}

	// Look for SKILL.md files in this directory
	for (const entry of entries) {
		if (entry !== "SKILL.md") continue;
		const fullPath = join(dir, entry);
		const relPath = relative(rootDir, fullPath);
		if (ignoreMatcher.ignores(relPath)) continue;

		const result = await loadSkillFromFile(fullPath);
		if (result.skill) skills.push(result.skill);
		diagnostics.push(...result.diagnostics);
		// Only one SKILL.md per directory — skip remaining entries
		return { skills, diagnostics };
	}

	// Process remaining entries
	for (const entry of entries.sort((a, b) => a.localeCompare(b))) {
		if (entry.startsWith(".") || entry === "node_modules") continue;
		const fullPath = join(dir, entry);
		const relPath = relative(rootDir, fullPath);
		const ignorePath = entry.endsWith("/") ? `${relPath}/` : relPath;
		if (ignoreMatcher.ignores(ignorePath)) continue;

		const entryInfo = await safeStat(fullPath);
		if (!entryInfo) continue;

		if (entryInfo.isDirectory()) {
			const result = await loadSkillsFromDir(
				fullPath,
				false,
				ignoreMatcher,
				rootDir,
			);
			skills.push(...result.skills);
			diagnostics.push(...result.diagnostics);
			continue;
		}

		if (!includeRootFiles || !entry.endsWith(".md")) continue;
		const result = await loadSkillFromFile(fullPath);
		if (result.skill) skills.push(result.skill);
		diagnostics.push(...result.diagnostics);
	}

	return { skills, diagnostics };
}

function loadIgnorePatterns(
	dir: string,
	rootDir: string,
	_diagnostics: SkillDiagnostic[],
): string[] {
	const relativeDir = relative(rootDir, dir);
	const prefix = relativeDir ? `${relativeDir}/` : "";
	const patterns: string[] = [];

	for (const filename of IGNORE_FILE_NAMES) {
		const ignorePath = join(dir, filename);
		let content: string;
		try {
			content = readFileSync(ignorePath, "utf8");
		} catch {
			continue;
		}

		const lines = content
			.split(/\r?\n/)
			.map((line) => prefixIgnorePattern(line, prefix))
			.filter((line): line is string => line !== null);

		patterns.push(...lines);
	}

	return patterns;
}

function prefixIgnorePattern(line: string, prefix: string): string | null {
	const trimmed = line.trim();
	if (!trimmed) return null;
	if (trimmed.startsWith("#") && !trimmed.startsWith("\\#")) return null;

	let pattern = line;
	let negated = false;
	if (pattern.startsWith("!")) {
		negated = true;
		pattern = pattern.slice(1);
	} else if (pattern.startsWith("\\!")) {
		pattern = pattern.slice(1);
	}
	if (pattern.startsWith("/")) pattern = pattern.slice(1);
	const prefixed = prefix ? `${prefix}${pattern}` : pattern;
	return negated ? `!${prefixed}` : prefixed;
}

async function loadSkillFromFile(
	filePath: string,
): Promise<{ skill: Skill | null; diagnostics: SkillDiagnostic[] }> {
	const diagnostics: SkillDiagnostic[] = [];

	const rawContent = await safeReadFile(filePath);
	if (!rawContent) {
		diagnostics.push({
			type: "warning",
			code: "read_failed",
			message: `Cannot read skill file: ${filePath}`,
			path: filePath,
		});
		return { skill: null, diagnostics };
	}

	const parsed = parseFrontmatter<SkillFrontmatter>(rawContent);
	if (!parsed.ok) {
		diagnostics.push({
			type: "warning",
			code: "parse_failed",
			message: parsed.error.message,
			path: filePath,
		});
		return { skill: null, diagnostics };
	}

	const { frontmatter, body } = parsed.value;
	const skillDir = dirname(filePath);
	const parentDirName = basename(skillDir);
	const description =
		typeof frontmatter.description === "string"
			? frontmatter.description
			: undefined;

	for (const error of validateDescription(description)) {
		diagnostics.push({
			type: "warning",
			code: "invalid_metadata",
			message: error,
			path: filePath,
		});
	}

	const frontmatterName =
		typeof frontmatter.name === "string" ? frontmatter.name : undefined;
	const name = frontmatterName || parentDirName;
	for (const error of validateName(name, parentDirName)) {
		diagnostics.push({
			type: "warning",
			code: "invalid_metadata",
			message: error,
			path: filePath,
		});
	}

	if (!description || description.trim() === "") {
		return { skill: null, diagnostics };
	}

	return {
		skill: {
			name,
			description,
			content: body,
			filePath,
			disableModelInvocation: frontmatter["disable-model-invocation"] === true,
		},
		diagnostics,
	};
}

function validateName(name: string, parentDirName: string): string[] {
	const errors: string[] = [];
	if (name !== parentDirName)
		errors.push(
			`name "${name}" does not match parent directory "${parentDirName}"`,
		);
	if (name.length > MAX_NAME_LENGTH)
		errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
	if (!/^[a-z0-9-]+$/.test(name)) {
		errors.push(
			"name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)",
		);
	}
	if (name.startsWith("-") || name.endsWith("-"))
		errors.push("name must not start or end with a hyphen");
	if (name.includes("--"))
		errors.push("name must not contain consecutive hyphens");
	return errors;
}

function validateDescription(description: string | undefined): string[] {
	const errors: string[] = [];
	if (!description || description.trim() === "") {
		errors.push("description is required");
	} else if (description.length > MAX_DESCRIPTION_LENGTH) {
		errors.push(
			`description exceeds ${MAX_DESCRIPTION_LENGTH} characters (${description.length})`,
		);
	}
	return errors;
}

function parseFrontmatter<T extends Record<string, unknown>>(
	content: string,
):
	| { ok: true; value: { frontmatter: T; body: string } }
	| { ok: false; error: Error } {
	try {
		const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
		if (!normalized.startsWith("---"))
			return {
				ok: true,
				value: { frontmatter: {} as T, body: normalized },
			};
		const endIndex = normalized.indexOf("\n---", 3);
		if (endIndex === -1)
			return {
				ok: true,
				value: { frontmatter: {} as T, body: normalized },
			};
		const yamlString = normalized.slice(4, endIndex);
		const body = normalized.slice(endIndex + 4).trim();
		return {
			ok: true,
			value: {
				frontmatter: (parse(yamlString) ?? {}) as T,
				body,
			},
		};
	} catch (error) {
		return { ok: false, error: error as Error };
	}
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function basename(p: string): string {
	const normalized = p.replace(/\/+$/, "");
	const slashIndex = normalized.lastIndexOf("/");
	return slashIndex === -1 ? normalized : normalized.slice(slashIndex + 1);
}

function escapeXml(s: string): string {
	return s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&apos;");
}

async function safeStat(p: string): Promise<ReturnType<typeof stat> | null> {
	try {
		return await stat(p);
	} catch {
		return null;
	}
}

async function safeReaddir(p: string): Promise<string[] | null> {
	try {
		return await readdir(p);
	} catch {
		return null;
	}
}

async function safeReadFile(p: string): Promise<string | null> {
	try {
		return await readFile(p, "utf8");
	} catch {
		return null;
	}
}
