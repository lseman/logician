// ── Skill loading system ──────────────────────────────────────────────────────
// Mirrors Pi's harness/skills.ts: recursive SKILL.md discovery, YAML frontmatter
// parsing, validation, ignore-file support, and system-prompt injection.

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { readdir, readFile, realpath, stat } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import ignore from "ignore";
import { parseFrontmatter } from "@logician/agent-core/tools/shared/frontmatter.ts";

// ── Types ────────────────────────────────────────────────────────────────────

export interface Skill {
	/** Stable invocation id, usually the path under a skills root: coding/file_ops. */
	name: string;
	/** Human-facing frontmatter name, e.g. "File Ops". */
	displayName: string;
	description: string;
	content: string;
	filePath: string;
	baseDir: string;
	/** Slash command-safe form derived from `name`, e.g. coding-file_ops. */
	slashName: string;
	disableModelInvocation: boolean;
	/** Tool-name allowlist suggested while this skill is active. */
	allowedTools?: string[];
	aliases?: string[];
	triggers?: string[];
	exampleQueries?: string[];
	whenNotToUse?: string[];
	nextSkills?: string[];
	preferredSequence?: string[];
	entryCriteria?: string[];
	decisionRules?: string[];
	failureRecovery?: string[];
	exitCriteria?: string[];
	antiPatterns?: string[];
	/** Hint shown for /skill-name argument completion. */
	argumentHint?: string;
	/** Preferred model for this skill (advisory). */
	model?: string;
	/** Source of this skill: user (global), project, or explicit path. */
	source: "user" | "project" | "path";
}

export type SkillDiagnosticCode =
	| "file_info_failed"
	| "list_failed"
	| "read_failed"
	| "parse_failed"
	| "invalid_metadata"
	| "collision";

export interface SkillDiagnostic {
	type: "warning" | "collision";
	code: SkillDiagnosticCode;
	message: string;
	path: string;
	winnerPath?: string;
	loserPath?: string;
}

interface SkillFrontmatter {
	name?: string;
	description?: string;
	aliases?: string[] | string;
	triggers?: string[] | string;
	"disable-model-invocation"?: boolean;
	"allowed-tools"?: string[] | string;
	allowed_tools?: string[] | string;
	"argument-hint"?: string;
	argument_hint?: string;
	preferred_tools?: string[] | string;
	example_queries?: string[] | string;
	when_not_to_use?: string[] | string;
	next_skills?: string[] | string;
	preferred_sequence?: string[] | string;
	entry_criteria?: string[] | string;
	decision_rules?: string[] | string;
	failure_recovery?: string[] | string;
	exit_criteria?: string[] | string;
	anti_patterns?: string[] | string;
	model?: string;
	[key: string]: unknown;
}

// ── Constants ────────────────────────────────────────────────────────────────

const MAX_NAME_LENGTH = 64;
const MAX_DESCRIPTION_LENGTH = 1024;
const IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"];
const RESOURCE_DIR_NAMES = ["references", "scripts"];

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Load skills from one or more directories.
 *
 * Traverses directories recursively, loads SKILL.md files, honors ignore files,
 * and returns diagnostics for invalid skill files and name collisions. Missing
 * input directories are skipped silently.
 *
 * @param dirs - Directories to scan. Can be a single path string, an array of
 *   path strings, or objects with `{ dir, source? }` for explicit source tagging.
 */
export async function loadSkills(
	dirs: (
	| string
	| { dir: string; source?: "user" | "project" | "path" }
	)[]
	| string,
): Promise<{ skills: Skill[]; diagnostics: SkillDiagnostic[] }> {
	const skillsMap = new Map<string, Skill>();
	const realPathSet = new Set<string>();
	const diagnostics: SkillDiagnostic[] = [];

	const normalized = Array.isArray(dirs)
		? dirs
		: [{ dir: dirs, source: "path" as const }];

	for (const entry of normalized) {
		const dir = typeof entry === "string" ? entry : entry.dir;
		const source: "user" | "project" | "path" =
			(typeof entry === "string" ? "path" : entry.source ?? "path");
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
		const result = await loadSkillsFromDir(
			dir,
			true,
			ignore(),
			dir,
			source,
		);
		for (const skill of result.skills) {
			const realPath = await safeRealpath(skill.filePath);
			if (realPathSet.has(realPath)) continue;
			realPathSet.add(realPath);

			const existing = skillsMap.get(skill.name);
			if (existing) {
				diagnostics.push({
					type: "collision",
					code: "collision",
					message: `skill "${skill.name}" collision`,
					path: skill.filePath,
					winnerPath: existing.filePath,
					loserPath: skill.filePath,
				});
			} else {
				skillsMap.set(skill.name, skill);
			}
		}
		diagnostics.push(...result.diagnostics);
	}

	return { skills: Array.from(skillsMap.values()), diagnostics };
}

/**
 * Format a skill as an XML invocation block carrying its full body. Used by the
 * on-demand read_skill tool, NOT for bulk system-prompt injection (that would
 * dump every skill body into context — see formatSkillCatalog).
 *
 * Optionally appends additional user instructions after the skill block.
 */
export function formatSkillInvocation(
	skill: Skill,
	additionalInstructions?: string,
): string {
	const toolsNote = skill.allowedTools?.length
		? `\nPreferred tools while following this skill: ${skill.allowedTools.join(", ")}.`
		: "";
	const metadata = formatSkillMetadata(skill);
	const resources = formatSkillResources(skill);
	const skillBlock =
		`<skill name="${escapeXml(skill.name)}" display_name="${escapeXml(skill.displayName)}" location="${escapeXml(skill.filePath)}" base_dir="${escapeXml(skill.baseDir)}">\n` +
		`${metadata ? `${metadata}\n\n` : ""}` +
		`${skill.content}` +
		`${resources ? `\n\n${resources}` : ""}` +
		`\n</skill>${toolsNote}`;
	return additionalInstructions
		? `${skillBlock}\n\n${additionalInstructions}`
		: skillBlock;
}

/**
 * Format a compact catalog of skills (name + description only) for system-prompt
 * injection. The model reads a skill's full body on demand via the read_skill
 * tool, so the prompt stays small regardless of how many skills are installed.
 */
export function formatSkillCatalog(
	skills: Skill[],
	options: { maxChars?: number } = {},
): string {
	const maxChars = options.maxChars ?? 8_000;
	const fullEntries = skills.map((skill) => formatCatalogEntry(skill));
	const nameEntries = skills.map(
		(skill) =>
			`  <skill name="${escapeXml(skill.name)}" slash_command="/${escapeXml(skill.slashName)}" />`,
	);
	const header =
		"<available-skills>\n" +
		"The following skills are available. Matching skills are activated automatically. " +
		"Use read_skill with an exact name to load another skill.\n";
	const footer = "\n</available-skills>";
	const entries: string[] = [];
	let used = header.length + footer.length;
	for (let index = 0; index < skills.length; index++) {
		const remainingNames = nameEntries
			.slice(index + 1)
			.reduce((sum, entry) => sum + entry.length + 1, 0);
		const full = fullEntries[index];
		const entry =
			used + full.length + 1 + remainingNames <= maxChars
				? full
				: nameEntries[index];
		entries.push(entry);
		used += entry.length + 1;
	}
	return (
		header +
		entries.join("\n") +
		footer
	);
}

/**
 * Format skills as an XML block for system-prompt injection.
 * Mirrors Pi's formatSkillsForSystemPrompt: includes name, description, and
 * location for each visible skill (excludes disableModelInvocation skills).
 */
export function formatSkillsForSystemPrompt(skills: Skill[]): string {
	const visibleSkills = skills.filter((s) => !s.disableModelInvocation);
	if (visibleSkills.length === 0) return "";

	const lines = [
		"The following skills provide specialized instructions for specific tasks.",
		"Read the full skill file when the task matches its description.",
		"When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
		"Skill names may be path-like identifiers such as coding/quality; use the exact name when requesting a skill.",
		"",
		"<available_skills>",
	];

	for (const skill of visibleSkills) {
		lines.push("  <skill>");
		lines.push(`    <name>${escapeXml(skill.name)}</name>`);
		lines.push(`    <display_name>${escapeXml(skill.displayName)}</display_name>`);
		lines.push(`    <description>${escapeXml(skill.description)}</description>`);
		lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
		if (skill.aliases?.length) {
			lines.push(`    <aliases>${escapeXml(skill.aliases.join(", "))}</aliases>`);
		}
		if (skill.triggers?.length) {
			lines.push(`    <triggers>${escapeXml(skill.triggers.join("; "))}</triggers>`);
		}
		lines.push("  </skill>");
	}

	lines.push("</available_skills>");
	return lines.join("\n");
}

export function skillLookupKeys(skill: Skill): string[] {
	return uniqueStrings([
		skill.name,
		skill.slashName,
		skill.displayName,
		slugSegment(skill.displayName),
		...skill.name.split("/").slice(-1),
		...(skill.aliases ?? []),
	]).flatMap((key) => uniqueStrings([key, normalizeSkillLookupKey(key)]));
}

export function normalizeSkillLookupKey(name: string): string {
	return name.trim().toLowerCase().replace(/^\/+/, "").replace(/\s+/g, "-");
}

export function findSkillByName(skills: Skill[], name: string): Skill | undefined {
	const normalized = normalizeSkillLookupKey(name);
	for (const skill of skills) {
		if (skillLookupKeys(skill).some((key) => normalizeSkillLookupKey(key) === normalized)) {
			return skill;
		}
	}
	return undefined;
}

// ── Internal ─────────────────────────────────────────────────────────────────

async function loadSkillsFromDir(
	dir: string,
	includeRootFiles: boolean,
	ignoreMatcher: ReturnType<typeof ignore>,
	rootDir: string,
	source: "user" | "project" | "path",
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

		const result = await loadSkillFromFile(fullPath, rootDir, source);
		if (result.skill) skills.push(result.skill);
		diagnostics.push(...result.diagnostics);
		break;
	}

	// Process remaining entries
	for (const entry of entries.sort((a, b) => a.localeCompare(b))) {
		if (entry === "SKILL.md") continue;
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
				source,
			);
			skills.push(...result.skills);
			diagnostics.push(...result.diagnostics);
			continue;
		}

		if (!includeRootFiles || !entry.endsWith(".md")) continue;
		const result = await loadSkillFromFile(fullPath, rootDir, source);
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
		} catch (e: unknown) {
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
	rootDir: string,
	source: "user" | "project" | "path" = "path",
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
	const displayName = frontmatterName || basename(skillDir) || "Unnamed";
	const name = skillIdFromPath(skillDir, rootDir, displayName);
	for (const error of validateName(name)) {
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

	const allowedTools = uniqueStrings([
		...parseStringList(frontmatter["allowed-tools"]),
		...parseStringList(frontmatter.allowed_tools),
		...parseStringList(frontmatter.preferred_tools),
	]);

	return {
		skill: {
			name,
			displayName,
			description,
			content: body,
			filePath,
			baseDir: skillDir,
			slashName: slashNameForSkill(name),
			disableModelInvocation: frontmatter["disable-model-invocation"] === true,
			allowedTools: allowedTools.length ? allowedTools : undefined,
			aliases: optionalStringList(frontmatter.aliases),
			triggers: optionalStringList(frontmatter.triggers),
			exampleQueries: optionalStringList(frontmatter.example_queries),
			whenNotToUse: optionalStringList(frontmatter.when_not_to_use),
			nextSkills: optionalStringList(frontmatter.next_skills),
			preferredSequence: optionalStringList(frontmatter.preferred_sequence),
			entryCriteria: optionalStringList(frontmatter.entry_criteria),
			decisionRules: optionalStringList(frontmatter.decision_rules),
			failureRecovery: optionalStringList(frontmatter.failure_recovery),
			exitCriteria: optionalStringList(frontmatter.exit_criteria),
			antiPatterns: optionalStringList(frontmatter.anti_patterns),
			argumentHint:
				typeof frontmatter["argument-hint"] === "string"
					? frontmatter["argument-hint"]
					: typeof frontmatter.argument_hint === "string"
						? frontmatter.argument_hint
					: undefined,
			model:
				typeof frontmatter.model === "string" ? frontmatter.model : undefined,
			source,
		},
		diagnostics,
	};
}

function validateName(name: string): string[] {
	const errors: string[] = [];
	if (name.length > MAX_NAME_LENGTH * 4)
		errors.push(`name exceeds ${MAX_NAME_LENGTH} characters (${name.length})`);
	if (!/^[a-z0-9][a-z0-9_-]*(\/[a-z0-9][a-z0-9_-]*)*$/.test(name)) {
		errors.push(
			"name contains invalid characters (must be slash-separated lowercase a-z, 0-9, underscores, or hyphens)",
		);
	}
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

// ── Helpers ──────────────────────────────────────────────────────────────────

function basename(p: string): string {
	const normalized = p.replace(/\/+$/, "");
	const slashIndex = normalized.lastIndexOf("/");
	return slashIndex === -1 ? normalized : normalized.slice(slashIndex + 1);
}

function toPosixPath(p: string): string {
	return p.split(sep).join("/");
}

function slugSegment(value: string): string {
	return value
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9_-]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/--+/g, "-") || "unnamed";
}

function skillIdFromPath(skillDir: string, rootDir: string, fallbackName: string): string {
	const rel = toPosixPath(relative(rootDir, skillDir));
	if (rel && !rel.startsWith("..")) {
		return rel
			.split("/")
			.filter(Boolean)
			.map(slugSegment)
			.join("/");
	}
	return slugSegment(fallbackName);
}

function slashNameForSkill(name: string): string {
	return name.replace(/\//g, "-");
}

function parseStringList(value: unknown): string[] {
	if (Array.isArray(value)) return value.map(String).map((s) => s.trim()).filter(Boolean);
	if (typeof value === "string") {
		return value
			.split(/\r?\n|,/)
			.map((s) => s.trim())
			.filter(Boolean);
	}
	return [];
}

function optionalStringList(value: unknown): string[] | undefined {
	const items = uniqueStrings(parseStringList(value));
	return items.length ? items : undefined;
}

function uniqueStrings(values: string[]): string[] {
	return Array.from(new Set(values.map((v) => v.trim()).filter(Boolean)));
}

function formatSkillMetadata(skill: Skill): string {
	const lines = ["<metadata>"];
	lines.push(`  <description>${escapeXml(skill.description)}</description>`);
	appendListXml(lines, "aliases", skill.aliases);
	appendListXml(lines, "triggers", skill.triggers);
	appendListXml(lines, "preferred_tools", skill.allowedTools);
	appendListXml(lines, "example_queries", skill.exampleQueries);
	appendListXml(lines, "when_not_to_use", skill.whenNotToUse);
	appendListXml(lines, "next_skills", skill.nextSkills);
	appendListXml(lines, "preferred_sequence", skill.preferredSequence);
	appendListXml(lines, "entry_criteria", skill.entryCriteria);
	appendListXml(lines, "decision_rules", skill.decisionRules);
	appendListXml(lines, "failure_recovery", skill.failureRecovery);
	appendListXml(lines, "exit_criteria", skill.exitCriteria);
	appendListXml(lines, "anti_patterns", skill.antiPatterns);
	lines.push("</metadata>");
	return lines.length > 2 ? lines.join("\n") : "";
}

function appendListXml(lines: string[], tag: string, values?: string[]): void {
	if (!values?.length) return;
	lines.push(`  <${tag}>`);
	for (const value of values) lines.push(`    <item>${escapeXml(value)}</item>`);
	lines.push(`  </${tag}>`);
}

function formatCatalogEntry(skill: Skill): string {
	const attrs = [
		`name="${escapeXml(skill.name)}"`,
		`display_name="${escapeXml(skill.displayName)}"`,
		`slash_command="/${escapeXml(skill.slashName)}"`,
	];
	if (skill.aliases?.length) attrs.push(`aliases="${escapeXml(skill.aliases.join(", "))}"`);
	if (skill.triggers?.length) attrs.push(`triggers="${escapeXml(skill.triggers.join("; "))}"`);
	if (skill.allowedTools?.length) attrs.push(`preferred_tools="${escapeXml(skill.allowedTools.join(", "))}"`);
	if (skill.nextSkills?.length) attrs.push(`next_skills="${escapeXml(skill.nextSkills.join(", "))}"`);
	return `  <skill ${attrs.join(" ")}>${escapeXml(skill.description)}</skill>`;
}

function formatSkillResources(skill: Skill): string {
	const lines = ["<resources>"];
	for (const dirName of RESOURCE_DIR_NAMES) {
		const dir = join(skill.baseDir, dirName);
		if (!existsSync(dir)) continue;
		lines.push(`  <${dirName} dir="${escapeXml(dir)}">`);
		for (const item of listResourceFiles(dir).slice(0, 40)) {
			lines.push(`    <file>${escapeXml(item)}</file>`);
		}
		lines.push(`  </${dirName}>`);
	}
	lines.push("</resources>");
	return lines.length > 2 ? lines.join("\n") : "";
}

function listResourceFiles(dir: string, base = dir): string[] {
	let entries: string[];
	try {
		entries = readdirSync(dir);
	} catch (e: unknown) {
		return [];
	}
	const files: string[] = [];
	for (const entry of entries.sort((a, b) => a.localeCompare(b))) {
		if (entry.startsWith(".")) continue;
		const full = join(dir, entry);
		const info = safeStatSync(full);
		if (!info) continue;
		if (info.isDirectory()) files.push(...listResourceFiles(full, base));
		else files.push(toPosixPath(relative(base, full)));
	}
	return files;
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
	} catch (e: unknown) {
		return null;
	}
}

async function safeReaddir(p: string): Promise<string[] | null> {
	try {
		return await readdir(p);
	} catch (e: unknown) {
		return null;
	}
}

async function safeReadFile(p: string): Promise<string | null> {
	try {
		return await readFile(p, "utf8");
	} catch (e: unknown) {
		return null;
	}
}

async function safeRealpath(p: string): Promise<string> {
	try {
		return await realpath(p);
	} catch (e: unknown) {
		return p;
	}
}

function safeStatSync(p: string): ReturnType<typeof statSync> | null {
	try {
		return statSync(p);
	} catch (e: unknown) {
		return null;
	}
}
