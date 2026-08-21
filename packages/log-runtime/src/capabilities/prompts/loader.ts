// ── Custom prompt loading ──────────────────────────────────────────────────
// Loads markdown files from `prompts/`/`.logician/prompts/` directories as
// direct, user-typed slash commands — distinct from the skill loader, which loads
// SKILL.md files for autonomous model invocation via read_skill. Prompts are
// never surfaced to the model; they only exist to be typed as /name by the
// user, mirroring Claude Code's `.claude/commands/*.md` convention.

import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { parseFrontmatter } from "@logician/log-core/frontmatter";

export interface Prompt {
	/** Stable invocation id, derived from the filename (without extension). */
	name: string;
	description: string;
	content: string;
	filePath: string;
	/** Slash command-safe form of `name` (currently identical to name). */
	slashName: string;
	aliases?: string[];
	/** Hint shown for /prompt-name argument completion. */
	argumentHint?: string;
}

interface PromptFrontmatter {
	description?: string;
	aliases?: string[] | string;
	"argument-hint"?: string;
	[key: string]: unknown;
}

function toStringList(
	value: string[] | string | undefined,
): string[] | undefined {
	if (value === undefined) return undefined;
	if (Array.isArray(value))
		return value.filter((v): v is string => typeof v === "string");
	return [value];
}

/**
 * Load prompts from one or more directories. Only root-level `*.md` files
 * are considered — prompts are flat, unlike skills' recursive SKILL.md
 * discovery. Missing directories are skipped silently.
 */
export async function loadPrompts(dirs: string[]): Promise<Prompt[]> {
	const byName = new Map<string, Prompt>();
	for (const dir of dirs) {
		let entries: string[];
		try {
			entries = await readdir(dir);
		} catch (_e: unknown) {
			continue; // missing/unreadable dir — skip silently
		}
		for (const entry of entries) {
			if (!entry.endsWith(".md")) continue;
			const filePath = join(dir, entry);
			let raw: string;
			try {
				raw = await readFile(filePath, "utf8");
			} catch (_e: unknown) {
				continue;
			}
			const parsed = parseFrontmatter<PromptFrontmatter>(raw);
			const frontmatter = parsed.ok ? parsed.value.frontmatter : {};
			const body = parsed.ok ? parsed.value.body : raw;
			const name = entry.slice(0, -3);
			const description =
				typeof frontmatter.description === "string" &&
				frontmatter.description.trim()
					? frontmatter.description
					: `Custom prompt: ${name}`;
			const argumentHint =
				typeof frontmatter["argument-hint"] === "string"
					? frontmatter["argument-hint"]
					: undefined;
			// Later dirs win on name collision (project-local overrides parent dirs).
			byName.set(name, {
				name,
				description,
				content: body,
				filePath,
				slashName: name,
				aliases: toStringList(frontmatter.aliases),
				argumentHint,
			});
		}
	}
	return Array.from(byName.values());
}

function normalizeLookupKey(name: string): string {
	return name.trim().toLowerCase().replace(/^\/+/, "").replace(/\s+/g, "-");
}

function promptLookupKeys(prompt: Prompt): string[] {
	return Array.from(
		new Set([prompt.name, prompt.slashName, ...(prompt.aliases ?? [])]),
	);
}

export function findPromptByName(
	prompts: Prompt[],
	name: string,
): Prompt | undefined {
	const normalized = normalizeLookupKey(name);
	for (const prompt of prompts) {
		if (
			promptLookupKeys(prompt).some(
				key => normalizeLookupKey(key) === normalized,
			)
		) {
			return prompt;
		}
	}
	return undefined;
}
