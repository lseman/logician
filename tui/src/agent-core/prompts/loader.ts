// ── Prompt templates loader ──────────────────────────────────────────────────
// Loads .md files from configured template directories.
// Supports variable substitution: {{variable_name}} -> value.
// Templates are available via /<name> slash commands.

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export interface PromptTemplate {
	name: string;
	path: string;
	content: string;
	source: "global" | "project";
}

export interface PromptTemplateResult {
	templates: PromptTemplate[];
	diagnostics: Array<{ type: "warning" | "error"; message: string; path: string }>;
}

/**
 * Load prompt templates from a directory.
 * All .md files are treated as templates. The filename (without extension)
 * becomes the template name, used as the slash command (e.g., /review).
 */
function loadTemplatesFromDir(dir: string, source: "global" | "project"): PromptTemplateResult {
	const templates: PromptTemplate[] = [];
	const diagnostics: PromptTemplateResult["diagnostics"] = [];

	if (!existsSync(dir)) return { templates, diagnostics };

	const entries = readdirSync(dir);
	for (const entry of entries) {
		if (!entry.endsWith(".md")) continue;
		const fullPath = join(dir, entry);
		const name = entry.slice(0, -3); // remove .md

		// Skip hidden files and directories
		if (name.startsWith(".")) continue;

		try {
			const content = readFileSync(fullPath, "utf-8");
			templates.push({ name, path: fullPath, content, source });
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			diagnostics.push({ type: "error", message: `Failed to read template: ${message}`, path: fullPath });
		}
	}

	return { templates, diagnostics };
}

export function loadPromptTemplates(options: {
	agentDir: string;
	cwd: string;
}): PromptTemplateResult {
	const allTemplates: PromptTemplate[] = [];
	const allDiagnostics: PromptTemplateResult["diagnostics"] = [];

	// Global templates
	const globalDir = join(options.agentDir, "prompts");
	const globalResult = loadTemplatesFromDir(globalDir, "global");
	allTemplates.push(...globalResult.templates);
	allDiagnostics.push(...globalResult.diagnostics);

	// Project templates
	const projectDir = join(options.cwd, ".logician", "prompts");
	const projectResult = loadTemplatesFromDir(projectDir, "project");
	allTemplates.push(...projectResult.templates);
	allDiagnostics.push(...projectResult.diagnostics);

	return { templates: allTemplates, diagnostics: allDiagnostics };
}

/**
 * Resolve a template by name. Returns the template content or undefined.
 */
export function resolveTemplate(templates: PromptTemplate[], name: string): PromptTemplate | undefined {
	return templates.find((t) => t.name.toLowerCase() === name.toLowerCase());
}

/**
 * Substitute variables in a template string.
 * Supports {{variable_name}} syntax.
 */
export function substituteTemplate(
	content: string,
	variables: Record<string, string>,
): string {
	let result = content;
	for (const [key, value] of Object.entries(variables)) {
		const regex = new RegExp(`\\{\\{${key}\\}\\}`, "g");
		result = result.replace(regex, value);
	}
	return result;
}

/**
 * Extract frontmatter variables from a template file.
 * Looks for a YAML frontmatter block with a `variables` key.
 */
export function extractTemplateVariables(content: string): Record<string, string> {
	if (!content.startsWith("---")) return {};

	const endIndex = content.indexOf("\n---", 3);
	if (endIndex === -1) return {};

	const yamlString = content.slice(4, endIndex);
	const lines = yamlString.split("\n");

	const variables: Record<string, string> = {};
	let inVariables = false;

	for (const line of lines) {
		if (line.trim() === "variables:") {
			inVariables = true;
			continue;
		}
		if (inVariables) {
			if (line.startsWith("  ") || line.startsWith("\t")) {
				const match = line.match(/^\s+(\w+):\s*(.+)$/);
				if (match) {
					variables[match[1]] = match[2].replace(/^["']|["']$/g, "");
				}
			} else {
				break;
			}
		}
	}

	return variables;
}
