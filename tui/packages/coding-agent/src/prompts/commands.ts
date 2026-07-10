// ── Prompt template slash command handler ─────────────────────────────────────
// /prompt <name> [variable=value ...] — expand a template and return the result.

import { substituteTemplate, type PromptTemplate } from "./loader.ts";

export interface TemplateCommandResult {
	success: boolean;
	content?: string;
	error?: string;
}

/**
 * Handle a /prompt <name> [key=value ...] command.
 */
export function handleTemplateCommand(
	templates: PromptTemplate[],
	args: string,
): TemplateCommandResult {
	const parts = args.trim().split(/\s+/);
	if (parts.length === 0) {
		return { success: false, error: "Usage: /prompt <name> [key=value ...]\n\nAvailable templates:\n" + listTemplates(templates) };
	}

	const name = parts[0];
	const template = templates.find((t) => t.name.toLowerCase() === name.toLowerCase());

	if (!template) {
		return { success: false, error: `Template "${name}" not found.\n\nAvailable templates:\n` + listTemplates(templates) };
	}

	// Parse variable assignments
	const variables: Record<string, string> = {};
	for (let i = 1; i < parts.length; i++) {
		const eqIndex = parts[i].indexOf("=");
		if (eqIndex > 0) {
			const key = parts[i].slice(0, eqIndex);
			const value = parts[i].slice(eqIndex + 1);
			variables[key] = value;
		} else {
			// Single arg that's not a variable — treat as content override
			variables["_content"] = parts[i];
		}
	}

	const content = substituteTemplate(template.content, variables);

	return { success: true, content };
}

function listTemplates(templates: PromptTemplate[]): string {
	const bySource = new Map<string, PromptTemplate[]>();
	for (const t of templates) {
		const list = bySource.get(t.source) ?? [];
		list.push(t);
		bySource.set(t.source, list);
	}

	let output = "";
	for (const [source, tpls] of bySource) {
		output += `\n  [${source}] ${tpls.map((t) => t.name).join(", ")}\n`;
	}
	return output.trim();
}
