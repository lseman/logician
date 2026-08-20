// ── Skill Formatting ──────────────────────────────────────────────────────
// Format skills for system prompt injection. Matches Pi's harness/skills.ts pattern.

import type { PromptTemplate } from "./prompt-templates.ts";

export interface Skill {
	name: string;
	filePath: string;
	content: string;
}

/** Format a skill for inclusion in the system prompt. */
export function formatSkillForSystemPrompt(skill: Skill): string {
	return `<skill name="${skill.name}" location="${skill.filePath}">
${skill.content}
</skill>`;
}

/** Format multiple skills for system prompt. */
export function formatSkillsForSystemPrompt(skills: Skill[]): string {
	if (!skills.length) return "";
	return skills.map(formatSkillForSystemPrompt).join("\n\n");
}

/** Format a prompt template for system prompt. */
export function formatPromptTemplateForSystemPrompt(
	template: PromptTemplate,
): string {
	return `<prompt-template name="${template.name}"${template.description ? ` description="${template.description}"` : ""}>
${template.content}
</prompt-template>`;
}

/** Format prompt templates for system prompt. */
export function formatPromptTemplatesForSystemPrompt(
	templates: PromptTemplate[],
): string {
	if (!templates.length) return "";
	return templates.map(formatPromptTemplateForSystemPrompt).join("\n\n");
}
