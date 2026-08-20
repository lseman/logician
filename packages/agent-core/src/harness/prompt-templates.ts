// ── Prompt templates ──────────────────────────────────────────────────────
// Template definitions for common prompt patterns.

export interface PromptTemplate {
	id: string;
	name: string;
	description: string;
	content: string;
	variables: string[];
}

export const PROMPT_TEMPLATES: PromptTemplate[] = [
	{
		id: "default",
		name: "Default Assistant",
		description: "Standard helpful assistant prompt",
		content: "You are a helpful coding assistant.",
		variables: [],
	},
	{
		id: "code-review",
		name: "Code Review",
		description: "Focused on reviewing code changes",
		content:
			"You are a code review assistant. Analyze the provided code for correctness, security, performance, and maintainability.",
		variables: ["language", "framework"],
	},
	{
		id: "debug",
		name: "Debug Assistant",
		description: "Focused on debugging and error analysis",
		content:
			"You are a debugging assistant. Analyze the provided error messages and code to identify root causes and suggest fixes.",
		variables: [],
	},
];

export function getTemplate(id: string): PromptTemplate | undefined {
	return PROMPT_TEMPLATES.find(t => t.id === id);
}

export function renderTemplate(
	template: PromptTemplate,
	variables: Record<string, string>,
): string {
	let content = template.content;
	for (const [key, value] of Object.entries(variables)) {
		content = content.replace(new RegExp(`\\{${key}\\}`, "g"), value);
	}
	return content;
}
