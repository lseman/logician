// ── System prompt utilities ───────────────────────────────────────────────
// Basic system prompt construction helpers.

import type { Message } from "../types/index.ts";
import { createSystemMessage } from "./messages.ts";

export function buildDefaultSystemPrompt(): string {
	return "You are a helpful, precise, and safe coding assistant.";
}

export function mergeSystemPrompt(base: string, appendix: string): string {
	if (!appendix.trim()) return base;
	return `${base}\n\n${appendix}`;
}

export function createSystemMessageFromTemplate(
	template: string,
	placeholders: Record<string, string>,
): Message {
	let content = template;
	for (const [key, value] of Object.entries(placeholders)) {
		content = content.replace(new RegExp(`\\{${key}\\}`, "g"), value);
	}
	return createSystemMessage(content);
}
