// ── Compaction utilities ────────────────────────────────────────────────────────
// Merged: Pi's serializeConversation (with thinking blocks) + Logician's
// file operation tracking.

import { DEFAULT_TRUNCATION } from "../agent/types/types-truncation.ts";

// ============================================================================
// File Operation Tracking
// ============================================================================

export interface FileOperations {
	read: Set<string>;
	written: Set<string>;
	edited: Set<string>;
}

export function createFileOps(): FileOperations {
	return {
		read: new Set(),
		written: new Set(),
		edited: new Set(),
	};
}

/**
 * Extract file operations from tool calls in an assistant message.
 * Handles both Logician tool names (read_file, write_file, edit_file) and
 * Pi tool names (read, write, edit).
 */
export function extractFileOpsFromMessage(
	message: { role: string; content?: unknown[] | string | null },
	fileOps: FileOperations,
): void {
	if (message.role !== "assistant") return;
	if (!Array.isArray(message.content)) return;

	for (const block of message.content) {
		if (typeof block !== "object" || block === null) continue;
		if (!("type" in block) || block.type !== "toolCall") continue;
		const call = block as {
			name?: string;
			arguments?: Record<string, unknown>;
		};
		if (!call.name || !call.arguments) continue;

		const args = call.arguments as Record<string, unknown>;
		const path = typeof args.path === "string" ? args.path : undefined;
		if (!path) continue;

		switch (call.name) {
			case "read_file":
			case "read":
				fileOps.read.add(path);
				break;
			case "write_file":
			case "write":
				fileOps.written.add(path);
				break;
			case "edit_file":
			case "edit":
				fileOps.edited.add(path);
				break;
		}
	}
}

/**
 * Compute final file lists from file operations.
 * Returns readFiles (files only read, not modified) and modifiedFiles.
 */
export function computeFileLists(fileOps: FileOperations): {
	readFiles: string[];
	modifiedFiles: string[];
} {
	const modified = new Set([...fileOps.edited, ...fileOps.written]);
	const readOnly = [...fileOps.read].filter(f => !modified.has(f)).sort();
	const modifiedFiles = [...modified].sort();
	return { readFiles: readOnly, modifiedFiles };
}

/**
 * Format file operations as XML tags for summary.
 */
export function formatFileOperations(
	readFiles: string[],
	modifiedFiles: string[],
): string {
	const sections: string[] = [];
	if (readFiles.length > 0) {
		sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
	}
	if (modifiedFiles.length > 0) {
		sections.push(
			`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`,
		);
	}
	if (sections.length === 0) return "";
	return `\n\n${sections.join("\n\n")}`;
}

// ============================================================================
// Message Serialization — Pi's version with thinking block support
// ============================================================================

const TOOL_RESULT_MAX_CHARS = DEFAULT_TRUNCATION.compactionSummaryMaxChars;

function safeJsonStringify(value: unknown): string {
	try {
		return JSON.stringify(value) ?? "undefined";
	} catch (_e: unknown) {
		return "[unserializable]";
	}
}

function truncateForSummary(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const truncatedChars = text.length - maxChars;
	return `${text.slice(0, maxChars)}\n\n[... ${truncatedChars} more characters truncated]`;
}

function textContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter(
			(block): block is { type: string; text: string } =>
				typeof block === "object" &&
				block !== null &&
				"type" in block &&
				block.type === "text" &&
				"text" in block &&
				typeof block.text === "string",
		)
		.map(block => block.text)
		.join("");
}

/**
 * Serialize conversation messages to text for summarization.
 * Prevents the model from treating it as a conversation to continue.
 * Includes thinking blocks (from Pi) for reasoning model awareness.
 */
export function serializeConversation(
	messages: Array<{
		role: string;
		content?: unknown;
	}>,
): string {
	const parts: string[] = [];

	for (const msg of messages) {
		if (!msg) continue;
		if (msg.role === "user") {
			const content = textContent(msg.content);
			if (content) parts.push(`[User]: ${content}`);
		} else if (msg.role === "assistant") {
			const textParts: string[] = [];
			const thinkingParts: string[] = [];
			const toolCalls: string[] = [];

			if (typeof msg.content === "string") {
				textParts.push(msg.content);
			} else if (Array.isArray(msg.content)) {
				for (const block of msg.content) {
					if (typeof block !== "object" || block === null) continue;
					if (block.type === "text" && block.text) {
						textParts.push(block.text);
					} else if (
						block.type === "thinking" &&
						"thinking" in block &&
						typeof block.thinking === "string"
					) {
						thinkingParts.push(block.thinking);
					} else if (
						block.type === "toolCall" &&
						typeof block === "object" &&
						block !== null &&
						"name" in block
					) {
						const call = block as {
							name?: string;
							arguments?: Record<string, unknown>;
						};
						const args = call.arguments ?? {};
						const argsStr = Object.entries(args)
							.map(([k, v]) => `${k}=${safeJsonStringify(v)}`)
							.join(", ");
						toolCalls.push(`${call.name ?? "unknown"}(${argsStr})`);
					}
				}
			}

			if (thinkingParts.length > 0) {
				parts.push(`[Assistant thinking]: ${thinkingParts.join("\n")}`);
			}
			if (textParts.length > 0)
				parts.push(`[Assistant]: ${textParts.join("\n")}`);
			if (toolCalls.length > 0)
				parts.push(`[Assistant tool calls]: ${toolCalls.join("; ")}`);
		} else if (msg.role === "toolResult" || msg.role === "tool_result") {
			const content = textContent(msg.content);
			if (content) {
				parts.push(
					`[Tool result]: ${truncateForSummary(content, TOOL_RESULT_MAX_CHARS)}`,
				);
			}
		}
	}

	return parts.join("\n\n");
}

// ============================================================================
// Summarization System Prompt
// ============================================================================

export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

// ============================================================================
// Summarization Prompts
// ============================================================================

export const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

export const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

/** Summarization prompt for turns split during compaction. */
export const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;
