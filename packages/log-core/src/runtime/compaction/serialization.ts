// ── Compaction utilities ────────────────────────────────────────────────────────

import { DEFAULT_TRUNCATION } from "../../system/types/types-config.ts";

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
