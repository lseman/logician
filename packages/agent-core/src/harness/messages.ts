// ── Message creation, conversion, and token estimation ──────────────────────
// Replaces core/messages.ts — the message factory and chat format converter.

import type {
	AgentMessage,
	BashExecutionMessage,
	BranchSummaryMessage,
	CompactionSummaryMessage,
	CustomMessage,
	Message,
} from "../types/index.ts";
import { microCompactMessages as microCompactCompactable } from "./compaction/compaction.ts";

export const COMPACTION_SUMMARY_PREFIX = `The conversation history before this point was compacted into the following summary:

<summary>
`;

export const COMPACTION_SUMMARY_SUFFIX = `
</summary>`;

export const BRANCH_SUMMARY_PREFIX = `The following is a summary of a branch that this conversation came back from:

<summary>
`;

export const BRANCH_SUMMARY_SUFFIX = "</summary>";

export function bashExecutionToText(msg: BashExecutionMessage): string {
	let text = `Ran \`${msg.command}\`\n`;
	if (msg.output) {
		text += `\`\`\`\n${msg.output}\n\`\`\``;
	} else {
		text += "(no output)";
	}
	if (msg.cancelled) {
		text += "\n\n(command cancelled)";
	} else if (
		msg.exitCode !== null &&
		msg.exitCode !== undefined &&
		msg.exitCode !== 0
	) {
		text += `\n\nCommand exited with code ${msg.exitCode}`;
	}
	if (msg.truncated && msg.fullOutputPath) {
		text += `\n\n[Output truncated. Full output: ${msg.fullOutputPath}]`;
	}
	return text;
}

/** Convert AgentMessage[] to LLM-compatible Message[]. */
export function convertToLlm(messages: AgentMessage[]): Message[] {
	return messages
		.map((m): Message | undefined => {
			if (!m) return undefined;
			const role = m.role as string;
			switch (role) {
				case "compactionSummary": {
					const msg = m as unknown as CompactionSummaryMessage;
					return {
						role: "user",
						content:
							COMPACTION_SUMMARY_PREFIX +
							msg.summary +
							COMPACTION_SUMMARY_SUFFIX,
						timestamp: msg.timestamp,
					};
				}
				case "branchSummary": {
					const msg = m as unknown as BranchSummaryMessage;
					return {
						role: "user",
						content:
							BRANCH_SUMMARY_PREFIX + msg.summary + BRANCH_SUMMARY_SUFFIX,
						timestamp: msg.timestamp,
					};
				}
				case "bashExecution": {
					const msg = m as unknown as BashExecutionMessage;
					if (msg.excludeFromContext) return undefined;
					return {
						role: "user",
						content: bashExecutionToText(msg),
						timestamp: msg.timestamp,
					};
				}
				case "custom": {
					const msg = m as unknown as CustomMessage;
					return {
						role: "user",
						content: msg.content,
						timestamp: msg.timestamp,
					};
				}
				case "system":
				case "user":
				case "assistant":
				case "tool":
					return m as Message;
				default:
					return undefined;
			}
		})
		.filter((m): m is Message => m !== undefined);
}

export function createUserMessage(content: string): Message {
	return { role: "user", content, timestamp: Date.now() };
}

export function sanitizeToolCallArguments<T extends { arguments: string }>(
	toolCalls: T[],
): T[] {
	let changed = false;
	const sanitized = toolCalls.map(call => {
		try {
			JSON.parse(call.arguments);
			return call;
		} catch {
			changed = true;
			return { ...call, arguments: "{}" };
		}
	});
	return changed ? sanitized : toolCalls;
}

export function createSystemMessage(content: string): Message {
	return { role: "system", content, timestamp: Date.now() };
}

export function createAssistantMessage(
	content: string,
	toolCalls?: Array<{ id: string; name: string; arguments: string }>,
): Message {
	const effectiveContent =
		toolCalls && toolCalls.length > 0 ? content || null : content || " ";
	return {
		role: "assistant",
		content: effectiveContent,
		tool_calls: toolCalls,
		timestamp: Date.now(),
	};
}

export function createToolResultMessage(
	toolCallId: string,
	toolName: string,
	result: string,
	_isError: boolean = false,
): Message {
	return {
		role: "tool",
		content: result,
		tool_call_id: toolCallId,
		name: toolName,
		timestamp: Date.now(),
	};
}

export function convertToChatFormat(
	messages: Message[],
): Record<string, unknown>[] {
	return messages
		.filter((m): m is Message => m != null)
		.map(m => {
			const obj: Record<string, unknown> = { role: m.role };
			if (m.content !== null && m.content !== undefined) {
				obj.content = m.content;
			}
			if (m.tool_call_id) obj.tool_call_id = m.tool_call_id;
			if (m.tool_calls?.length) {
				obj.tool_calls = m.tool_calls.map(tc => ({
					id: tc.id,
					type: "function",
					function: {
						name: tc.name,
						arguments: tc.arguments,
					},
				}));
			}
			if (m.name) obj.name = m.name;
			if (m.role === "assistant" && !obj.content && !obj.tool_calls) {
				obj.content = "";
			}
			return obj;
		});
}

export function estimateTokens(text: string): number {
	if (!text || text.length === 0) return 0;
	if (isJsonLike(text)) return estimateJsonTokens(text);
	if (isCodeLike(text)) return estimateCodeTokens(text);
	return estimateNaturalLanguageTokens(text);
}

function isJsonLike(text: string): boolean {
	const trimmed = text.trim();
	if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return false;
	const jsonChars = (text.match(/[{}[\]:,]/g) || []).length;
	return jsonChars / Math.max(1, text.length) > 0.05;
}

function isCodeLike(text: string): boolean {
	const trimmed = text.trim();
	if (trimmed.length < 20) return false;
	const codePatterns = [
		/\b(function|const|let|var|class|import|export|def|return|if|else|for|while|async|await|try|catch|new|this|public|private|protected)\b/g,
		/\b(=>|===|!==|&&|\|\||\?\?|\.\.|\*\*|\+\+|--)\b/g,
		/[{}()[\];]/g,
	];
	let matchCount = 0;
	for (const pattern of codePatterns) {
		const matches = trimmed.match(pattern);
		if (matches) matchCount += matches.length;
	}
	const ratio = matchCount / Math.max(1, trimmed.split(/\s+/).length);
	return ratio > 0.3;
}

function estimateJsonTokens(text: string): number {
	const trimmed = text.trim();
	const strings = trimmed.match(/"[^"\\]*(?:\\.[^"\\]*)*"/g) || [];
	const keys = trimmed.match(/"[^"]+"\s*:/g) || [];
	const numbers = trimmed.match(/\b\d+(?:\.\d+)?\b/g) || [];
	const booleans = (trimmed.match(/\b(true|false)\b/g) || []).length;
	const nulls = (trimmed.match(/\bnull\b/g) || []).length;
	let tokens = strings.reduce((sum, s) => sum + 1 + Math.ceil(s.length / 3), 0);
	tokens += keys.length * 1.5;
	tokens += numbers.length;
	tokens += booleans + nulls;
	return Math.max(1, Math.ceil(tokens));
}

function estimateCodeTokens(text: string): number {
	const trimmed = text.trim();
	const words = trimmed.split(/\s+/).filter(Boolean);
	let tokens = 0;
	for (const word of words) {
		if (word.length <= 2) tokens += 1;
		else if (word.length <= 4) tokens += 1.2;
		else tokens += 1 + Math.ceil((word.length - 4) / 3);
	}
	const operators = (trimmed.match(/[{}()[\];,.]/g) || []).length;
	tokens += operators * 0.8;
	return Math.max(1, Math.ceil(tokens));
}

function estimateNaturalLanguageTokens(text: string): number {
	const trimmed = text.trim();
	const words = trimmed.split(/\s+/).filter(Boolean);
	if (words.length === 0) return 0;
	let totalTokens = 0;
	for (const word of words) {
		const multiByte = (
			word.match(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]/g) ||
			[]
		).length;
		const ascii = word.length - multiByte;
		totalTokens += Math.ceil(multiByte / 1.5);
		totalTokens += Math.ceil(ascii / 3.5);
	}
	const sentences = trimmed.split(/[.!?]+/).filter(Boolean).length;
	totalTokens += sentences * 1.2;
	return Math.max(1, Math.ceil(totalTokens));
}

export function estimateMessageTokens(messages: Message[]): number {
	return estimateChatPayloadTokens(messages);
}

export function estimateChatPayloadTokens(
	messages: Message[],
	tools?: Record<string, unknown>[],
): number {
	return estimateTokens(
		JSON.stringify({
			messages: convertToChatFormat(messages),
			tools: tools || [],
		}),
	);
}

export const COMPACTION_TARGET_FRACTION = 0.65;

export interface CompactionResult {
	messages: Message[];
	tokensBefore: number;
	tokensAfter: number;
	changed: boolean;
}

export function microCompactMessages(messages: Message[]): CompactionResult {
	const result = microCompactCompactable(messages);
	return {
		messages: result.messages as Message[],
		tokensBefore: result.tokensBefore,
		tokensAfter: result.tokensAfter,
		changed: result.changed,
	};
}
