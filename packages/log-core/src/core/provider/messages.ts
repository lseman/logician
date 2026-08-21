import {
	microCompactCompactableMessages,
	pruneHistoricalToolOutputs,
	type PrunedToolOutputsResult,
	type PruneHistoricalToolOutputsOptions,
} from "../compaction/engine.ts";
import type {
	AgentMessage,
	BashExecutionMessage,
	BranchSummaryMessage,
	CompactionSummaryMessage,
	CustomMessage,
	Message,
} from "../types/types-messages.ts";

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

/** Convert AgentMessage[] to LLM-compatible Message[]. Handles custom message types. */
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

/**
 * Replace unparseable `arguments` strings with "{}" before persisting to
 * history. A call gets truncated mid-argument when the completion hits the
 * output token limit (stopReason "length"); if that raw string is ever saved,
 * every future turn resends unparseable JSON and the backend fails on it
 * forever — history can't be fixed retroactively once saved. This repairs
 * rather than drops the call so its `id` still matches the tool-result
 * message already generated for it elsewhere (e.g. the executor's own
 * "not executed, truncated" result) — dropping it would orphan that result
 * and trip the provider's tool_call/tool_result pairing check instead.
 */
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
	// An assistant message must have either content or tool_calls or the API
	// rejects it with 400 "Assistant message must contain either 'content' or
	// 'tool_calls'!".  When the model returns nothing, keep content as a
	// non-empty string so the message is still valid and the loop can recover
	// (compact, nudge, or abort) instead of looping forever on the same error.
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
			// Defensive: an assistant message must have at least 'content' or
			// 'tool_calls' or the API rejects it with 400.  createAssistantMessage
			// already prevents this, but guard here too so any stray message
			// doesn't crash the loop.
			if (m.role === "assistant" && !obj.content && !obj.tool_calls) {
				obj.content = "";
			}
			return obj;
		});
}

export function estimateTokens(text: string): number {
	if (!text || text.length === 0) return 0;

	// ── Detect content type and apply appropriate tokenizer ──────────────

	// JSON-heavy content (tool definitions, API responses, structured data)
	if (isJsonLike(text)) {
		return estimateJsonTokens(text);
	}

	// Code-heavy content (source code, scripts)
	if (isCodeLike(text)) {
		return estimateCodeTokens(text);
	}

	// Natural language (prompts, summaries, conversation text)
	return estimateNaturalLanguageTokens(text);
}

/** Check if text looks like JSON or structured data. */
function isJsonLike(text: string): boolean {
	const trimmed = text.trim();
	// Must start with { or [ and contain common JSON markers
	if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return false;
	// High proportion of JSON-special characters
	const jsonChars = (text.match(/[{}[\]:,]/g) || []).length;
	return jsonChars / Math.max(1, text.length) > 0.05;
}

/** Check if text looks like code. */
function isCodeLike(text: string): boolean {
	const trimmed = text.trim();
	if (trimmed.length < 20) return false;
	// Look for code patterns: keywords, operators, braces
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

/** Estimate tokens for JSON/structured data. */
function estimateJsonTokens(text: string): number {
	const trimmed = text.trim();
	// Count JSON structural elements and string values
	const strings = trimmed.match(/"[^"\\]*(?:\\.[^"\\]*)*"/g) || [];
	const keys = trimmed.match(/"[^"]+"\s*:/g) || [];
	const numbers = trimmed.match(/\b\d+(?:\.\d+)?\b/g) || [];
	const booleans = (trimmed.match(/\b(true|false)\b/g) || []).length;
	const nulls = (trimmed.match(/\bnull\b/g) || []).length;

	// Each string value ≈ 1 token + chars/3 for the content
	let tokens = strings.reduce((sum, s) => sum + 1 + Math.ceil(s.length / 3), 0);
	// Keys count as separate tokens
	tokens += keys.length * 1.5;
	// Numbers count as 1 token each
	tokens += numbers.length;
	// Booleans and nulls count as 1 token each
	tokens += booleans + nulls;
	// Structural chars are mostly covered by the above
	return Math.max(1, Math.ceil(tokens));
}

/** Estimate tokens for code content. */
function estimateCodeTokens(text: string): number {
	const trimmed = text.trim();
	// Code has more whitespace and multi-char operators
	// Split by whitespace and count tokens per "word"
	const words = trimmed.split(/\s+/).filter(Boolean);
	let tokens = 0;
	for (const word of words) {
		if (word.length <= 2) {
			tokens += 1; // short words/operators
		} else if (word.length <= 4) {
			tokens += 1.2; // common identifiers
		} else {
			tokens += 1 + Math.ceil((word.length - 4) / 3); // longer identifiers
		}
	}
	// Add tokens for punctuation and operators
	const operators = (trimmed.match(/[{}()[\];,.]/g) || []).length;
	tokens += operators * 0.8;
	return Math.max(1, Math.ceil(tokens));
}

/** Estimate tokens for natural language text. */
function estimateNaturalLanguageTokens(text: string): number {
	const trimmed = text.trim();
	// Split into words, handling multi-byte characters
	const words = trimmed.split(/\s+/).filter(Boolean);
	if (words.length === 0) return 0;

	// BPE tokenizers typically produce ~1 token per 4 chars for English
	// but ~1 token per 2-3 chars for multi-byte scripts
	let totalTokens = 0;
	for (const word of words) {
		// Count multi-byte characters
		const multiByte = (
			word.match(/[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]/g) ||
			[]
		).length;
		const ascii = word.length - multiByte;

		// Multi-byte chars: ~1 token per 1-2 chars
		// ASCII chars: ~1 token per 3-4 chars
		totalTokens += Math.ceil(multiByte / 1.5);
		totalTokens += Math.ceil(ascii / 3.5);
	}

	// Add sentence-level tokens (each sentence has ~1-2 extra tokens)
	const sentences = trimmed.split(/[.!?]+/).filter(Boolean).length;
	totalTokens += sentences * 1.2;

	return Math.max(1, Math.ceil(totalTokens));
}

// All token estimates use the same basis (serialized chat payload) so that
// budgets and compaction before/after deltas are directly comparable.
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

// Shared compaction target: summarizing compaction aims to bring the payload
// down to this fraction of the context window. Referenced by the proactive
// builtin hook, the loop's context-full path, and the harness's manual compact.
export const COMPACTION_TARGET_FRACTION = 0.65;

export interface CompactionResult {
	messages: Message[];
	tokensBefore: number;
	tokensAfter: number;
	changed: boolean;
}

// Delegates to the single micro-compaction implementation in
// compaction/engine.ts (role-aware caps, keeps recent messages intact).
export function microCompactMessages(messages: Message[]): CompactionResult {
	const result = microCompactCompactableMessages(messages);
	return {
		messages: result.messages as Message[],
		tokensBefore: result.tokensBefore,
		tokensAfter: result.tokensAfter,
		changed: result.changed,
	};
}

export {
	pruneHistoricalToolOutputs,
	type PrunedToolOutputsResult,
	type PruneHistoricalToolOutputsOptions,
};
