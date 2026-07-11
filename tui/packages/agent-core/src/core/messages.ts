// ── Message handling ──────────────────────────────────────────────────────────────
// Message creation, chat format conversion, and compaction integration.

import type {
	AgentMessage,
	BashExecutionMessage,
	BranchSummaryMessage,
	CompactionSummaryMessage,
	CustomMessage,
	Message,
	CompactableMessage,
} from "./types.ts";

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
	return messages.map((m) => {
		const obj: Record<string, unknown> = { role: m.role };
		if (m.content !== null && m.content !== undefined) {
			obj.content = m.content;
		}
		if (m.tool_call_id) obj.tool_call_id = m.tool_call_id;
		if (m.tool_calls?.length) {
			obj.tool_calls = m.tool_calls.map((tc) => ({
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

// Keep this many most-recent non-system messages verbatim during compaction.
const DEFAULT_KEEP_RECENT = 8;

/** Compaction thresholds and retention settings. */
export interface CompactionSettings {
	/** Enable automatic compaction decisions. */
	enabled: boolean;
	/** Tokens reserved for summary prompt and output. */
	reserveTokens: number;
	/** Approximate recent-context tokens to keep after compaction. */
	keepRecentTokens: number;
}

/** Default compaction settings used by the harness. */
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
};

/** Return whether context usage exceeds the configured compaction threshold. */
export function shouldCompact(
	contextTokens: number,
	contextWindow: number,
	settings: CompactionSettings,
): boolean {
	if (!settings.enabled) return false;
	return contextTokens > contextWindow - settings.reserveTokens;
}

export interface CompactionResult {
	messages: Message[];
	tokensBefore: number;
	tokensAfter: number;
	changed: boolean;
	/** File operations in the compacted history. */
	fileOps?: FileOperations;
}

/**
 * Split messages into the system prefix, the older block to summarize, and the
 * recent block to keep verbatim. Tail boundary is nudged back so a tool result
 * is never separated from its preceding assistant call.
 */
export function splitForCompaction(
	messages: Message[],
	keepRecentMessages = DEFAULT_KEEP_RECENT,
): { system: Message[]; older: Message[]; recent: Message[] } {
	// One recent message is sufficient unless boundary repair below pulls in
	// its assistant tool-call parent. Requiring two made tight target budgets
	// impossible even when callers explicitly requested a one-message tail.
	const keep = Math.max(1, keepRecentMessages);
	const system = messages.filter((m) => m.role === "system");
	const nonSystem = messages.filter((m) => m.role !== "system");
	const tailStart = adjustTailStartForToolPairs(
		nonSystem,
		Math.max(0, nonSystem.length - keep),
	);
	return {
		system,
		older: nonSystem.slice(0, tailStart),
		recent: nonSystem.slice(tailStart),
	};
}

/**
 * Summarizing compaction skeleton shared by manual and automatic compaction.
 * `summarize` produces the replacement summary text for the older block; pass
 * the LLM summarizer for manual compaction, or omit for the local text
 * summarizer. Returns the compacted messages, or the micro-compacted messages
 * when there is nothing old enough to summarize.
 */
export async function compactMessages(
	messages: Message[],
	options: {
		reason: string;
		keepRecentMessages?: number;
		maxSummaryChars?: number;
		summarize?: (older: Message[], system: Message[]) => Promise<string | null>;
	},
): Promise<CompactionResult> {
	const tokensBefore = estimateMessageTokens(messages);
	const { system, older, recent } = splitForCompaction(
		messages,
		options.keepRecentMessages,
	);

	// Nothing old enough to summarize — just trim oversized bodies.
	if (!older.length) return microCompactMessages(messages);

	const summary = options.summarize
		? await options.summarize(older, system)
		: summarizeMessages(older, options.maxSummaryChars || 12000);

	// Summarizer failed — fall back to local truncation of the older block.
	if (!summary) {
		return microCompactMessages(messages);
	}

	const compacted = [
		...system,
		createUserMessage(
			`<context-compaction reason="${options.reason}">\n${summary}\n</context-compaction>`,
		),
		...recent,
	];
	const tokensAfter = estimateMessageTokens(compacted);
	return {
		messages: compacted,
		tokensBefore,
		tokensAfter,
		changed: tokensAfter < tokensBefore,
	};
}

export function compactMessagesForContext(
	messages: Message[],
	options: {
		targetTokens?: number;
		keepRecentMessages?: number;
		maxSummaryChars?: number;
	} = {},
): CompactionResult {
	const tokensBefore = estimateMessageTokens(messages);
	// First trim oversized bodies, then summarize the older block of the result.
	const contentCompacted = messages.map((message) =>
		compactLargeMessageContent(message),
	);
	const contentTokens = estimateMessageTokens(contentCompacted);

	// Extract file operations from all messages
	const fileOps = createFileOps();
	for (const msg of contentCompacted) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	if (!options.targetTokens && contentTokens < tokensBefore) {
		// Preserve the legacy fast path when simple body trimming is sufficient.
		const initial = splitForCompaction(
			contentCompacted,
			options.keepRecentMessages,
		);
		if (!initial.older.length) {
			return {
				messages: contentCompacted,
				tokensBefore,
				tokensAfter: contentTokens,
				changed: true,
				fileOps:
					fileOps.read.size > 0 ||
					fileOps.written.size > 0 ||
					fileOps.edited.size > 0
						? fileOps
						: undefined,
			};
		}
	}

	const requestedKeep = Math.max(1, options.keepRecentMessages ?? 8);
	const keepCandidates = Array.from(
		new Set(
			[requestedKeep, 6, 4, 2, 1].filter((value) => value <= requestedKeep),
		),
	);
	// Small context windows need summaries smaller than 500 characters. The
	// floor is deliberately low; the ladder still prefers richer summaries.
	const requestedSummaryChars = Math.max(120, options.maxSummaryChars ?? 12000);
	const summaryCandidates = Array.from(
		new Set(
			[requestedSummaryChars, 6000, 3000, 1500, 750, 500, 250, 120].filter(
				(value) => value <= requestedSummaryChars,
			),
		),
	);
	let best: CompactionResult | undefined;

	for (const keepRecentMessages of keepCandidates) {
		for (const maxSummaryChars of summaryCandidates) {
			const { system, older, recent } = splitForCompaction(
				contentCompacted,
				keepRecentMessages,
			);
			if (!older.length) continue;
			const summary = summarizeMessages(older, maxSummaryChars);
			const compacted = [
				...system,
				createUserMessage(
					`<context-compaction reason="context_full">\n${summary}\n</context-compaction>`,
				),
				...recent,
			];
			const candidate: CompactionResult = {
				messages: compacted,
				tokensBefore,
				tokensAfter: estimateMessageTokens(compacted),
				changed: estimateMessageTokens(compacted) < tokensBefore,
			};
			if (!best || candidate.tokensAfter < best.tokensAfter) best = candidate;
			if (
				options.targetTokens === undefined ||
				candidate.tokensAfter <= options.targetTokens
			) {
				return {
					...candidate,
					fileOps:
						fileOps.read.size > 0 ||
						fileOps.written.size > 0 ||
						fileOps.edited.size > 0
							? fileOps
							: undefined,
				};
			}
		}
	}

	if (!best) {
		return {
			messages: contentCompacted,
			tokensBefore,
			tokensAfter: contentTokens,
			changed: contentTokens < tokensBefore,
			fileOps:
				fileOps.read.size > 0 ||
				fileOps.written.size > 0 ||
				fileOps.edited.size > 0
					? fileOps
					: undefined,
		};
	}
	return {
		...best,
		fileOps:
			fileOps.read.size > 0 ||
			fileOps.written.size > 0 ||
			fileOps.edited.size > 0
				? fileOps
				: undefined,
	};
}

export function microCompactMessages(messages: Message[]): CompactionResult {
	const tokensBefore = estimateMessageTokens(messages);
	const out = messages.map((message) => compactLargeMessageContent(message));
	const tokensAfter = estimateMessageTokens(out);
	return {
		messages: out,
		tokensBefore,
		tokensAfter,
		changed: tokensAfter < tokensBefore,
	};
}

function compactLargeMessageContent(message: Message): Message {
	if (typeof message.content !== "string") return message;
	const maxChars =
		message.role === "tool"
			? 6000
			: message.role === "assistant"
				? 10000
				: 14000;
	if (message.content.length <= maxChars) return message;
	return {
		...message,
		content: truncateMiddle(message.content, maxChars),
	};
}

function adjustTailStartForToolPairs(
	messages: Message[],
	start: number,
): number {
	let adjusted = start;
	while (adjusted > 0 && messages[adjusted]?.role === "tool") {
		adjusted--;
	}
	return adjusted;
}

function summarizeMessages(messages: Message[], maxChars: number): string {
	const lines = [
		"The earlier conversation was compacted automatically because the model context was full.",
		"Important prior messages and tool results are summarized below.",
		"",
	];

	for (const message of messages) {
		if (message.role === "system") continue;
		const label =
			message.role === "tool"
				? `tool:${message.name || "unknown"}`
				: message.role;
		const text = messageToText(message);
		if (!text) continue;
		lines.push(`## ${label}`);
		lines.push(truncateMiddle(text, 1800));
		if (message.tool_calls?.length) {
			lines.push(
				`Tool calls: ${message.tool_calls
					.map(
						(tool) =>
							`${tool.name}(${truncateMiddle(tool.arguments || "{}", 500)})`,
					)
					.join("; ")}`,
			);
		}
		lines.push("");
	}

	return truncateMiddle(lines.join("\n"), maxChars);
}

function messageToText(message: Message): string {
	if (typeof message.content === "string") return message.content.trim();
	if (message.content === null || message.content === undefined) return "";
	return String(message.content).trim();
}

function truncateMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 32) / 2));
	return `${text.slice(0, half)}\n...[compacted ${text.length - half * 2} chars]...\n${text.slice(-half)}`;
}

// ── File operations tracking ──────────────────────────────────────────

export interface FileOperations {
	read: Set<string>;
	written: Set<string>;
	edited: Set<string>;
}

export function createFileOps(): FileOperations {
	return { read: new Set(), written: new Set(), edited: new Set() };
}

export function extractFileOpsFromMessage(
	message: Message | CompactableMessage,
	fileOps: FileOperations,
): void {
	const text =
		typeof message.content === "string"
			? message.content
			: String(message.content || "");

	// Extract read file operations from read_file tool calls/results
	const readMatches = text.matchAll(/read_file\s*[(\s]*["']([^"']+)["']/gi);
	for (const match of readMatches) {
		fileOps.read.add(match[1]);
	}

	// Extract write/edit file operations from write_file/edit_file tool calls/results
	const writeMatches = text.matchAll(/write_file\s*[(\s]*["']([^"']+)["']/gi);
	for (const match of writeMatches) {
		fileOps.written.add(match[1]);
	}

	const editMatches = text.matchAll(/edit_file\s*[(\s]*["']([^"']+)["']/gi);
	for (const match of editMatches) {
		fileOps.edited.add(match[1]);
	}
}

// ── Structured compaction summary ───────────────────────────────────

export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

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

export const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;

export interface SummaryGenerationResult {
	summary: string;
	tokensBefore: number;
}

/** Serialize conversation to text for summarization. */
export function serializeConversation(messages: Message[]): string {
	const lines: string[] = [];
	for (const msg of messages) {
		const role = msg.role;
		const text = messageToText(msg);
		if (!text) continue;
		lines.push(`[${role}]: ${truncateMiddle(text, 2000)}`);
	}
	return lines.join("\n\n");
}
