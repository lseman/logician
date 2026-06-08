// ── Message handling ──────────────────────────────────────────────────────────────
// Message creation and chat format conversion.

import type { AgentMessage, Message, MessageRole } from "./types.ts";

/** Convert AgentMessage[] to LLM-compatible Message[]. Filters out custom messages. */
export function convertToLlm(messages: AgentMessage[]): Message[] {
	const standardRoles = new Set<MessageRole>([
		"system",
		"user",
		"assistant",
		"tool",
	]);
	return messages.filter((m) =>
		standardRoles.has(m.role as MessageRole),
	) as Message[];
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
	return {
		role: "assistant",
		content: content || null,
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
		return obj;
	});
}

export function estimateTokens(text: string): number {
	return Math.max(1, Math.ceil(text.length / 4));
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

export interface CompactionResult {
	messages: Message[];
	tokensBefore: number;
	tokensAfter: number;
	changed: boolean;
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
	const keep = Math.max(2, keepRecentMessages);
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

	const { system, older, recent } = splitForCompaction(
		contentCompacted,
		options.keepRecentMessages,
	);
	if (!older.length) {
		return {
			messages: contentCompacted,
			tokensBefore,
			tokensAfter: contentTokens,
			changed: contentTokens < tokensBefore,
		};
	}

	const summary = summarizeMessages(older, options.maxSummaryChars || 12000);
	const compacted = [
		...system,
		createUserMessage(
			`<context-compaction reason="context_full">\n${summary}\n</context-compaction>`,
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

/**
 * Token-budget compaction ladder shared by the proactive builtin hook and the
 * loop's context-full recovery. Single source of truth for the
 * estimate → micro → (full if still over) sequence:
 *
 *  1. If already under `triggerTokens` and not `force`d, do nothing.
 *  2. Run the cheap micro pass (trim oversized bodies). If that brings the
 *     payload under `triggerTokens`, stop there.
 *  3. Otherwise run the full summarizing pass targeting `targetTokens`.
 *
 * `triggerTokens` is the threshold that fires compaction (e.g. window * 0.8);
 * `targetTokens` is what the full pass aims to reach (e.g. window * 0.65).
 * `force` skips the under-threshold checks — used when the provider already
 * rejected the request as too long, so compaction must run regardless of the
 * local estimate.
 */
export function compactToFit(
	messages: Message[],
	opts: {
		triggerTokens: number;
		targetTokens?: number;
		toolDefs?: Record<string, unknown>[];
		keepRecentMessages?: number;
	},
): CompactionResult {
	const { triggerTokens, targetTokens, toolDefs, keepRecentMessages } = opts;
	const force = triggerTokens <= 0;
	const estimate = (msgs: Message[]) =>
		estimateChatPayloadTokens(msgs, toolDefs);
	const noop: CompactionResult = {
		messages,
		tokensBefore: estimate(messages),
		tokensAfter: estimate(messages),
		changed: false,
	};

	if (!force && noop.tokensBefore < triggerTokens) return noop;

	// Cheap pass first.
	const micro = microCompactMessages(messages);
	if (!force && estimate(micro.messages) < triggerTokens) return micro;

	// Still over (or forced): full summarizing pass on the micro'd messages.
	return compactMessagesForContext(micro.messages, {
		targetTokens,
		keepRecentMessages,
	});
}

// Cheap standalone pass: truncate only oversized message bodies, leave history
// structure intact. Used for proactive compaction before the full summarizing
// pass is needed.
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
