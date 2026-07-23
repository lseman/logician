// ── Context compaction for long sessions ─────────────────────────────────────────
// Merged system: Pi's turn-boundary cut points + usage-based tracking + branch
// awareness, combined with Logician's file-op tracking and simpler message API.
//
// Key improvements over the original:
// - Turn-boundary-aware cut points (never cuts mid-turn)
// - Provider usage tracking when available (falls back to char estimation)
// - UUID-based entry IDs for cut points (survives message reordering)
// - Branch summarization for conversation divergence
// - Structured summaries (Goal / Constraints / Progress / Decisions / Next Steps)
// - Turn-prefix summarization when cut splits an in-flight turn
// - Usage metrics: tokensBefore / tokensAfter on compaction results

import { randomUUID } from "node:crypto";
import type { AgentMessage, CompactableMessage } from "../core/types.ts";
import { DEFAULT_TRUNCATION } from "../core/types/types-truncation.ts";
import {
	type FileOperations,
	createFileOps,
	extractFileOpsFromMessage,
	computeFileLists,
	formatFileOperations,
	serializeConversation,
} from "./utils";

// ============================================================================
// Types
// ============================================================================

/** Compaction thresholds and retention settings. */
export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
	contextWindow?: number;
	/** Number of recent messages to always preserve (regardless of token budget). */
	protectedMessageCount?: number;
	/** Whether to force compaction regardless of current token usage. */
	force?: boolean;
}

export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
	contextWindow: 128000,
	protectedMessageCount: 3,
	force: false,
};

// ============================================================================
// Token estimation — dual-mode: provider usage when available, char heuristic fallback
// ============================================================================

const ESTIMATED_IMAGE_CHARS = 4800;

/** Estimate tokens for one message using character heuristic. Conservative (overestimates). */
export function estimateCompressableTokens(
	message: AgentMessage | CompactableMessage,
): number {
	const msg = message as AgentMessage & { content?: unknown };
	let chars = 0;

	const content = typeof msg.content === "string" ? msg.content : "";
	const textContent = content || "";

	const role = msg.role as string;
	if (role === "user") {
		chars += textContent.length;
	} else if (role === "assistant") {
		const agentMsg = msg as unknown as AgentMessage & { content?: unknown[] };
		if (Array.isArray(agentMsg.content)) {
			for (const block of agentMsg.content ?? []) {
				if (typeof block === "object" && block !== null) {
					const bo = block as Record<string, unknown>;
					if (bo.type === "text" && typeof bo.text === "string") {
						chars += bo.text.length;
					} else if (
						bo.type === "thinking" &&
						typeof bo.thinking === "string"
					) {
						chars += bo.thinking.length;
					} else if (bo.type === "toolCall" && typeof bo.name === "string") {
						const argsStr =
							typeof bo.arguments === "string"
								? bo.arguments
								: JSON.stringify(bo.arguments ?? {});
						chars += bo.name.length + argsStr.length;
					} else if ((block as { type?: string }).type === "image") {
						chars += ESTIMATED_IMAGE_CHARS;
					}
				}
			}
		} else {
			chars += textContent.length;
		}
	} else if (role === "toolResult" || role === "tool") {
		chars += textContent.length;
	} else if (role === "custom") {
		chars += textContent.length;
	} else if (role === "branchSummary" || role === "compactionSummary") {
		const branchMsg = msg as { summary?: string };
		chars += branchMsg.summary?.length ?? 0;
	} else {
		chars += textContent.length;
	}

	// Use content-aware heuristic instead of naive char/4
	const contentType = estimateCompressableTokens.classifyContentType(content);
	const ratio = estimateCompressableTokens.getRatio(contentType);
	return Math.ceil(chars / ratio);
}

/** Classify content type for token estimation. */
estimateCompressableTokens.classifyContentType = (text: string): string => {
	if (!text || text.length < 16) return "natural";
	const trimmed = text.trim();

	// JSON detection: starts with { or [, balanced braces
	const startsJSON =
		(trimmed.startsWith("{") || trimmed.startsWith("[")) &&
		trimmed.includes(":") &&
		(trimmed.match(/"[^"]*"\s*:/g) || []).length >= 2;
	if (startsJSON) {
		const braceRatio =
			(trimmed.match(/[{}[\]]/g) || []).length /
			trimmed.replace(/[\s\n\r]/g, "").length;
		if (braceRatio > 0.1) return "json";
	}

	// Code detection: known keywords, operators, patterns
	const codeKeywords = [
		"function|const|let|var|class|import|export|return|if|else|for|while",
		"def |async |await |yield |lambda |struct|enum|interface",
		"public|private|protected|static|void|extends|implements",
		"try|catch|finally|throw|new|this|super|instanceof",
	];
	const codePattern = new RegExp(codeKeywords.join("|"), "i");
	const codeRatio =
		(
			trimmed.match(
				/\b(function|const|let|var|class|import|export|return|if|else|for|while|def |async |await)\b/gi,
			) || []
		).length / Math.max(1, trimmed.split(/\s+/).length);
	if (codePattern.test(trimmed) || codeRatio > 0.05) return "code";

	// Multi-byte detection for natural language
	const multiByte = (trimmed.match(/[\u0080-\uFFFF]/g) || []).length;
	const multiByteRatio = multiByte / trimmed.length;

	if (multiByteRatio > 0.3) return "unicode";
	if (multiByteRatio > 0.05) return "natural-unicode";
	return "natural";
};

/** Get token-to-char ratio for content type. */
estimateCompressableTokens.getRatio = (type: string): number => {
	switch (type) {
		case "json":
			return 1.5;
		case "code":
			return 2;
		case "unicode":
			return 1.5;
		case "natural-unicode":
			return 2.5;
		default:
			return 3.5;
	}
};

/** Estimated context-token usage for a message list. */
export interface ContextUsageEstimate {
	tokens: number;
	usageTokens: number;
	trailingTokens: number;
	lastUsageIndex: number | null;
}

/** Estimate context tokens using provider usage (when available) + estimation. */
export function estimateContextTokens(
	messages: CompactableMessage[],
): ContextUsageEstimate {
	// Try to find provider-reported usage from the last assistant message
	let usageTokens = 0;
	let lastUsageIndex: number | null = null;

	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i] as AgentMessage & {
			usage?: Record<string, number>;
		};
		if (msg.role === "assistant" && msg.usage) {
			usageTokens =
				msg.usage.totalTokens ??
				(msg.usage.input || 0) +
					(msg.usage.output || 0) +
					(msg.usage.cacheRead || 0) +
					(msg.usage.cacheWrite || 0);
			lastUsageIndex = i;
			break;
		}
	}

	if (lastUsageIndex !== null && usageTokens > 0) {
		// Usage-based: provider gave us exact token count up to this message
		let trailingTokens = 0;
		for (let i = lastUsageIndex + 1; i < messages.length; i++) {
			trailingTokens += estimateCompressableTokens(messages[i]);
		}
		return {
			tokens: usageTokens + trailingTokens,
			usageTokens,
			trailingTokens,
			lastUsageIndex,
		};
	}

	// Fallback: full char-based estimation
	let estimated = 0;
	for (const msg of messages) {
		estimated += estimateCompressableTokens(
			msg as AgentMessage | CompactableMessage,
		);
	}
	return {
		tokens: estimated,
		usageTokens: 0,
		trailingTokens: estimated,
		lastUsageIndex: null,
	};
}

// ============================================================================
// Compaction trigger
// ============================================================================

/** Check if compaction should trigger. */
export function shouldCompact(
	messages: CompactableMessage[],
	settings: CompactionSettings,
): boolean {
	if (!settings.enabled) return false;
	const contextTokens = estimateContextTokens(messages).tokens;
	const threshold = (settings.contextWindow ?? 128000) - settings.reserveTokens;
	return contextTokens > threshold;
}

// ============================================================================
// Cut point detection — turn-boundary-aware (never cuts mid-turn)
// ============================================================================

/** Valid cut-point positions in a message list (user message boundaries). */
function findValidCutPoints(
	messages: CompactableMessage[],
	startIndex: number,
	endIndex: number,
): number[] {
	const cutPoints: number[] = [];
	for (let i = startIndex; i < endIndex; i++) {
		const role = messages[i].role;
		if (
			role === "user" ||
			role === "custom" ||
			role === "branchSummary" ||
			role === "compactionSummary"
		) {
			cutPoints.push(i);
		}
		// Never cut inside tool results — they belong to the assistant's turn
	}
	return cutPoints;
}

/** Find the user-visible message that starts the turn containing a given index. */
function findTurnStartIndex(
	messages: CompactableMessage[],
	entryIndex: number,
	startIndex: number,
): number {
	for (let i = entryIndex; i >= startIndex; i--) {
		const role = messages[i].role;
		if (
			role === "custom" ||
			role === "branchSummary" ||
			role === "compactionSummary"
		) {
			return i;
		}
		if (role === "user") {
			return i;
		}
	}
	return -1;
}

/** Cut point result for compaction. */
export interface CutPointResult {
	firstKeptIndex: number;
	/** UUID of the first kept entry (set when messages carry entryId). */
	firstKeptEntryId?: string;
	turnStartIndex: number; // -1 if cut is clean (at user message)
	isSplitTurn: boolean;
	/** Index of the first protected message (system prompt boundary). */
	protectedStartIndex: number;
	/** Index of the last protected message (recent messages boundary). */
	protectedEndIndex: number;
}

/** Find the compaction cut point keeping approximately keepRecentTokens from the end. */
function findCutPoint(
	messages: CompactableMessage[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
): CutPointResult {
	// Assign entry IDs to messages that don't have them
	for (const msg of messages) {
		const m = msg as CompactableMessage & { entryId?: string };
		if (!m.entryId) {
			m.entryId = randomUUID();
		}
	}
	const cutPoints = findValidCutPoints(messages, startIndex, endIndex);

	if (cutPoints.length === 0) {
		return {
			firstKeptIndex: startIndex,
			turnStartIndex: -1,
			isSplitTurn: false,
			protectedStartIndex: startIndex,
			protectedEndIndex: endIndex,
		};
	}

	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0];

	// Walk backwards accumulating tokens
	for (let i = endIndex - 1; i >= startIndex; i--) {
		const msgTokens = estimateCompressableTokens(messages[i]);
		accumulatedTokens += msgTokens;

		if (accumulatedTokens >= keepRecentTokens) {
			// Find the nearest valid cut point >= this position
			for (const cp of cutPoints) {
				if (cp >= i) {
					cutIndex = cp;
					break;
				}
			}
			break;
		}
	}

	// Walk backward past non-message entries (metadata, labels, etc.)
	while (cutIndex > startIndex) {
		const role = messages[cutIndex - 1].role;
		if (role === "compactionSummary" || role === "branchSummary") {
			break;
		}
		if (role === "user" || role === "assistant" || role === "toolResult") {
			break;
		}
		cutIndex--;
	}

	const isUserMessage = messages[cutIndex].role === "user";
	const turnStartIndex = isUserMessage
		? -1
		: findTurnStartIndex(messages, cutIndex, startIndex);

	return {
		firstKeptIndex: cutIndex,
		firstKeptEntryId: messages[cutIndex]?.entryId,
		turnStartIndex,
		isSplitTurn: !isUserMessage && turnStartIndex !== -1,
		protectedStartIndex: startIndex,
		protectedEndIndex: endIndex,
	};
}

// ============================================================================
// compactToFit — bridges Message[] format to the new compaction system
// ============================================================================

/** Result of compactToFit, compatible with messages.ts CompactionResult. */
export interface CompactToFitResult {
	messages: CompactableMessage[];
	tokensBefore: number;
	tokensAfter: number;
	changed: boolean;
}

/**
 * Token-budget compaction using the new turn-aware system.
 * Sync version: uses synchronous full compaction with inline structured summary.
 * For LLM-based summarization, use the async `compact()` function directly.
 *
 * Sequence:
 *  1. If already under `triggerTokens` and not `force`d, do nothing.
 *  2. Run the cheap micro pass (trim oversized bodies). If that brings the
 *     payload under `triggerTokens`, stop there.
 *  3. Otherwise run the full summarizing pass targeting `targetTokens`.
 */
export function compactToFit(
	messages: CompactableMessage[],
	opts: {
		triggerTokens: number;
		targetTokens?: number;
		keepRecentMessages?: number;
		settings?: Partial<CompactionSettings>;
	},
): CompactToFitResult {
	const { triggerTokens, keepRecentMessages, settings } = opts;
	const force = triggerTokens <= 0;

	// Estimate current tokens
	const estimate = () => estimateContextTokens(messages).tokens;
	const tokensBefore = estimate();

	const effectiveSettings: CompactionSettings = {
		enabled: true,
		reserveTokens:
			settings?.reserveTokens ?? DEFAULT_COMPACTION_SETTINGS.reserveTokens,
		keepRecentTokens:
			keepRecentMessages ?? DEFAULT_COMPACTION_SETTINGS.keepRecentTokens,
		contextWindow:
			settings?.contextWindow ?? DEFAULT_COMPACTION_SETTINGS.contextWindow,
		...settings,
	};

	if (!force && tokensBefore < triggerTokens) {
		return {
			messages,
			tokensBefore,
			tokensAfter: tokensBefore,
			changed: false,
		};
	}

	// Cheap pass: micro-compact (trim oversized bodies)
	const micro = microCompactMessages(messages);
	const microTokens = estimateContextTokens(micro.messages).tokens;
	if (!force && microTokens < triggerTokens) {
		return { ...micro, changed: micro.tokensAfter < micro.tokensBefore };
	}

	// Full synchronous summarizing pass
	return compactToFitSync(micro.messages, effectiveSettings);
}

// How many trailing messages micro-compaction leaves untouched — the model is
// usually still acting on them.
const MICRO_COMPACT_KEEP_RECENT = 6;

export function microCompactMaxChars(role: string): number {
	// Tool results tolerate the most trimming; user prompts the least — losing
	// part of the task statement is worse than a long context.
	const limits = DEFAULT_TRUNCATION.microCompactMaxChars;
	if (role === "tool") return limits.tool;
	if (role === "assistant") return limits.assistant;
	return limits.default;
}

export function truncateMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 32) / 2));
	return `${text.slice(0, half)}\n...[compacted ${text.length - half * 2} chars]...\n${text.slice(-half)}`;
}

export function microCompactMessages(
	messages: CompactableMessage[],
): CompactToFitResult {
	const tokensBefore = estimateContextTokens(messages).tokens;
	// Micro-compaction: trim oversized bodies, role-aware. Recent messages are
	// left intact; middle-truncation keeps both the head and the tail of long
	// bodies (tool output tails often carry the error/summary).
	const trimmed = messages.map((m, index) => {
		if (index >= messages.length - MICRO_COMPACT_KEEP_RECENT) return m;
		if (typeof m.content !== "string") return m;
		const maxChars = microCompactMaxChars(m.role);
		if (m.content.length <= maxChars) return m;
		return { ...m, content: truncateMiddle(m.content, maxChars) };
	});
	const tokensAfter = estimateContextTokens(trimmed).tokens;
	return {
		messages: trimmed,
		tokensBefore,
		tokensAfter,
		changed: tokensAfter < tokensBefore,
	};
}

function compactToFitSync(
	messages: CompactableMessage[],
	settings: CompactionSettings,
): CompactToFitResult {
	// Find the cut point using turn-boundary-aware logic
	const cutPoint = findCutPoint(
		messages,
		0,
		messages.length,
		settings.keepRecentTokens,
	);

	if (cutPoint.firstKeptIndex >= messages.length) {
		return {
			messages,
			tokensBefore: estimateContextTokens(messages).tokens,
			tokensAfter: estimateContextTokens(messages).tokens,
			changed: false,
		};
	}

	const messagesToKeep = messages.slice(cutPoint.firstKeptIndex);

	// Generate a structured inline summary of the compacted portion
	const messagesToSummarize = messages.slice(0, cutPoint.firstKeptIndex);
	const summary = generateInlineSummary(messagesToSummarize, settings);

	// Build compacted message list
	const compacted: CompactableMessage[] = [
		{ role: "compactionSummary", content: summary },
		...messagesToKeep,
	];

	return {
		messages: compacted,
		tokensBefore: estimateContextTokens(messages).tokens,
		tokensAfter: estimateContextTokens(compacted).tokens,
		changed: true,
	};
}

function generateInlineSummary(
	messages: CompactableMessage[],
	settings: CompactionSettings,
): string {
	const conversationText = serializeConversation(
		messages as Array<{
			role: string;
			content: string | Array<{ type: string; text?: string }>;
		}>,
	);

	const tokenBudget =
		(settings.contextWindow ?? 128000) - settings.reserveTokens;
	const maxSummaryChars = Math.min(
		2000,
		Math.max(200, Math.floor(tokenBudget * 0.3)),
	);

	// Generate a structured summary (Goal/Progress/Next Steps format)
	// This is a synchronous inline summary — for LLM-based quality, use `compact()` directly
	let summary = `<context-compaction reason="auto">
## Goal
[Auto-generated context checkpoint]

## Progress
### In Progress
- [Session compacted — context preserved below]

## Next Steps
[Continue from the retained recent context]
</context-compaction>`;

	// Append truncated conversation for context
	if (conversationText.length > maxSummaryChars) {
		summary += `\n\n[Conversation context (truncated to ${maxSummaryChars} chars):]\n${conversationText.slice(0, maxSummaryChars)}`;
	} else {
		summary += `\n\n${conversationText}`;
	}

	return summary;
}
