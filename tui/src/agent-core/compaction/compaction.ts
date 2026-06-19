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
import {
	SUMMARIZATION_PROMPT,
	TURN_PREFIX_SUMMARIZATION_PROMPT,
	UPDATE_SUMMARIZATION_PROMPT,
	CREATE_FILE_OPS,
	EXTRACT_FILE_OPS_FROM_MESSAGE,
	COMPUTE_FILE_LISTS,
	FORMAT_FILE_OPERATIONS,
	SUMMARIZATION_SYSTEM_PROMPT,
	serializeConversation,
} from "./utils";

// Re-export for branch summarization
export { SUMMARIZATION_SYSTEM_PROMPT };

// ============================================================================
// Types
// ============================================================================

/** File-operation details stored on generated compaction entries. */
export interface CompactionDetails {
	readFiles: string[];
	modifiedFiles: string[];
}

/** Compaction result ready to be persisted. */
export interface CompactionResult<T = unknown> {
	summary: string;
	/** Array index of the first kept message (legacy, for direct array slicing). */
	firstKeptIndex: number;
	/** UUID of the first kept entry (survives message reordering, Pi-compatible). */
	firstKeptEntryId?: string;
	/** Estimated tokens after compaction. */
	tokensAfter: number;
	/** Tokens before compaction. */
	tokensBefore: number;
	messagesToKeep: CompactableMessage[];
	details?: T;
}

/** Compaction thresholds and retention settings. */
export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
	contextWindow?: number;
}

export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
	contextWindow: 128000,
};

// ============================================================================
// Token estimation — dual-mode: provider usage when available, char heuristic fallback
// ============================================================================

const ESTIMATED_CHARS_PER_TOKEN = 4;
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
	switch (role) {
		case "user":
			chars += textContent.length;
			break;
		case "assistant": {
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
			break;
		}
		case "toolResult":
		case "tool":
			chars += textContent.length;
			break;
		case "custom":
			chars += textContent.length;
			break;
		case "branchSummary":
		case "compactionSummary": {
			const branchMsg = msg as { summary?: string };
			chars += branchMsg.summary?.length ?? 0;
			break;
		}
		default:
			chars += textContent.length;
	}

	return Math.ceil(chars / ESTIMATED_CHARS_PER_TOKEN);
}

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
		estimated += estimateCompressableTokens(msg as AgentMessage | CompactableMessage);
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
	};
}

// ============================================================================
// Summarization
// ============================================================================

/** Generate or update a conversation summary for compaction. */
export async function generateSummary(
	messages: CompactableMessage[],
	previousSummary?: string,
	customInstructions?: string,
): Promise<string> {
	const llmMessages = convertMessagesToLlmFormat(messages);
	const conversationText = serializeConversation(
		llmMessages as Array<{
			role: string;
			content: string | Array<{ type: string; text?: string }>;
		}>,
	);
	let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;

	if (previousSummary) {
		promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
		promptText += UPDATE_SUMMARIZATION_PROMPT;
	} else {
		promptText += SUMMARIZATION_PROMPT;
	}

	if (customInstructions) {
		promptText += `\n\nAdditional focus: ${customInstructions}`;
	}

	// This is where an LLM call would be made. Returns a placeholder here.
	// In production: convert to LLM format, build prompt, call provider, parse response.
	return `[Summary would be generated here by calling the LLM with the conversation above.\n\nPrompt length: ${promptText.length} chars]`;
}

/** Convert message array to LLM-compatible format for serialization. */
function convertMessagesToLlmFormat(messages: CompactableMessage[]): Array<{
	role: string;
	content: string | Array<{ type: string; text?: string }>;
}> {
	const result: Array<{
		role: string;
		content: string | Array<{ type: string; text?: string }>;
	}> = [];

	for (const msg of messages) {
		if (msg.role === "assistant" && Array.isArray(msg.content)) {
			const textParts = msg.content
				.filter(
					(b: unknown) =>
						typeof b === "object" &&
						b !== null &&
						"type" in b &&
						b.type === "text",
				)
				.map((b: unknown) => ({
					type: "text" as const,
					text: (b as { text: string }).text,
				}));
			result.push({
				role: msg.role,
				content: textParts.length > 0 ? textParts : "",
			});
		} else {
			result.push({
				role: msg.role,
				content: typeof msg.content === "string" ? msg.content : "",
			});
		}
	}

	return result;
}

// ============================================================================
// Prepare compaction
// ============================================================================

/** Prepared inputs for a compaction run. */
interface CompactionPreparation {
	messagesToSummarize: CompactableMessage[];
	messagesToKeep: CompactableMessage[];
	turnPrefixMessages?: CompactableMessage[];
	isSplitTurn: boolean;
	tokensBefore: number;
	cutPoint: CutPointResult;
	previousSummary?: string;
	fileOps: ReturnType<typeof CREATE_FILE_OPS>;
	settings: CompactionSettings;
}

/** Prepare session messages for compaction. */
function prepareCompaction(
	messages: CompactableMessage[],
	settings: CompactionSettings,
	previousSummary?: string,
): CompactionPreparation | undefined {
	if (messages.length === 0) return undefined;

	// If the last entry is already a compaction, nothing to do
	const lastRole = messages[messages.length - 1].role;
	if (lastRole === "compactionSummary") return undefined;

	// Find the boundary (start of current compaction range)
	let boundaryStart = 0;
	if (previousSummary) {
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === "compactionSummary") {
				boundaryStart = i + 1;
				break;
			}
		}
	}

	const tokensBefore = estimateContextTokens(messages).tokens;
	const cutPoint = findCutPoint(
		messages,
		boundaryStart,
		messages.length,
		settings.keepRecentTokens,
	);

	const firstKeptIndex = cutPoint.firstKeptIndex;
	const historyEnd = cutPoint.isSplitTurn
		? cutPoint.turnStartIndex >= 0
			? cutPoint.turnStartIndex
			: firstKeptIndex
		: firstKeptIndex;

	const messagesToSummarize: CompactableMessage[] = [];
	for (let i = boundaryStart; i < historyEnd; i++) {
		messagesToSummarize.push(messages[i]);
	}

	const messagesToKeep: CompactableMessage[] = [];
	for (let i = firstKeptIndex; i < messages.length; i++) {
		messagesToKeep.push(messages[i]);
	}

	const turnPrefixMessages: CompactableMessage[] | undefined =
		cutPoint.isSplitTurn
			? cutPoint.turnStartIndex >= 0
				? messages.slice(cutPoint.turnStartIndex, firstKeptIndex)
				: undefined
			: undefined;

	// Extract file operations
	const fileOps = CREATE_FILE_OPS();
	for (const msg of messagesToSummarize) {
		EXTRACT_FILE_OPS_FROM_MESSAGE(msg, fileOps);
	}
	if (turnPrefixMessages) {
		for (const msg of turnPrefixMessages) {
			EXTRACT_FILE_OPS_FROM_MESSAGE(msg, fileOps);
		}
	}

	return {
		messagesToSummarize,
		messagesToKeep,
		turnPrefixMessages,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		cutPoint,
		previousSummary,
		fileOps,
		settings,
	};
}

// ============================================================================
// Main compaction function
// ============================================================================

/** Summary function signature for custom summarization. */
export type SummaryFn = (
	messages: CompactableMessage[],
	prevSummary?: string,
	customInstructions?: string,
) => Promise<string>;

/** Generate compaction summary and return compacted state. */
export async function compact(
	messages: CompactableMessage[],
	settings: CompactionSettings,
	previousSummary?: string,
	customInstructions?: string,
	summarize?: SummaryFn,
): Promise<CompactionResult<CompactionDetails>> {
	const preparation = prepareCompaction(messages, settings, previousSummary);
	if (!preparation) {
		const toks = estimateContextTokens(messages).tokens;
		return {
			summary: previousSummary ?? "No prior history.",
			firstKeptIndex: 0,
			firstKeptEntryId: messages[0]?.entryId,
			tokensAfter: toks,
			tokensBefore: toks,
			messagesToKeep: messages,
			details: { readFiles: [], modifiedFiles: [] },
		};
	}

	const summaryFn = summarize || generateSummary;

	let summary: string;

	if (preparation.isSplitTurn && preparation.turnPrefixMessages) {
		const [historyResult, turnPrefixResult] = await Promise.all([
			preparation.messagesToSummarize.length > 0
				? summaryFn(
						preparation.messagesToSummarize,
						previousSummary,
						customInstructions,
					)
				: Promise.resolve("No prior history."),
			generateTurnPrefixSummary(preparation.turnPrefixMessages),
		]);
		summary = `${historyResult}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult}`;
	} else {
		summary = await summaryFn(
			preparation.messagesToSummarize,
			previousSummary,
			customInstructions,
		);
	}

	const { readFiles, modifiedFiles } = COMPUTE_FILE_LISTS(preparation.fileOps);
	summary += FORMAT_FILE_OPERATIONS(readFiles, modifiedFiles);

	// Estimate tokens after compaction: summary + kept messages
	const compactedPayload = estimateCompressableTokens({
		role: "compactionSummary",
		content: summary,
	} as CompactableMessage);
	const keptTokens = preparation.messagesToKeep.reduce(
		(sum, m) => sum + estimateCompressableTokens(m),
		0,
	);
	const tokensAfter = compactedPayload + keptTokens;

	return {
		summary,
		firstKeptIndex: preparation.cutPoint.firstKeptIndex,
		firstKeptEntryId: preparation.cutPoint.firstKeptEntryId,
		tokensAfter,
		tokensBefore: preparation.tokensBefore,
		messagesToKeep: preparation.messagesToKeep,
		details: { readFiles, modifiedFiles },
	};
}

async function generateTurnPrefixSummary(
	messages: CompactableMessage[],
): Promise<string> {
	const conversationText = serializeConversation(
		messages as Array<{
			role: string;
			content: string | Array<{ type: string; text?: string }>;
		}>,
	);
	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;
	return `[Turn prefix summary would be generated here. Prompt length: ${promptText.length} chars]`;
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

export function microCompactMessages(
	messages: CompactableMessage[],
): CompactToFitResult {
	const tokensBefore = estimateContextTokens(messages).tokens;
	// Micro-compaction: trim oversized message bodies (up to 4000 chars each)
	const trimmed = messages.map((m) => {
		if (typeof m.content === "string" && m.content.length > 4000) {
			return {
				...m,
				content: m.content.slice(0, 4000) + "\n\n[... truncated]",
			};
		}
		return m;
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
