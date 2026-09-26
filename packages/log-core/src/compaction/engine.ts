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
import {
	type Archive,
	computeFrameTokenOverhead,
	type Frame,
	type FrameConfig,
	PRESERVE_KEY,
	compact as snapcompact,
} from "@logician/log-snapcompact";
import { DEFAULT_TRUNCATION } from "../types/config.ts";
import type {
	AgentMessage,
	CompactableMessage,
	Message,
} from "../types/messages.ts";
import { serializeConversation } from "./serialization.ts";

// ============================================================================
// Types
// ============================================================================
interface RemoteCompactionOptions {
	/** Provider-specific compaction endpoint. When unset, uses the provider's native endpoint (e.g. /responses/compact for OpenAI). */
	endpoint?: string;
	/** Model to use for compaction. Falls back to the current model. */
	model?: string;
	/** Maximum tokens for the compaction response. */
	maxTokens?: number;
	/** Timeout in milliseconds for the remote compaction request. Defaults to 30 seconds. */
	timeoutMs?: number;
}

/** Result from a remote (server-side) compaction call. */
export interface RemoteCompactionResult {
	/** The compacted summary text returned by the provider. */
	summary: string;
	/** Optional preserve data returned by the provider (e.g. replacement history, compaction items). */
	preserveData?: Record<string, unknown>;
	/** Token usage reported by the provider, when available. */
	usage?:
		| {
				promptTokens?: number | undefined;
				completionTokens?: number | undefined;
				totalTokens?: number | undefined;
		  }
		| undefined;
}

/** Compaction thresholds and retention settings. */
export interface CompactionSettings {
	enabled: boolean;
	reserveTokens: number;
	keepRecentTokens: number;
	contextWindow?: number | undefined;
	/** Number of recent messages to always preserve (regardless of token budget). */
	protectedMessageCount?: number | undefined;
	/** Whether to force compaction regardless of current token usage. */
	force?: boolean | undefined;
	/** Compaction strategy: "auto" (micro + LLM or inline summary), "snapcompact" (local bitmap frames), "remote" (provider-native server compaction), or "shake" (drop recoverable content). */
	mode?: "auto" | "llm" | "snapcompact" | "remote" | "shake";
	/** Provider-aware frame sizing for snapcompact PNG rendering. */
	frameOptions?: FrameConfig;
	/** Remote (server-side) compaction configuration. */
	remoteCompaction?: RemoteCompactionOptions;
}

const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	mode: "snapcompact",
	reserveTokens: 16384,
	keepRecentTokens: 20000,
	contextWindow: 128000,
	protectedMessageCount: 3,
	force: false,
};

/** Payload fraction targeted after summarizing compaction. */
export const COMPACTION_TARGET_FRACTION = 0.65;

// ============================================================================
// Token estimation — dual-mode: provider usage when available, char heuristic fallback
// ============================================================================

const ESTIMATED_IMAGE_CHARS = 4800;

/** Estimate tokens for one message using character heuristic. Conservative (overestimates). */
function estimateCompressableTokens(
	message: AgentMessage | CompactableMessage,
): number {
	if (!message) return 0;
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
		// CompactableMessage stores this text under `.content` (already read
		// above as `textContent`) — `.summary` is a stale field name that is
		// never set on the actual constructed message and always undefined.
		chars += textContent.length;
		// Add token overhead for PNG frames stored in snapcompact archive.
		const branchMsg = msg as { snapcompact?: Record<string, unknown> };
		const archive = branchMsg.snapcompact as
			| { snapcompact?: { frames?: Array<{ data: string }> } }
			| undefined;
		const frames = archive?.snapcompact?.frames ?? [];
		chars += computeFrameTokenOverhead(frames as Frame[]) * 4; // rough char-equiv: 4 chars ≈ 1 token
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
interface ContextUsageEstimate {
	tokens: number;
	usageTokens: number;
	trailingTokens: number;
	lastUsageIndex: number | null;
}

/**
 * The engine's own token estimate (provider usage + content heuristics), in
 * the units compactToFit compares `triggerTokens` against. It can differ from
 * the exact BPE count by ±60% depending on content, so callers that decide
 * on BPE counts should scale their trigger into these units.
 */
export function estimateCompactionTokens(
	messages: CompactableMessage[],
): number {
	return estimateContextTokens(messages).tokens;
}

/** Estimate context tokens using provider usage (when available) + estimation. */
function estimateContextTokens(
	messages: CompactableMessage[],
): ContextUsageEstimate {
	// Try to find provider-reported usage from the last assistant message
	let usageTokens = 0;
	let lastUsageIndex: number | null = null;

	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i] as AgentMessage & {
			usage?: Record<string, number> | undefined;
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
			const message = messages[i];
			if (message) trailingTokens += estimateCompressableTokens(message);
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
		const message = messages[i];
		if (!message) continue;
		const role = message.role;
		// An assistant message is a valid split-turn cut: the kept tail starts
		// with the assistant call and its results follow it, so a call is never
		// separated from its results. Without it, a long agentic turn (one user
		// message, many tool rounds) has no cut point and can't be compacted.
		if (
			role === "user" ||
			role === "assistant" ||
			role === "custom" ||
			role === "branchSummary" ||
			role === "compactionSummary"
		) {
			cutPoints.push(i);
		}
		// Never cut at tool results — they belong to the preceding call.
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
		const message = messages[i];
		if (!message) continue;
		const role = message.role;
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
interface CutPointResult {
	firstKeptIndex: number;
	/** UUID of the first kept entry (set when messages carry entryId). */
	firstKeptEntryId?: string | undefined;
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
		if (!msg) continue;
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
	let cutIndex = endIndex; // No message old enough to cut unless found below.
	let crossedBudget = false;

	// Walk backwards accumulating tokens
	for (let i = endIndex - 1; i >= startIndex; i--) {
		const message = messages[i];
		if (!message) continue;
		accumulatedTokens += estimateCompressableTokens(message);

		if (accumulatedTokens >= keepRecentTokens) {
			crossedBudget = true;
			// Find the nearest valid cut point >= this position
			cutIndex = cutPoints[0] ?? endIndex;
			for (const cp of cutPoints) {
				if (cp >= i) {
					cutIndex = cp;
					break;
				}
			}
			break;
		}
	}

	// Entire range fits within the recent-token budget — nothing to compact.
	if (!crossedBudget) {
		return {
			firstKeptIndex: startIndex,
			turnStartIndex: -1,
			isSplitTurn: false,
			protectedStartIndex: startIndex,
			protectedEndIndex: endIndex,
		};
	}

	// Walk backward past non-message entries (metadata, labels, etc.)
	while (cutIndex > startIndex) {
		const role = messages[cutIndex - 1]?.role;
		if (role === "compactionSummary" || role === "branchSummary") {
			break;
		}
		if (
			role === "user" ||
			role === "assistant" ||
			role === "toolResult" ||
			role === "tool"
		) {
			break;
		}
		cutIndex--;
	}

	const isUserMessage = messages[cutIndex]?.role === "user";
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

/** Convert compacted messages back to Message[]. Safe because compaction preserves structure. */
export function toMessages(msgs: CompactableMessage[]): Message[] {
	return msgs as Message[];
}

/** Produces the replacement summary text for the older (compacted-away) block. */
export type CompactionSummarizer = (
	messages: CompactableMessage[],
) => Promise<string | null>;

/**
 * Single entry point for all compaction: harness manual/auto compact, the
 * loop's context-full retry, and the builtin proactive hook all call this.
 *
 * Sequence:
 *  1. If already under `triggerTokens` and not `force`d, do nothing.
 *  2. Run the cheap micro pass (trim oversized bodies). If that brings the
 *     payload under `triggerTokens`, stop there.
 *  3. Otherwise run the full summarizing pass targeting `targetTokens`,
 *     using `summarize` when supplied (LLM-quality) or falling back to the
 *     local structured-text summary.
 */
export async function compactToFit(
	messages: CompactableMessage[],
	opts: {
		triggerTokens: number;
		targetTokens?: number | undefined;
		keepRecentMessages?: number | undefined;
		settings?: Partial<CompactionSettings> | undefined;
		summarize?: CompactionSummarizer | undefined;
		/** Remote (server-side) compaction summary function. */
		remoteSummarizer?: (messages: CompactableMessage[]) => Promise<string>;
	},
): Promise<CompactToFitResult> {
	const {
		triggerTokens,
		keepRecentMessages,
		settings,
		summarize,
		remoteSummarizer,
	} = opts;
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
	const micro = microCompactCompactableMessages(messages);
	const microTokens = estimateContextTokens(micro.messages).tokens;
	if (!force && microTokens < triggerTokens) {
		return { ...micro, changed: micro.tokensAfter < micro.tokensBefore };
	}

	// Full summarizing pass. Without an explicit target, one pass is enough.
	const first = await compactToFitFull(
		micro.messages,
		effectiveSettings,
		summarize,
		remoteSummarizer,
	);
	if (
		opts.targetTokens === undefined ||
		first.tokensAfter <= opts.targetTokens
	) {
		return first;
	}

	// Still over target — tighten the kept-recent-tokens budget and retry with
	// a smaller tail until the target is met or there's nothing left to cut.
	let best = first;
	for (const keepRecentTokens of [
		effectiveSettings.keepRecentTokens / 2,
		effectiveSettings.keepRecentTokens / 4,
		effectiveSettings.keepRecentTokens / 8,
		0,
	]) {
		const attempt = await compactToFitFull(
			micro.messages,
			{ ...effectiveSettings, keepRecentTokens },
			summarize,
			remoteSummarizer,
		);
		if (attempt.tokensAfter < best.tokensAfter) best = attempt;
		if (attempt.tokensAfter <= opts.targetTokens) return attempt;
	}
	return best;
}

// How many trailing messages micro-compaction leaves untouched — the model is
// usually still acting on them.
const MICRO_COMPACT_KEEP_RECENT = 6;

function microCompactMaxChars(role: string): number {
	// Tool results tolerate the most trimming; user prompts the least — losing
	// part of the task statement is worse than a long context.
	const limits = DEFAULT_TRUNCATION.microCompactMaxChars;
	if (role === "tool" || role === "toolResult") return limits.tool;
	if (role === "assistant") return limits.assistant;
	return limits.default;
}

function truncateMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 32) / 2));
	return `${text.slice(0, half)}\n...[compacted ${text.length - half * 2} chars]...\n${text.slice(-half)}`;
}

export interface PruneHistoricalToolOutputsOptions {
	/** Number of recent turns (user-assistant cycles) to keep untouched. Default: 2. */
	keepRecentTurns?: number | undefined;
	/** Maximum character length before a historical tool output is trimmed. Default: 600. */
	maxHistoricalChars?: number | undefined;
	/** Number of head lines to keep when trimming. Default: 5. */
	headLines?: number | undefined;
	/** Number of tail lines to keep when trimming. Default: 5. */
	tailLines?: number | undefined;
}

export interface PrunedToolOutputsResult {
	messages: CompactableMessage[];
	prunedCount: number;
	charactersSaved: number;
	tokensSaved: number;
	changed: boolean;
}

/**
 * Prunes historical tool results in older turns while keeping recent turns intact.
 * SOTA coding agents do not retain 500-line historical compiler logs from 10 turns ago.
 */
export function pruneHistoricalToolOutputs(
	messages: CompactableMessage[],
	options: PruneHistoricalToolOutputsOptions = {},
): PrunedToolOutputsResult {
	const keepRecentTurns = options.keepRecentTurns ?? 2;
	const maxChars = options.maxHistoricalChars ?? 600;
	const headLines = options.headLines ?? 5;
	const tailLines = options.tailLines ?? 5;

	// Find the start index of the protected recent window
	let userTurnsSeen = 0;
	let protectedStartIndex = 0;

	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (msg && msg.role === "user") {
			userTurnsSeen++;
			if (userTurnsSeen === keepRecentTurns) {
				protectedStartIndex = i;
				break;
			}
		}
	}

	let prunedCount = 0;
	let charactersSaved = 0;
	const tokensBefore = estimateContextTokens(messages).tokens;

	const prunedMessages = messages.map((m, index) => {
		if (index >= protectedStartIndex) return m;
		const role = m.role as string;
		if (role !== "tool" && role !== "toolResult" && role !== "bashExecution") {
			return m;
		}

		const content = typeof m.content === "string" ? m.content : "";
		if (content.length <= maxChars) return m;

		const lines = content.split("\n");
		let trimmedContent: string;

		if (lines.length > headLines + tailLines + 2) {
			const head = lines.slice(0, headLines).join("\n");
			const tail = lines.slice(-tailLines).join("\n");
			const omittedLines = lines.length - headLines - tailLines;
			trimmedContent = `${head}\n\n[... historical output trimmed (${omittedLines} lines omitted) ...]\n\n${tail}`;
		} else {
			trimmedContent = truncateMiddle(content, maxChars);
		}

		prunedCount++;
		charactersSaved += content.length - trimmedContent.length;
		return { ...m, content: trimmedContent };
	});

	const tokensAfter = estimateContextTokens(prunedMessages).tokens;

	return {
		messages: prunedMessages,
		prunedCount,
		charactersSaved,
		tokensSaved: Math.max(0, tokensBefore - tokensAfter),
		changed: prunedCount > 0,
	};
}

export function microCompactCompactableMessages(
	messages: CompactableMessage[],
): CompactToFitResult {
	const tokensBefore = estimateContextTokens(messages).tokens;

	// Step 1: Prune historical tool outputs older than 2 turns
	const pruned = pruneHistoricalToolOutputs(messages, {
		keepRecentTurns: 2,
		maxHistoricalChars: 600,
	});

	// Step 2: Role-aware micro-compaction on oversized messages
	const trimmed = pruned.messages.map((m, index) => {
		if (index >= pruned.messages.length - MICRO_COMPACT_KEEP_RECENT) return m;
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

async function compactToFitFull(
	messages: CompactableMessage[],
	settings: CompactionSettings,
	summarize: CompactionSummarizer | undefined,
	remoteSummarizer:
		| ((messages: CompactableMessage[]) => Promise<string>)
		| undefined,
): Promise<CompactToFitResult> {
	// Find the cut point using turn-boundary-aware logic
	const cutPoint = findCutPoint(
		messages,
		0,
		messages.length,
		settings.keepRecentTokens,
	);

	if (
		cutPoint.firstKeptIndex >= messages.length ||
		cutPoint.firstKeptIndex <= 0
	) {
		return {
			messages,
			tokensBefore: estimateContextTokens(messages).tokens,
			tokensAfter: estimateContextTokens(messages).tokens,
			changed: false,
		};
	}

	const messagesToKeep = messages.slice(cutPoint.firstKeptIndex);
	const messagesToSummarize = messages.slice(0, cutPoint.firstKeptIndex);

	// Build compacted message list.
	let compacted: CompactableMessage[];
	let tokensAfter: number;
	if (settings.mode === "remote") {
		// Remote (server-side) compaction: delegate to provider's native endpoint.
		const summary = await remoteSummarizer?.(messagesToSummarize);
		const fallbackSummary =
			summary ?? generateInlineSummary(messagesToSummarize, settings);

		compacted = [
			{ role: "compactionSummary" as const, content: fallbackSummary },
			...messagesToKeep,
		];
		tokensAfter = estimateContextTokens(compacted).tokens;
	} else if (settings.mode === "snapcompact") {
		// Snapcompact: local, deterministic bitmap frame rendering with provider-aware sizing.
		const firstKeptId =
			messagesToSummarize[messagesToSummarize.length - 1]?.entryId;
		const frameConfig = settings.frameOptions;
		const compactOpts: {
			maxFrames?: number;
			firstKeptEntryId?: string;
			shape?: { cols?: number; rows?: number; lineRepeat?: number };
			serializeOptions?: { provider?: string; render?: boolean };
			render?: boolean;
			previousArchive?: Archive;
			previousSummary?: string;
		} = { maxFrames: frameConfig?.maxFrames ?? 40 };
		if (firstKeptId) compactOpts.firstKeptEntryId = firstKeptId;
		if (frameConfig) {
			if (frameConfig.cols || frameConfig.rows) {
				const shape: { cols?: number; rows?: number } = {};
				if (frameConfig.cols) shape.cols = frameConfig.cols;
				if (frameConfig.rows) shape.rows = frameConfig.rows;
				compactOpts.shape = shape;
			}
			if (frameConfig.provider || frameConfig.render !== undefined) {
				const opts: { provider?: string; render?: boolean } = {};
				if (frameConfig.provider) opts.provider = frameConfig.provider;
				if (frameConfig.render !== undefined) opts.render = frameConfig.render;
				compactOpts.serializeOptions = opts;
			}
		}

		// Fold forward a prior compactionSummary about to be cut away, so its
		// archived text/frames aren't silently dropped — serializeMessages()
		// only understands user/assistant/toolResult roles and would otherwise
		// contribute nothing for it.
		const priorSummary = [...messagesToSummarize]
			.reverse()
			.find(m => m.role === "compactionSummary");
		if (priorSummary) {
			const priorArchive = (
				priorSummary.snapcompact as Record<string, unknown> | undefined
			)?.[PRESERVE_KEY] as Archive | undefined;
			if (priorArchive) {
				compactOpts.previousArchive = priorArchive;
			} else if (
				typeof priorSummary.content === "string" &&
				priorSummary.content
			) {
				compactOpts.previousSummary = priorSummary.content;
			}
		}

		const result = await snapcompact(messagesToSummarize, compactOpts);

		compacted = [
			{
				role: "compactionSummary" as const,
				content: result.summary,
				timestamp: Date.now(),
				readFiles: [],
				modifiedFiles: [],
				snapcompact: result.preserveData,
			} as CompactableMessage,
			...messagesToKeep,
		];
		// Account for PNG frame overhead when frames carry base64 image data.
		const frameOverhead = computeFrameTokenOverhead(result.frames ?? []);
		tokensAfter = estimateContextTokens(compacted).tokens + frameOverhead;
	} else {
		// Prefer the caller-supplied (typically LLM-based) summarizer; fall back
		// to the local structured-text summary when omitted or it fails.
		const summary =
			(await summarize?.(messagesToSummarize)) ??
			generateInlineSummary(messagesToSummarize, settings);

		compacted = [
			{ role: "compactionSummary" as const, content: summary },
			...messagesToKeep,
		];
		tokensAfter = estimateContextTokens(compacted).tokens;
	}

	return {
		messages: compacted,
		tokensBefore: estimateContextTokens(messages).tokens,
		tokensAfter,
		changed: true,
	};
}

function extractTouchedFiles(messages: CompactableMessage[]): string[] {
	const files = new Set<string>();
	for (const msg of messages) {
		if (!msg) continue;
		if (typeof msg.content === "string") {
			const matches = msg.content.match(
				/(?:(?:\/[a-zA-Z0-9_.-]+)+|[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+|\b\w+\.(?:ts|js|json|md|py|rs|go|html|css|tsx|jsx)\b)/g,
			);
			if (matches) {
				for (const match of matches.slice(0, 10)) {
					if (match.length > 2 && !match.startsWith("//")) files.add(match);
				}
			}
		}
	}
	return Array.from(files).slice(0, 12);
}

function extractInitialGoal(messages: CompactableMessage[]): string {
	for (const msg of messages) {
		if (msg && msg.role === "user" && typeof msg.content === "string") {
			const text = msg.content.trim();
			if (text.length > 0) {
				return text.length > 300 ? `${text.slice(0, 297)}...` : text;
			}
		}
	}
	return "Complete user request";
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

	const initialGoal = extractInitialGoal(messages);
	const touchedFiles = extractTouchedFiles(messages);
	const filesList =
		touchedFiles.length > 0
			? touchedFiles.map(f => `- \`${f}\``).join("\n")
			: "- (no specific files identified)";

	let summary = `<context-compaction reason="auto">
# Context Compaction Summary

## Goal
${initialGoal}

## Key Files Touched
${filesList}

## Progress & Completed Operations
- Earlier conversation compacted to fit token budget.
- Context and decisions summarized below.

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
// ============================================================================
// Shake compaction — drop recoverable heavy content without LLM
// ============================================================================

export interface ShakeCompactionOptions {
	/** Number of recent turns to keep fully intact. Default: 2. */
	keepRecentTurns?: number | undefined;
	/** Max chars for a historical tool result before dropping it entirely. Default: 2000. */
	historicalToolResultThreshold?: number | undefined;
	/** Max chars for a historical assistant message content. Default: 500. */
	historicalAssistantThreshold?: number | undefined;
	/** Max chars for a historical user message content. Default: 500. */
	historicalUserThreshold?: number | undefined;
}

const DEFAULT_SHAKE_OPTIONS: ShakeCompactionOptions = {
	keepRecentTurns: 2,
	historicalToolResultThreshold: 2000,
	historicalAssistantThreshold: 500,
	historicalUserThreshold: 500,
};

/**
 * Shake compaction: drop recoverable heavy content from old turns.
 *
 * Strategy:
 * - Bash tool results: keep command + exit code, drop stdout/stderr
 * - File read/write tool results: keep file path + error/summary, drop content
 * - Large assistant messages: truncate to minimal summary
 * - Large user messages: truncate to minimal summary
 * - Recent turns (last N): untouched
 *
 * This is the "shake" from oh-my-pi — content can be re-read/re-run, so
 * dropping it loses nothing essential. The model can always re-read a file
 * or re-run a command if needed.
 */
export function shakeCompaction(
	messages: CompactableMessage[],
	options: ShakeCompactionOptions = DEFAULT_SHAKE_OPTIONS,
): CompactToFitResult {
	const {
		keepRecentTurns,
		historicalToolResultThreshold,
		historicalAssistantThreshold,
		historicalUserThreshold,
	} = { ...DEFAULT_SHAKE_OPTIONS, ...options };

	const tokensBefore = estimateContextTokens(messages).tokens;

	// Find the boundary of the recent-turn protection zone
	let userTurnsSeen = 0;
	let protectedEndIndex = messages.length;
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i]?.role === "user") {
			userTurnsSeen++;
			if (userTurnsSeen === keepRecentTurns) {
				protectedEndIndex = i;
				break;
			}
		}
	}

	let changed = false;
	const shaken = messages.map((msg, index) => {
		const role = msg.role;
		const content = typeof msg.content === "string" ? msg.content : "";

		// Recent turns are fully protected
		if (index >= protectedEndIndex) return msg;

		let newContent: string = content;

		if (role === "toolResult" || role === "tool" || role === "bashExecution") {
			// For tool results, keep only a minimal summary if content is large
			if (content.length > (historicalToolResultThreshold ?? 2000)) {
				newContent = `[Tool output dropped (${content.length} chars, can be re-run/re-read)]`;
				changed = true;
			} else if (content.length > 800) {
				// Trim large outputs to head/tail with summary
				const lines = content.split("\n");
				if (lines.length > 30) {
					const head = lines.slice(0, 10).join("\n");
					const tail = lines.slice(-10).join("\n");
					newContent = `${head}\n\n[... ${lines.length - 20} lines omitted]...\n\n${tail}`;
					changed = true;
				}
			}
		} else if (role === "assistant") {
			// Truncate long assistant messages to summary
			if (content.length > (historicalAssistantThreshold ?? 500)) {
				const threshold = historicalAssistantThreshold ?? 500;
				const summary = content.slice(0, threshold);
				newContent = `${summary}\n\n... [truncated, ${content.length - threshold} chars dropped]`;
				changed = true;
			}
		} else if (role === "user") {
			// Truncate long user messages to summary
			if (content.length > (historicalUserThreshold ?? 500)) {
				const threshold = historicalUserThreshold ?? 500;
				const summary = content.slice(0, threshold);
				newContent = `${summary}\n\n... [truncated, ${content.length - threshold} chars dropped]`;
				changed = true;
			}
		}

		if (newContent !== content) {
			return { ...msg, content: newContent };
		}
		return msg;
	});

	const tokensAfter = estimateContextTokens(shaken).tokens;

	return {
		messages: shaken,
		tokensBefore,
		tokensAfter,
		changed: changed || tokensAfter < tokensBefore,
	};
}

/** Compaction mode: shake, auto, or llm. */
export type CompactionMode = "shake" | "auto" | "llm" | "snapcompact";
