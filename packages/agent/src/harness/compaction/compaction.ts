// ── Context compaction ────────────────────────────────────────────────────
// Merges pi coding agent's Entry-native, turn-boundary-aware compaction
// pipeline (harness/compaction/compaction.ts: cut-point selection, split-turn
// prefix summarization, file-operation tracking) with our previous
// agent-core's content-type-aware token estimator and cheap micro-compact
// pass (neither of which pi has). LLM calls go through agent's own
// ai/openai-completions.ts adapter instead of pi-ai's Models registry.

import type { AgentMessage, ThinkingLevel } from "../../agent/types.ts";
import { streamSimple } from "../../ai/openai-completions.ts";
import { retryAssistantCall } from "../../ai/retry-assistant.ts";
import { contentText } from "../../ai/text.ts";
import type {
	Model,
	ThinkingLevel as ProviderThinkingLevel,
	Usage,
} from "../../ai/types.ts";
import { CompactionError } from "../../core/errors.ts";
import { err, ok, type Result } from "../../core/result.ts";
import type { RetryCallbacks, RetryPolicy } from "../../core/retry.ts";
import {
	convertToLlm,
	createBranchSummaryMessage,
	createCompactionSummaryMessage,
} from "../messages.ts";
import { buildSessionContext } from "../session/context.ts";
import type { CompactionEntry, Entry } from "../session/types.ts";
import {
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessage,
	type FileOperations,
	formatFileOperations,
	serializeConversation,
} from "./utils.ts";

/** File-operation details stored on generated compaction entries. */
export interface CompactionDetails {
	/** Files read in the compacted history. */
	readFiles: string[];
	/** Files modified in the compacted history. */
	modifiedFiles: string[];
}

function extractFileOperations(
	messages: AgentMessage[],
	entries: Entry[],
	prevCompactionIndex: number,
): FileOperations {
	const fileOps = createFileOps();
	if (prevCompactionIndex >= 0) {
		const prevCompaction = entries[prevCompactionIndex] as CompactionEntry;
		if (prevCompaction.details) {
			const details = prevCompaction.details as CompactionDetails;
			if (Array.isArray(details.readFiles)) {
				for (const f of details.readFiles) fileOps.read.add(f);
			}
			if (Array.isArray(details.modifiedFiles)) {
				for (const f of details.modifiedFiles) fileOps.edited.add(f);
			}
		}
	}
	for (const msg of messages) {
		extractFileOpsFromMessage(msg, fileOps);
	}

	return fileOps;
}

function getMessageFromEntry(entry: Entry): AgentMessage | undefined {
	if (entry.type === "message") {
		return entry.message;
	}
	if (entry.type === "branch_summary") {
		return createBranchSummaryMessage(
			entry.summary,
			entry.fromId,
			entry.timestamp,
		);
	}
	if (entry.type === "compaction") {
		return createCompactionSummaryMessage(
			entry.summary,
			entry.tokensBefore,
			entry.timestamp,
		);
	}
	return undefined;
}

function getMessageFromEntryForCompaction(
	entry: Entry,
): AgentMessage | undefined {
	if (entry.type === "compaction") return undefined;
	return getMessageFromEntry(entry);
}

/** Generated compaction data ready to be persisted as a compaction entry. */
export interface CompactResult<T = unknown> {
	/** Summary text that replaces compacted history in future context. */
	summary: string;
	/** Estimated context tokens before compaction. */
	tokensBefore: number;
	/** Usage from the LLM call(s) that generated this summary, if available. */
	usage?: Usage;
	/** Retained recent messages stored directly on the compaction entry. */
	retainedTail: AgentMessage[];
	/** Optional implementation-specific details stored with the compaction entry. */
	details?: T;
}

/** Complete a single-turn summarization request with retries, unwrapping to the assistant message. */
async function completeSimpleWithRetries(
	model: Model,
	systemPrompt: string,
	promptText: string,
	options: {
		maxTokens: number;
		signal?: AbortSignal;
		reasoning?: ProviderThinkingLevel;
	},
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
) {
	const context = {
		systemPrompt,
		messages: [
			{
				role: "user" as const,
				content: [{ type: "text" as const, text: promptText }],
				timestamp: Date.now(),
			},
		],
	};
	return retryAssistantCall(
		async () => {
			const stream = streamSimple(model, context, {
				maxTokens: options.maxTokens,
				signal: options.signal,
				reasoning: options.reasoning,
				cacheRetention: "none",
			});
			for await (const _event of stream) {
				// drain — we only need the final result
			}
			return stream.result();
		},
		retry,
		options.signal,
		callbacks,
	);
}

function combineUsage(first: Usage, second: Usage): Usage {
	return {
		input: first.input + second.input,
		output: first.output + second.output,
		cacheRead: first.cacheRead + second.cacheRead,
		cacheWrite: first.cacheWrite + second.cacheWrite,
		...((first.reasoning !== undefined || second.reasoning !== undefined) && {
			reasoning: (first.reasoning ?? 0) + (second.reasoning ?? 0),
		}),
		totalTokens: first.totalTokens + second.totalTokens,
		cost: {
			input: first.cost.input + second.cost.input,
			output: first.cost.output + second.cost.output,
			cacheRead: first.cost.cacheRead + second.cost.cacheRead,
			cacheWrite: first.cost.cacheWrite + second.cost.cacheWrite,
			total: first.cost.total + second.cost.total,
		},
	};
}

/** Compaction thresholds and retention settings. */
export interface CompactionSettings {
	/** Enable automatic compaction decisions. */
	enabled: boolean;
	/** Tokens reserved for summary prompt and output. */
	reserveTokens: number;
	/** Approximate recent-context tokens to keep after compaction. */
	keepRecentTokens: number;
	/** Total model context window. Used by shouldCompact/estimation when the model isn't otherwise known. */
	contextWindow?: number;
	/** Number of recent messages to always preserve (regardless of token budget). */
	protectedMessageCount?: number;
	/** Whether to force compaction regardless of current token usage. */
	force?: boolean;
}

/** Default compaction settings used by the harness. */
export const DEFAULT_COMPACTION_SETTINGS: CompactionSettings = {
	enabled: true,
	reserveTokens: 16384,
	keepRecentTokens: 20000,
	contextWindow: 128000,
	protectedMessageCount: 3,
	force: false,
};

/** Calculate total context tokens from provider usage. */
export function calculateContextTokens(usage: Usage): number {
	return (
		usage.totalTokens ||
		usage.input + usage.output + usage.cacheRead + usage.cacheWrite
	);
}

function getAssistantUsage(msg: AgentMessage): Usage | undefined {
	if (msg.role === "assistant" && "usage" in msg) {
		if (
			msg.stopReason !== "aborted" &&
			msg.stopReason !== "error" &&
			msg.usage &&
			calculateContextTokens(msg.usage) > 0
		) {
			return msg.usage;
		}
	}
	return undefined;
}

/** Return usage from the last valid assistant message in session entries. */
export function getLastAssistantUsage(entries: Entry[]): Usage | undefined {
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry?.type === "message") {
			const usage = getAssistantUsage(entry.message);
			if (usage) return usage;
		}
	}
	return undefined;
}

/** Estimated context-token usage for a message list. */
export interface ContextUsageEstimate {
	/** Estimated total context tokens. */
	tokens: number;
	/** Tokens reported by the most recent assistant usage block. */
	usageTokens: number;
	/** Estimated tokens after the most recent assistant usage block. */
	trailingTokens: number;
	/** Index of the message that provided usage, or null when none exists. */
	lastUsageIndex: number | null;
}

// ── Content-type-aware token estimation (ported from our previous agent-core;
// pi's estimateTokens uses a flat char/4 heuristic — this classifies content
// as json/code/unicode/natural-language and applies a tighter, more accurate
// char-to-token ratio per class). ──────────────────────────────────────────

const ESTIMATED_IMAGE_CHARS = 4800;

function classifyContentType(text: string): string {
	if (!text || text.length < 16) return "natural";
	const trimmed = text.trim();

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

	const codePattern =
		/\b(function|const|let|var|class|import|export|return|if|else|for|while|def |async |await )\b/i;
	const codeRatio =
		(
			trimmed.match(
				/\b(function|const|let|var|class|import|export|return|if|else|for|while|def |async |await)\b/gi,
			) || []
		).length / Math.max(1, trimmed.split(/\s+/).length);
	if (codePattern.test(trimmed) || codeRatio > 0.05) return "code";

	let multiByte = 0;
	for (const char of trimmed) {
		if (char.charCodeAt(0) > 0x7f) multiByte++;
	}
	const multiByteRatio = multiByte / trimmed.length;
	if (multiByteRatio > 0.3) return "unicode";
	if (multiByteRatio > 0.05) return "natural-unicode";
	return "natural";
}

function tokenRatioForContentType(type: string): number {
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
}

function safeJsonStringify(value: unknown): string {
	try {
		return JSON.stringify(value) ?? "undefined";
	} catch {
		return "[unserializable]";
	}
}

/** Estimate token count for one message using a content-type-aware character heuristic. */
export function estimateTokens(message: AgentMessage): number {
	let chars = 0;
	let sampleText = "";

	switch (message.role) {
		case "user": {
			if (typeof message.content === "string") {
				chars = message.content.length;
				sampleText = message.content;
			} else {
				for (const block of message.content) {
					if (block.type === "text") {
						chars += block.text.length;
						sampleText = sampleText || block.text;
					} else if (block.type === "image") {
						chars += ESTIMATED_IMAGE_CHARS;
					}
				}
			}
			break;
		}
		case "assistant": {
			for (const block of message.content) {
				if (block.type === "text") {
					chars += block.text.length;
					sampleText = sampleText || block.text;
				} else if (block.type === "thinking") {
					chars += block.thinking.length;
				} else if (block.type === "toolCall") {
					chars +=
						block.name.length + safeJsonStringify(block.arguments).length;
				}
			}
			break;
		}
		case "custom":
		case "toolResult": {
			if (typeof message.content === "string") {
				chars = message.content.length;
				sampleText = message.content;
			} else {
				for (const block of message.content) {
					if (block.type === "text") {
						chars += block.text.length;
						sampleText = sampleText || block.text;
					} else if (block.type === "image") {
						chars += ESTIMATED_IMAGE_CHARS;
					}
				}
			}
			break;
		}
		case "bashExecution": {
			chars = message.command.length + message.output.length;
			sampleText = message.output;
			break;
		}
		case "branchSummary":
		case "compactionSummary": {
			chars = message.summary.length;
			sampleText = message.summary;
			break;
		}
		default:
			return 0;
	}

	const ratio = tokenRatioForContentType(classifyContentType(sampleText));
	return Math.ceil(chars / ratio);
}

function getLastAssistantUsageInfo(
	messages: AgentMessage[],
): { usage: Usage; index: number } | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message === undefined) continue;
		const usage = getAssistantUsage(message);
		if (usage) return { usage, index: i };
	}
	return undefined;
}

/** Estimate context tokens for messages using provider usage when available. */
export function estimateContextTokens(
	messages: AgentMessage[],
): ContextUsageEstimate {
	const usageInfo = getLastAssistantUsageInfo(messages);

	if (!usageInfo) {
		let estimated = 0;
		for (const message of messages) {
			estimated += estimateTokens(message);
		}
		return {
			tokens: estimated,
			usageTokens: 0,
			trailingTokens: estimated,
			lastUsageIndex: null,
		};
	}

	const usageTokens = calculateContextTokens(usageInfo.usage);
	let trailingTokens = 0;
	for (let i = usageInfo.index + 1; i < messages.length; i++) {
		const message = messages[i];
		if (message !== undefined) trailingTokens += estimateTokens(message);
	}

	return {
		tokens: usageTokens + trailingTokens,
		usageTokens,
		trailingTokens,
		lastUsageIndex: usageInfo.index,
	};
}

/** Return whether context usage exceeds the configured compaction threshold. */
export function shouldCompact(
	contextTokens: number,
	contextWindow: number,
	settings: CompactionSettings,
): boolean {
	if (!settings.enabled) return false;
	if (settings.force) return true;
	return contextTokens > contextWindow - settings.reserveTokens;
}

// ── Cut point detection (turn-boundary-aware; never cuts mid-turn) ─────────

function findValidCutPoints(
	entries: Entry[],
	startIndex: number,
	endIndex: number,
): number[] {
	const cutPoints: number[] = [];
	for (let i = startIndex; i < endIndex; i++) {
		const entry = entries[i];
		if (entry === undefined) continue;
		switch (entry.type) {
			case "message": {
				const role = entry.message.role;
				switch (role) {
					case "bashExecution":
					case "custom":
					case "branchSummary":
					case "compactionSummary":
					case "user":
					case "assistant":
						cutPoints.push(i);
						break;
					case "toolResult":
						break;
				}
				break;
			}
			case "thinking_level_change":
			case "model_change":
			case "active_tools_change":
			case "compaction":
				break;
		}
		if (entry.type === "branch_summary") cutPoints.push(i);
	}
	return cutPoints;
}

/** Find the user-visible message that starts the turn containing an entry. */
export function findTurnStartIndex(
	entries: Entry[],
	entryIndex: number,
	startIndex: number,
): number {
	for (let i = entryIndex; i >= startIndex; i--) {
		const entry = entries[i];
		if (entry === undefined) continue;
		if (entry.type === "branch_summary") return i;
		if (entry.type === "message") {
			const role = entry.message.role;
			if (role === "user" || role === "bashExecution") return i;
		}
	}
	return -1;
}

/** Cut point selected for compaction. */
export interface CutPointResult {
	/** Index of the first entry retained after compaction. */
	firstKeptEntryIndex: number;
	/** Index of the turn-start entry when the cut splits a turn, otherwise -1. */
	turnStartIndex: number;
	/** Whether the selected cut point splits an in-progress turn. */
	isSplitTurn: boolean;
}

/** Find the compaction cut point that keeps approximately the requested recent-token budget. */
export function findCutPoint(
	entries: Entry[],
	startIndex: number,
	endIndex: number,
	keepRecentTokens: number,
): CutPointResult {
	const cutPoints = findValidCutPoints(entries, startIndex, endIndex);

	if (cutPoints.length === 0) {
		return {
			firstKeptEntryIndex: startIndex,
			turnStartIndex: -1,
			isSplitTurn: false,
		};
	}
	let accumulatedTokens = 0;
	let cutIndex = cutPoints[0] ?? startIndex;

	for (let i = endIndex - 1; i >= startIndex; i--) {
		const entry = entries[i];
		if (entry?.type !== "message") continue;
		const messageTokens = estimateTokens(entry.message);
		accumulatedTokens += messageTokens;
		if (accumulatedTokens >= keepRecentTokens) {
			for (const cp of cutPoints) {
				if (cp >= i) {
					cutIndex = cp;
					break;
				}
			}
			break;
		}
	}
	while (cutIndex > startIndex) {
		const prevEntry = entries[cutIndex - 1];
		if (prevEntry?.type === "compaction" || prevEntry?.type === "message")
			break;
		cutIndex--;
	}
	const cutEntry = entries[cutIndex];
	const isUserMessage =
		cutEntry?.type === "message" && cutEntry.message.role === "user";
	const turnStartIndex = isUserMessage
		? -1
		: findTurnStartIndex(entries, cutIndex, startIndex);

	return {
		firstKeptEntryIndex: cutIndex,
		turnStartIndex,
		isSplitTurn: !isUserMessage && turnStartIndex !== -1,
	};
}

// ── Micro-compact pass (ported from our previous agent-core; pi has no
// equivalent cheap pre-pass — it always goes straight to LLM summarization). ─

const MICRO_COMPACT_KEEP_RECENT = 6;
const MICRO_COMPACT_MAX_CHARS = { tool: 4000, assistant: 6000, default: 3000 };

function microCompactMaxChars(role: string): number {
	if (role === "toolResult") return MICRO_COMPACT_MAX_CHARS.tool;
	if (role === "assistant") return MICRO_COMPACT_MAX_CHARS.assistant;
	return MICRO_COMPACT_MAX_CHARS.default;
}

export function truncateMiddle(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const half = Math.max(1, Math.floor((maxChars - 32) / 2));
	return `${text.slice(0, half)}\n...[compacted ${text.length - half * 2} chars]...\n${text.slice(-half)}`;
}

export interface MicroCompactResult {
	messages: AgentMessage[];
	tokensBefore: number;
	tokensAfter: number;
	changed: boolean;
}

/** Cheap trim-only pass: shortens oversized string-content message bodies, role-aware. Recent messages are left intact. */
export function microCompactMessages(
	messages: AgentMessage[],
): MicroCompactResult {
	const tokensBefore = estimateContextTokens(messages).tokens;
	const trimmed = messages.map((m, index) => {
		if (index >= messages.length - MICRO_COMPACT_KEEP_RECENT) return m;
		if (!("content" in m) || typeof m.content !== "string") return m;
		const maxChars = microCompactMaxChars(m.role);
		if (m.content.length <= maxChars) return m;
		return {
			...m,
			content: truncateMiddle(m.content, maxChars),
		} as AgentMessage;
	});
	const tokensAfter = estimateContextTokens(trimmed).tokens;
	return {
		messages: trimmed,
		tokensBefore,
		tokensAfter,
		changed: tokensAfter < tokensBefore,
	};
}

// ── Summarization prompts ───────────────────────────────────────────────────

export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

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

const UPDATE_SUMMARIZATION_PROMPT = `The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

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

/** Generate or update a conversation summary for compaction. */
export async function generateSummary(
	currentMessages: AgentMessage[],
	model: Model,
	reserveTokens: number,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
	thinkingLevel?: ThinkingLevel,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
): Promise<Result<string, CompactionError>> {
	const result = await generateSummaryWithUsage(
		currentMessages,
		model,
		reserveTokens,
		signal,
		customInstructions,
		previousSummary,
		thinkingLevel,
		retry,
		callbacks,
	);
	return result.ok ? ok(result.value.text) : err(result.error);
}

/** Generate or update a conversation summary and return its provider usage. */
export async function generateSummaryWithUsage(
	currentMessages: AgentMessage[],
	model: Model,
	reserveTokens: number,
	signal?: AbortSignal,
	customInstructions?: string,
	previousSummary?: string,
	thinkingLevel?: ThinkingLevel,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
): Promise<Result<{ text: string; usage: Usage }, CompactionError>> {
	const maxTokens = Math.min(
		Math.floor(0.8 * reserveTokens),
		model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
	);
	let basePrompt = previousSummary
		? UPDATE_SUMMARIZATION_PROMPT
		: SUMMARIZATION_PROMPT;
	if (customInstructions) {
		basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
	}
	const llmMessages = convertToLlm(currentMessages);
	const conversationText = serializeConversation(llmMessages);
	let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
	if (previousSummary) {
		promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
	}
	promptText += basePrompt;

	const response = await completeSimpleWithRetries(
		model,
		SUMMARIZATION_SYSTEM_PROMPT,
		promptText,
		{
			maxTokens,
			signal,
			reasoning:
				model.reasoning && thinkingLevel && thinkingLevel !== "off"
					? thinkingLevel
					: undefined,
		},
		retry,
		callbacks,
	);

	if (response.stopReason === "aborted") {
		return err(
			new CompactionError(
				"aborted",
				response.errorMessage || "Summarization aborted",
			),
		);
	}
	if (response.stopReason === "error") {
		return err(
			new CompactionError(
				"summarization_failed",
				`Summarization failed: ${response.errorMessage || "Unknown error"}`,
			),
		);
	}

	return ok({ text: contentText(response.content), usage: response.usage });
}

/** Prepared inputs for a compaction run. */
export interface CompactionPreparation {
	/** Messages summarized into the history summary. */
	messagesToSummarize: AgentMessage[];
	/** Prefix messages summarized separately when compaction splits a turn. */
	turnPrefixMessages: AgentMessage[];
	/** Recent messages retained after compaction and stored on the compaction entry. */
	retainedTail: AgentMessage[];
	/** Whether compaction splits a turn. */
	isSplitTurn: boolean;
	/** Estimated context tokens before compaction. */
	tokensBefore: number;
	/** Previous compaction summary used for iterative updates. */
	previousSummary?: string;
	/** File operations extracted from summarized history. */
	fileOps: FileOperations;
	/** Settings used to prepare compaction. */
	settings: CompactionSettings;
}

/** Prepare session entries for compaction, or return undefined when compaction is not applicable. */
export function prepareCompaction(
	pathEntries: Entry[],
	settings: CompactionSettings,
): Result<CompactionPreparation | undefined, CompactionError> {
	if (
		pathEntries.length === 0 ||
		pathEntries[pathEntries.length - 1]?.type === "compaction"
	) {
		return ok(undefined);
	}

	let prevCompactionIndex = -1;
	for (let i = pathEntries.length - 1; i >= 0; i--) {
		if (pathEntries[i]?.type === "compaction") {
			prevCompactionIndex = i;
			break;
		}
	}

	let previousSummary: string | undefined;
	let compactableEntries = pathEntries;
	if (prevCompactionIndex >= 0) {
		const prevCompaction = pathEntries[prevCompactionIndex] as CompactionEntry;
		previousSummary = prevCompaction.summary;
		const virtualRetainedEntries: Entry[] = prevCompaction.retainedTail.map(
			(message, index) => ({
				type: "message",
				id: `${prevCompaction.id}:retained:${index}`,
				parentId:
					index === 0
						? prevCompaction.id
						: `${prevCompaction.id}:retained:${index - 1}`,
				seq: prevCompaction.seq,
				timestamp: message.timestamp,
				message,
			}),
		);
		compactableEntries = [
			...virtualRetainedEntries,
			...pathEntries.slice(prevCompactionIndex + 1),
		];
	}
	const boundaryEnd = compactableEntries.length;

	const tokensBefore = estimateContextTokens(
		buildSessionContext(pathEntries).messages,
	).tokens;

	const cutPoint = findCutPoint(
		compactableEntries,
		0,
		boundaryEnd,
		settings.keepRecentTokens,
	);
	const historyEnd = cutPoint.isSplitTurn
		? cutPoint.turnStartIndex
		: cutPoint.firstKeptEntryIndex;
	const messagesToSummarize: AgentMessage[] = [];
	for (let i = 0; i < historyEnd; i++) {
		const entry = compactableEntries[i];
		if (entry === undefined) continue;
		const msg = getMessageFromEntryForCompaction(entry);
		if (msg) messagesToSummarize.push(msg);
	}
	const turnPrefixMessages: AgentMessage[] = [];
	if (cutPoint.isSplitTurn) {
		for (
			let i = cutPoint.turnStartIndex;
			i < cutPoint.firstKeptEntryIndex;
			i++
		) {
			const entry = compactableEntries[i];
			if (entry === undefined) continue;
			const msg = getMessageFromEntryForCompaction(entry);
			if (msg) turnPrefixMessages.push(msg);
		}
	}
	const retainedTail: AgentMessage[] = [];
	for (let i = cutPoint.firstKeptEntryIndex; i < boundaryEnd; i++) {
		const entry = compactableEntries[i];
		if (entry === undefined) continue;
		const msg = getMessageFromEntryForCompaction(entry);
		if (msg) retainedTail.push(msg);
	}
	const fileOps = extractFileOperations(
		messagesToSummarize,
		pathEntries,
		prevCompactionIndex,
	);
	if (cutPoint.isSplitTurn) {
		for (const msg of turnPrefixMessages) {
			extractFileOpsFromMessage(msg, fileOps);
		}
	}

	return ok({
		messagesToSummarize,
		turnPrefixMessages,
		retainedTail,
		isSplitTurn: cutPoint.isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	});
}

const TURN_PREFIX_SUMMARIZATION_PROMPT = `This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix.`;

export { serializeConversation } from "./utils.ts";

/** Generate compaction summary data from prepared session history. */
export async function compact(
	preparation: CompactionPreparation,
	model: Model,
	customInstructions?: string,
	signal?: AbortSignal,
	thinkingLevel?: ThinkingLevel,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
): Promise<Result<CompactResult, CompactionError>> {
	const {
		messagesToSummarize,
		turnPrefixMessages,
		retainedTail,
		isSplitTurn,
		tokensBefore,
		previousSummary,
		fileOps,
		settings,
	} = preparation;

	let summary: string;
	let summaryUsage: Usage;

	if (isSplitTurn && turnPrefixMessages.length > 0) {
		let historyText = "No prior history.";
		let historyUsage: Usage | undefined;
		if (messagesToSummarize.length > 0) {
			const historyResult = await generateSummaryWithUsage(
				messagesToSummarize,
				model,
				settings.reserveTokens,
				signal,
				customInstructions,
				previousSummary,
				thinkingLevel,
				retry,
				callbacks,
			);
			if (!historyResult.ok) return err(historyResult.error);
			historyText = historyResult.value.text;
			historyUsage = historyResult.value.usage;
		}
		const turnPrefixResult = await generateTurnPrefixSummary(
			turnPrefixMessages,
			model,
			settings.reserveTokens,
			signal,
			thinkingLevel,
			retry,
			callbacks,
		);
		if (!turnPrefixResult.ok) return err(turnPrefixResult.error);
		summary = `${historyText}\n\n---\n\n**Turn Context (split turn):**\n\n${turnPrefixResult.value.text}`;
		summaryUsage = historyUsage
			? combineUsage(historyUsage, turnPrefixResult.value.usage)
			: turnPrefixResult.value.usage;
	} else {
		const summaryResult = await generateSummaryWithUsage(
			messagesToSummarize,
			model,
			settings.reserveTokens,
			signal,
			customInstructions,
			previousSummary,
			thinkingLevel,
			retry,
			callbacks,
		);
		if (!summaryResult.ok) return err(summaryResult.error);
		summary = summaryResult.value.text;
		summaryUsage = summaryResult.value.usage;
	}

	const { readFiles, modifiedFiles } = computeFileLists(fileOps);
	summary += formatFileOperations(readFiles, modifiedFiles);

	return ok({
		summary,
		tokensBefore,
		usage: summaryUsage,
		retainedTail,
		details: { readFiles, modifiedFiles } as CompactionDetails,
	});
}

async function generateTurnPrefixSummary(
	messages: AgentMessage[],
	model: Model,
	reserveTokens: number,
	signal?: AbortSignal,
	thinkingLevel?: ThinkingLevel,
	retry?: RetryPolicy,
	callbacks?: RetryCallbacks,
): Promise<Result<{ text: string; usage: Usage }, CompactionError>> {
	const maxTokens = Math.min(
		Math.floor(0.5 * reserveTokens),
		model.maxTokens > 0 ? model.maxTokens : Number.POSITIVE_INFINITY,
	);
	const llmMessages = convertToLlm(messages);
	const conversationText = serializeConversation(llmMessages);
	const promptText = `<conversation>\n${conversationText}\n</conversation>\n\n${TURN_PREFIX_SUMMARIZATION_PROMPT}`;

	const response = await completeSimpleWithRetries(
		model,
		SUMMARIZATION_SYSTEM_PROMPT,
		promptText,
		{
			maxTokens,
			signal,
			reasoning:
				model.reasoning && thinkingLevel && thinkingLevel !== "off"
					? thinkingLevel
					: undefined,
		},
		retry,
		callbacks,
	);

	if (response.stopReason === "aborted") {
		return err(
			new CompactionError(
				"aborted",
				response.errorMessage || "Turn prefix summarization aborted",
			),
		);
	}
	if (response.stopReason === "error") {
		return err(
			new CompactionError(
				"summarization_failed",
				`Turn prefix summarization failed: ${response.errorMessage || "Unknown error"}`,
			),
		);
	}

	return ok({ text: contentText(response.content), usage: response.usage });
}
