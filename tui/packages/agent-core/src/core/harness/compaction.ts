// ── Compaction operations for AgentHarness ────────────────────────────────
// Pulled out of harness.ts: manual compact(), auto-compaction threshold
// check, and the shared before/after-compact event + hook plumbing.

import type { LLMBackend } from "../backend.ts";
import { compactMessages, estimateChatPayloadTokens } from "../messages.ts";
import { generateCompactionSummary } from "../summaries/summary-generation.ts";
import type { CompactionSettings } from "../../compaction/index.ts";
import type { Message, ThinkingLevel } from "../types.ts";

export interface CompactionOutcome {
	changed: boolean;
	messages: Message[];
	tokensBefore: number;
	tokensAfter: number;
}

/**
 * Run compactMessages against `history`, using either a pre-supplied summary
 * (from a beforeCompact hook) or an LLM-generated one. Returns the outcome;
 * caller applies `messages` to its own history field on `changed`.
 */
export async function runCompaction(
	backend: LLMBackend,
	history: Message[],
	tokensBefore: number,
	options: {
		reason: "auto" | "manual";
		presetSummary?: string;
		temperature?: number;
		maxTokens?: number;
		thinkingLevel?: ThinkingLevel;
	},
): Promise<CompactionOutcome> {
	const result = await compactMessages(history, {
		reason: options.reason,
		summarize: options.presetSummary
			? async () => options.presetSummary!
			: (older, system) =>
					generateCompactionSummary(
						backend,
						older.map((m) => ({
							role: m.role as Message["role"],
							content: m.content ?? "",
							tool_call_id: m.tool_call_id,
							tool_calls: m.tool_calls,
							name: m.name,
							timestamp: m.timestamp,
						})),
						system.map((m) => ({
							role: m.role as Message["role"],
							content: m.content ?? "",
							tool_call_id: m.tool_call_id,
							tool_calls: m.tool_calls,
							name: m.name,
							timestamp: m.timestamp,
						})),
						{
							temperature: options.temperature,
							maxTokens: options.maxTokens,
							thinkingLevel: options.thinkingLevel,
						},
					),
	});

	if (!result.changed) {
		return { changed: false, messages: history, tokensBefore, tokensAfter: tokensBefore };
	}

	const tokensAfter = estimateChatPayloadTokens(result.messages);
	return { changed: true, messages: result.messages, tokensBefore, tokensAfter };
}

/** Whether auto-compaction should fire given current settings + message history. */
export function shouldAutoCompact(
	settings: CompactionSettings,
	messages: Message[],
): boolean {
	if (!settings.enabled) return false;
	const contextWindow = settings.contextWindow ?? 128000;
	const threshold = contextWindow - (settings.reserveTokens ?? 16384);
	const currentTokens = estimateChatPayloadTokens(messages);
	return currentTokens > threshold;
}
