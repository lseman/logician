// ── Compaction operations for AgentHarness ────────────────────────────────
// Pulled out of harness.ts: manual compact(), auto-compaction threshold
// check, and the shared before/after-compact event + hook plumbing.
// Delegates to the single compaction engine (compactToFit) shared with the
// loop's context-full retry and the builtin proactive hook.

import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import { estimateChatPayloadTokens } from "../../capabilities/provider/messages.ts";
import { generateCompactionSummary } from "../../capabilities/session/summaries/summary-generation.ts";
import type { ThinkingLevel } from "../../system/types/types-config.ts";
import type { CompactableMessage, Message } from "../../system/types/types-messages.ts";
import { type CompactionSettings, compactToFit } from "./engine.ts";

export interface CompactionOutcome {
	changed: boolean;
	messages: Message[];
	tokensBefore: number;
	tokensAfter: number;
}

/**
 * Run the shared compaction engine against `history`, using either a
 * pre-supplied summary (from a beforeCompact hook) or an LLM-generated one.
 * Returns the outcome; caller applies `messages` to its own history field on
 * `changed`. Forced (`triggerTokens: 0`) — the caller has already decided to
 * compact via its own token estimate.
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
	const summarize = async (older: CompactableMessage[]) => {
		if (options.presetSummary) return options.presetSummary;
		return generateCompactionSummary(
			backend,
			older as unknown as Message[],
			[],
			{
				temperature: options.temperature,
				maxTokens: options.maxTokens,
				thinkingLevel: options.thinkingLevel,
			},
		);
	};

	const result = await compactToFit(history as CompactableMessage[], {
		triggerTokens: 0,
		summarize,
	});

	if (!result.changed) {
		return {
			changed: false,
			messages: history,
			tokensBefore,
			tokensAfter: tokensBefore,
		};
	}

	return {
		changed: true,
		messages: result.messages as unknown as Message[],
		tokensBefore,
		tokensAfter: result.tokensAfter,
	};
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
