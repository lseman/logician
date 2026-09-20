// ── Compaction operations for AgentSession ────────────────────────────────
// Pulled out of harness.ts: manual compact(), auto-compaction threshold
// check, and the shared before/after-compact event + hook plumbing.
// Delegates to the single compaction engine (compactToFit) shared with the
// loop's context-full retry and the builtin proactive hook.

import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import { estimateChatPayloadTokens } from "../../capabilities/provider/messages.ts";
import { generateCompactionSummary } from "../../capabilities/session/summaries/summary-generation.ts";
import type { ThinkingLevel } from "../../system/types/types-config.ts";
import type {
	CompactableMessage,
	Message,
} from "../../system/types/types-messages.ts";
import {
	type CompactionSettings,
	compactToFit,
	shakeCompaction,
} from "./engine.ts";
import type { FrameConfig } from "@logician/log-snapcompact";

export interface CompactionOutcome {
	changed: boolean;
	/** Compacted message history — always Message[] compatible. */
	messages: CompactableMessage[];
	tokensBefore: number;
	tokensAfter: number;
}

/**
 * Run the shared compaction engine against `history`.
 *
 * Modes:
 * - "llm" (default): LLM-summarize older turns
 * - "shake": drop recoverable heavy content without LLM (bash output, large
 *   tool results, oversized messages)
 * - "auto": try shake first, fall back to LLM if shake wasn't enough
 * - "snapcompact": local, deterministic bitmap frame rendering (no LLM call)
 */
export async function runCompaction(
	backend: LLMBackend,
	history: Message[],
	tokensBefore: number,
	options: {
		reason: "auto" | "manual";
		mode?: "llm" | "shake" | "auto" | "remote" | "snapcompact" | undefined;
		presetSummary?: string | undefined;
		temperature?: number | undefined;
		maxTokens?: number | undefined;
		thinkingLevel?: ThinkingLevel | undefined;
		/** Provider-aware frame sizing, used only when mode is "snapcompact". */
		frameOptions?: FrameConfig | undefined;
	},
): Promise<CompactionOutcome> {
	const { mode = "auto" } = options;

	// Shake pass: drop recoverable heavy content without LLM
	const shakeResult = shakeCompaction(history as CompactableMessage[], {});

	if (!shakeResult.changed) {
		return {
			changed: false,
			messages: history,
			tokensBefore,
			tokensAfter: tokensBefore,
		};
	}

	// Shake-only mode: use the shake result as-is
	if (mode === "shake") {
		return {
			changed: true,
			messages: shakeResult.messages,
			tokensBefore,
			tokensAfter: shakeResult.tokensAfter,
		};
	}

	// Auto mode: if shake brought us well under budget, use it; otherwise LLM
	const shakeTokens = shakeResult.tokensAfter;
	const shakeSaved = tokensBefore - shakeTokens;

	// If shake saved enough tokens (8k+), use shake result
	if (mode === "auto" && shakeSaved >= 8000) {
		return {
			changed: true,
			messages: shakeResult.messages,
			tokensBefore,
			tokensAfter: shakeTokens,
		};
	}

	// Remote mode: use provider's native compaction endpoint
	if (mode === "remote") {
		const remoteSummarizer = async (older: CompactableMessage[]) => {
			const result = await backend.remote(older as unknown as Record<string, unknown>[], {
				maxTokens: options.maxTokens ?? 2048,
			});
			return result.summary;
		};

		const remoteResult = await compactToFit(shakeResult.messages, {
			triggerTokens: 0,
			remoteSummarizer,
			settings: { mode: "remote" },
		});

		return {
			changed: remoteResult.changed,
			messages: remoteResult.messages,
			tokensBefore,
			tokensAfter: remoteResult.tokensAfter,
		};
	}

	// Snapcompact mode: local, deterministic bitmap frame rendering — no LLM call.
	if (mode === "snapcompact") {
		const snapcompactResult = await compactToFit(shakeResult.messages, {
			triggerTokens: 0,
			settings: {
				mode: "snapcompact",
				...(options.frameOptions ? { frameOptions: options.frameOptions } : {}),
			},
		});

		return {
			changed: snapcompactResult.changed,
			messages: snapcompactResult.messages,
			tokensBefore,
			tokensAfter: snapcompactResult.tokensAfter,
		};
	}

	// LLM or auto-needs-LLM: summarize with LLM (on shake-processed history)
	const summarize = async (older: CompactableMessage[]) => {
		if (options.presetSummary) return options.presetSummary;
		return generateCompactionSummary(backend, older as Message[], [], {
			temperature: options.temperature,
			maxTokens: options.maxTokens,
			thinkingLevel: options.thinkingLevel,
		});
	};

	const llmResult = await compactToFit(shakeResult.messages, {
		triggerTokens: 0,
		summarize,
	});

	return {
		changed: llmResult.changed,
		messages: llmResult.messages,
		tokensBefore,
		tokensAfter: llmResult.tokensAfter,
	};
}

/** Whether auto-compaction should fire given current settings + message history. */
export async function shouldAutoCompact(
	settings: CompactionSettings,
	messages: Message[],
	tokenEncoding?: string,
): Promise<boolean> {
	if (!settings.enabled) return false;
	const contextWindow = settings.contextWindow ?? 128000;
	const threshold = contextWindow - (settings.reserveTokens ?? 16384);
	const currentTokens = await estimateChatPayloadTokens(
		messages,
		undefined,
		tokenEncoding,
	);
	return currentTokens > threshold;
}
