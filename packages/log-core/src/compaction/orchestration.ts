// ── Compaction operations for AgentSession ────────────────────────────────
// Pulled out of harness.ts: manual compact(), auto-compaction threshold
// check, and the shared before/after-compact event + hook plumbing.
// Delegates to the single compaction engine (compactToFit) shared with the
// loop's context-full retry and the builtin proactive hook.

import type { FrameConfig } from "@logician/log-snapcompact";
import type { LLMBackend } from "../provider/backend.ts";
import { estimateChatPayloadTokens } from "../provider/messages.ts";
import { generateCompactionSummary } from "../session/summaries/summary-generation.ts";
import type { ThinkingLevel } from "../types/config.ts";
import type { CompactableMessage, Message } from "../types/messages.ts";
import {
	type CompactionSettings,
	type CompactToFitResult,
	compactToFit,
	shakeCompaction,
} from "./engine.ts";

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
		/** Recent-context budget the summarizing pass leaves verbatim. */
		keepRecentTokens?: number | undefined;
	},
): Promise<CompactionOutcome> {
	const { mode = "auto" } = options;

	// Shake: drop recoverable heavy content without an LLM. It is a cheap
	// pre-pass, not a gate — a history with nothing heavy to drop (plain chat)
	// must still reach the summarizing pass of the requested mode.
	const shakeResult = shakeCompaction(history as CompactableMessage[], {});
	const shaken: CompactionOutcome = shakeResult.changed
		? {
				changed: true,
				messages: shakeResult.messages,
				tokensBefore,
				tokensAfter: shakeResult.tokensAfter,
			}
		: {
				changed: false,
				messages: history as CompactableMessage[],
				tokensBefore,
				tokensAfter: tokensBefore,
			};

	// Shake-only mode: use the shake result as-is.
	if (mode === "shake") return shaken;

	// Auto mode: if shake alone saved enough (8k+), stop there.
	if (mode === "auto" && tokensBefore - shaken.tokensAfter >= 8000) {
		return shaken;
	}

	const keep =
		options.keepRecentTokens !== undefined
			? { keepRecentTokens: options.keepRecentTokens }
			: {};
	let pass: CompactToFitResult;
	if (mode === "remote") {
		// Provider-native compaction endpoint.
		const remoteSummarizer = async (older: CompactableMessage[]) => {
			const result = await backend.remote(
				older as unknown as Record<string, unknown>[],
				{ maxTokens: options.maxTokens ?? 2048 },
			);
			return result.summary;
		};
		pass = await compactToFit(shaken.messages, {
			triggerTokens: 0,
			remoteSummarizer,
			settings: { mode: "remote", ...keep },
		});
	} else if (mode === "snapcompact") {
		// Local, deterministic frame rendering — no LLM call.
		pass = await compactToFit(shaken.messages, {
			triggerTokens: 0,
			settings: {
				mode: "snapcompact",
				...keep,
				...(options.frameOptions ? { frameOptions: options.frameOptions } : {}),
			},
		});
	} else {
		// LLM (or auto that shake couldn't satisfy): summarize older turns.
		const summarize = async (older: CompactableMessage[]) => {
			if (options.presetSummary) return options.presetSummary;
			return generateCompactionSummary(backend, older as Message[], [], {
				temperature: options.temperature,
				maxTokens: options.maxTokens,
				thinkingLevel: options.thinkingLevel,
			});
		};
		pass = await compactToFit(shaken.messages, {
			triggerTokens: 0,
			summarize,
			settings: keep,
		});
	}

	// The summarizing pass may find nothing to cut (everything fits in the
	// kept tail); shake's savings still stand in that case.
	return pass.changed
		? {
				changed: true,
				messages: pass.messages,
				tokensBefore,
				tokensAfter: pass.tokensAfter,
			}
		: shaken;
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
