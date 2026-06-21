// ── Context-full recovery ────────────────────────────────────────────────────
// Extracted from loop.ts callLLM() to handle "context_full" LLM errors.
// Calls compactToFit with forced ladder (triggerTokens=0) and returns the
// new messages + metrics for the caller to emit.

import { compactToFit } from "./compaction.ts";
import type { CompactableMessage } from "../core/types.ts";

const COMPACTION_TARGET_FRACTION = 0.65;

export interface ContextFullRecoveryOptions {
	messages: CompactableMessage[];
	contextWindowTokens: number;
}

export interface ContextFullRecoveryResult {
	/** Whether compaction produced new messages. */
	success: boolean;
	/** Compacted messages (same as input if success=false). */
	messages: CompactableMessage[];
	/** Token count before compaction. */
	tokensBefore: number;
	/** Token count after compaction. */
	tokensAfter: number;
}

/**
 * Handle a "context_full" LLM error by compacting the conversation.
 * Forces compaction regardless of local estimate (the provider already
 * rejected the request as too long).
 */
export function recoverFromContextFull(
	opts: ContextFullRecoveryOptions,
): ContextFullRecoveryResult {
	const { messages, contextWindowTokens } = opts;
	const targetTokens = Math.floor(
		contextWindowTokens * COMPACTION_TARGET_FRACTION,
	);

	const result = compactToFit(messages, {
		triggerTokens: 0, // Force: provider already rejected
		targetTokens,
	});

	if (!result.changed) {
		return { success: false, messages, tokensBefore: 0, tokensAfter: 0 };
	}

	return {
		success: true,
		messages: result.messages,
		tokensBefore: result.tokensBefore,
		tokensAfter: result.tokensAfter,
	};
}
