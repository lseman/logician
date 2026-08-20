// ── Assistant call retry ─────────────────────────────────────────────────
// Bounded-attempt retry loop for a single streamed assistant response,
// keyed off the transport error category our own openai-completions.ts
// adapter attaches to AssistantMessage.errorCategory. Unlike pi-ai's
// retryAssistantCall (which classifies retryability from a provider-specific
// error-message regex), this classifies from the structured TransportError
// category recorded at the HTTP boundary — see ai/errors.ts.

import type { RetryCallbacks, RetryPolicy } from "../core/retry.ts";
import type { AssistantMessage } from "./types.ts";

class RetrySleepAbortError extends Error {
	constructor() {
		super("Aborted");
	}
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) {
			reject(new RetrySleepAbortError());
			return;
		}
		const timeout = setTimeout(resolve, ms);
		const onAbort = () => {
			clearTimeout(timeout);
			reject(new RetrySleepAbortError());
		};
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

/** Whether a failed assistant message looks retryable, per our transport error classification. */
export function isRetryableAssistantError(message: AssistantMessage): boolean {
	if (message.stopReason !== "error") return false;
	return (
		message.errorCategory === "rate_limit" ||
		message.errorCategory === "transient"
	);
}

/**
 * Retry a streamed assistant call with exponential backoff, bounded by `policy`.
 * Aborted responses are returned immediately without retrying. Non-retryable errors
 * (client errors, poisoned history, context-full) are returned immediately too —
 * callers should handle those categories (e.g. compaction on context_full) before retrying.
 */
export async function retryAssistantCall(
	produce: () => Promise<AssistantMessage>,
	policy: RetryPolicy | undefined,
	signal: AbortSignal | undefined,
	callbacks?: RetryCallbacks,
): Promise<AssistantMessage> {
	const maxAttempts = policy?.enabled ? policy.maxRetries : 0;

	let attempt = 0;
	for (;;) {
		const response = await produce();

		if (response.stopReason === "aborted") return response;
		if (response.stopReason !== "error") return response;
		if (attempt >= maxAttempts || !isRetryableAssistantError(response))
			return response;

		attempt++;
		const delayMs = (policy?.baseDelayMs ?? 1000) * 2 ** (attempt - 1);
		await callbacks?.onRetryScheduled?.(
			attempt,
			maxAttempts,
			delayMs,
			response.errorMessage || "Unknown error",
		);

		try {
			await sleep(delayMs, signal);
		} catch {
			return {
				...response,
				stopReason: "aborted",
				errorMessage: undefined,
				errorCategory: undefined,
			};
		}
		await callbacks?.onRetryStarted?.(attempt, maxAttempts);
	}
}
