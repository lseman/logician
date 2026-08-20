// ── Retry policy ──────────────────────────────────────────────────────────
// Ported from pi-ai's utils/retry.ts (policy shape only — the bounded-attempt
// exponential-backoff loop itself lives with whichever caller drives retries).

/** Retry policy: bounded attempts with exponential backoff (`baseDelayMs * 2^(attempt-1)`). */
export interface RetryPolicy {
	enabled: boolean;
	/** Max retry attempts (0 = no retries). The initial call never counts as a retry. */
	maxRetries: number;
	/** Base delay in ms. Per-attempt delay is `baseDelayMs * 2^(attempt-1)` before jitter. */
	baseDelayMs: number;
}

/** Optional callbacks emitted around each retry. */
export interface RetryCallbacks {
	/** Emitted before the backoff sleep of each retry attempt (1-indexed). */
	onRetryScheduled?: (
		attempt: number,
		maxAttempts: number,
		delayMs: number,
		errorMessage: string,
	) => void | Promise<void>;
	/** Emitted after the backoff sleep, immediately before the retried call starts. */
	onRetryStarted?: (
		attempt: number,
		maxAttempts: number,
	) => void | Promise<void>;
}
