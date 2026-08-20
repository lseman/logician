// ── Transport errors ─────────────────────────────────────────────────────
// Classifies provider/network failures at the HTTP boundary so callers can
// branch on a stable category instead of re-sniffing error message strings.
// Ported from our previous agent-core's utils/backend.ts.

export type TransportErrorCategory =
	// Prompt exceeds the model's context window. Recover by compacting, not by
	// retrying the same request.
	| "context_full"
	// Provider rate limit (HTTP 429). Retryable with backoff.
	| "rate_limit"
	// Transient server / network failure (HTTP 5xx, connection errors). Retryable with backoff.
	| "transient"
	// Client error (HTTP 4xx other than 429): malformed request. Not retryable.
	| "client"
	// A tool call already stored in history has arguments the provider can't parse as JSON.
	// Retrying resends the identical unparseable history and fails identically every time;
	// compaction doesn't help either since it never inspects/repairs individual tool calls.
	| "poisoned_history"
	// Anything the backend couldn't classify.
	| "unknown";

export class TransportError extends Error {
	readonly category: TransportErrorCategory;
	readonly status?: number;
	/** Whether retrying the same request could succeed (rate_limit / transient). */
	readonly retryable: boolean;
	/** Provider-requested retry delay (Retry-After header), when present. */
	readonly retryAfterMs?: number;

	constructor(opts: {
		category: TransportErrorCategory;
		message: string;
		status?: number;
		retryAfterMs?: number;
	}) {
		super(opts.message);
		this.name = "TransportError";
		this.category = opts.category;
		this.status = opts.status;
		this.retryAfterMs = opts.retryAfterMs;
		this.retryable =
			opts.category === "rate_limit" || opts.category === "transient";
	}
}

/**
 * Classify an HTTP error response by status + body. Context-full is detected from the body
 * text since providers signal it inconsistently (400 or 413 with a "context"/"too long"/"tokens" message).
 */
export function classifyHttpError(
	status: number,
	body: string,
	retryAfterHeader?: string | null,
): TransportError {
	const lower = body.toLowerCase();
	const looksPoisonedHistory = [
		"failed to parse tool call arguments",
		"failed to parse tool calls",
		"invalid tool call arguments",
	].some(p => lower.includes(p));
	if (looksPoisonedHistory) {
		return new TransportError({
			category: "poisoned_history",
			message: `LLM request failed: ${status} ${body}`,
			status,
		});
	}

	const looksContextFull = [
		"context",
		"too long",
		"too many tokens",
		"maximum context",
		"reduce the length",
		"n_ctx",
	].some(p => lower.includes(p));
	const message = `LLM request failed: ${status} ${body}`;

	if (looksContextFull)
		return new TransportError({ category: "context_full", message, status });
	if (status === 429)
		return new TransportError({
			category: "rate_limit",
			message,
			status,
			retryAfterMs: parseRetryAfter(retryAfterHeader),
		});
	if (status >= 500)
		return new TransportError({ category: "transient", message, status });
	if (status >= 400)
		return new TransportError({ category: "client", message, status });
	return new TransportError({ category: "unknown", message, status });
}

/** Parse a Retry-After header: either delay-seconds or an HTTP date. Returns ms, clamped to [0, 5 min]. */
function parseRetryAfter(header?: string | null): number | undefined {
	if (!header) return undefined;
	const trimmed = header.trim();
	const seconds = Number(trimmed);
	let ms: number;
	if (Number.isFinite(seconds)) {
		ms = seconds * 1000;
	} else {
		const date = Date.parse(trimmed);
		if (Number.isNaN(date)) return undefined;
		ms = date - Date.now();
	}
	return Math.min(Math.max(ms, 0), 5 * 60_000);
}

/** Classify a thrown network/fetch error (no HTTP response). Connection-level failures are transient. */
export function classifyNetworkError(error: Error): TransportError {
	const msg = `${error.name || ""} ${error.message || ""}`.toLowerCase();
	const transient = [
		"econnrefused",
		"econnreset",
		"etimedout",
		"eai-again",
		"socket hang up",
		"connection refused",
		"connection reset",
		"connection timeout",
		"network error",
		"fetch failed",
	].some(p => msg.includes(p));
	return new TransportError({
		category: transient ? "transient" : "unknown",
		message: error.message,
	});
}
