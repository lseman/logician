// ── LLM Backend ──────────────────────────────────────────────────────────────────
// OpenAI-compatible HTTP client for streaming LLM responses.
// Mirrors Python LlamaCppClient/VLLMClient but simplified for TS.

import type { ThinkingLevel, ToolCall } from "../types/index.ts";

// ── Typed backend errors ───────────────────────────────────────────────────
// The backend classifies provider/network failures at the boundary so the loop
// can branch on a category instead of re-sniffing error message strings. The
// loop keeps a string-matching fallback for errors thrown outside the backend.

export type BackendErrorCategory =
	// Prompt exceeds the model's context window. Recover by compacting, not by
	// retrying the same request.
	| "context_full"
	// Provider rate limit (HTTP 429). Retryable with backoff.
	| "rate_limit"
	// Transient server / network failure (HTTP 5xx, connection errors).
	// Retryable with backoff.
	| "transient"
	// Client error (HTTP 4xx other than 429): malformed request. Not retryable.
	| "client"
	// A tool call already stored in history has arguments the provider can't
	// parse as JSON. Retrying resends the identical unparseable history and
	// fails identically every time; compaction doesn't help either since it
	// never inspects/repairs individual tool_calls. Not retryable.
	| "poisoned_history"
	// Anything the backend couldn't classify.
	| "unknown";

export class BackendError extends Error {
	readonly category: BackendErrorCategory;
	readonly status?: number;
	/** Whether retrying the same request could succeed (rate_limit / transient). */
	readonly retryable: boolean;
	/** Provider-requested retry delay (Retry-After header), when present. */
	readonly retryAfterMs?: number;

	constructor(opts: {
		category: BackendErrorCategory;
		message: string;
		status?: number;
		retryAfterMs?: number;
	}) {
		super(opts.message);
		this.name = "BackendError";
		this.category = opts.category;
		this.status = opts.status;
		this.retryAfterMs = opts.retryAfterMs;
		this.retryable =
			opts.category === "rate_limit" || opts.category === "transient";
	}
}

// Classify an HTTP error response by status + body. Context-full is detected
// from the body text since providers signal it inconsistently (400 or 413 with
// a "context"/"too long"/"tokens" message).
export function classifyHttpError(
	status: number,
	body: string,
	retryAfterHeader?: string | null,
): BackendError {
	const lower = body.toLowerCase();
	// "Failed to parse tool call arguments as JSON" means a previously-stored
	// assistant message has a tool_call whose arguments are malformed (usually
	// truncated by the output token limit before it was saved). Resending the
	// same history always fails the same way — this must not be treated as
	// transient/retryable.
	const looksPoisonedHistory = [
		"failed to parse tool call arguments",
		"failed to parse tool calls",
		"invalid tool call arguments",
	].some(p => lower.includes(p));
	if (looksPoisonedHistory) {
		return new BackendError({
			category: "poisoned_history",
			message: `LLM request failed: ${status} ${body}`,
			status,
		});
	}
	// "Assistant message must contain either 'content' or 'tool_calls'" is
	// NOT a context-full error — it means a previously-stored assistant message
	// is malformed (empty content, no tool_calls).  Compaction won't fix it;
	// the same bad message would be resent.  Classify as client so the loop
	// can recover by compacting (which drops the bad message) or aborting.
	const looksContextFull = [
		"context",
		"too long",
		"too many tokens",
		"maximum context",
		"reduce the length",
		"n_ctx",
	].some(p => lower.includes(p));
	const message = `LLM request failed: ${status} ${body}`;

	if (looksContextFull) {
		return new BackendError({ category: "context_full", message, status });
	}
	if (status === 429) {
		return new BackendError({
			category: "rate_limit",
			message,
			status,
			retryAfterMs: parseRetryAfter(retryAfterHeader),
		});
	}
	if (status >= 500) {
		return new BackendError({ category: "transient", message, status });
	}
	if (status >= 400) {
		return new BackendError({ category: "client", message, status });
	}
	return new BackendError({ category: "unknown", message, status });
}

// Parse a Retry-After header: either delay-seconds or an HTTP date. Returns
// milliseconds, clamped to [0, 5 min]; undefined when absent/unparseable.
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

// Classify a thrown network/fetch error (no HTTP response). Connection-level
// failures are transient; an abort is rethrown unchanged by the caller.
export function classifyNetworkError(error: Error): BackendError {
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
	return new BackendError({
		category: transient ? "transient" : "unknown",
		message: error.message,
	});
}

export interface LLMResponse {
	content: string | null;
	toolCalls: ToolCall[];
	stopReason: "stop" | "length" | "error";
	errorMessage?: string;
	// Provider-reported token usage from the final stream chunk, when available.
	// Lets the loop report real context size instead of a local char/4 estimate.
	usage?: {
		promptTokens?: number;
		completionTokens?: number;
		totalTokens?: number;
		/** Prompt tokens served from the provider's cache, when reported. */
		cachedTokens?: number;
	};
}

interface ProviderUsage {
	prompt_tokens?: unknown;
	completion_tokens?: unknown;
	total_tokens?: unknown;
	cached_tokens?: unknown;
	cache_read_input_tokens?: unknown;
	prompt_tokens_details?: { cached_tokens?: unknown };
	input_tokens_details?: { cached_tokens?: unknown };
}

interface ProviderTimings {
	cache_n?: unknown;
}

function tokenCount(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value >= 0
		? Math.floor(value)
		: undefined;
}

/** Normalize OpenAI-compatible (including llama.cpp) usage telemetry. */
export function parseProviderUsage(
	raw: unknown,
	rawTimings?: unknown,
): LLMResponse["usage"] {
	if (
		(!raw || typeof raw !== "object") &&
		(!rawTimings || typeof rawTimings !== "object")
	) {
		return undefined;
	}
	const usage = (raw && typeof raw === "object" ? raw : {}) as ProviderUsage;
	const timings = (
		rawTimings && typeof rawTimings === "object" ? rawTimings : {}
	) as ProviderTimings;
	const cachedTokens = tokenCount(
		usage.prompt_tokens_details?.cached_tokens ??
			usage.input_tokens_details?.cached_tokens ??
			usage.cache_read_input_tokens ??
			usage.cached_tokens ??
			timings.cache_n,
	);
	return {
		promptTokens: tokenCount(usage.prompt_tokens),
		completionTokens: tokenCount(usage.completion_tokens),
		totalTokens: tokenCount(usage.total_tokens),
		...(cachedTokens !== undefined && { cachedTokens }),
	};
}

/**
 * Re-role any system message that is not at the start of the array to `user`.
 *
 * The loop appends request-time context (memory index, task state) as trailing
 * system messages so the leading system prompt — the cacheable prefix — stays
 * stable. Chat templates behind many OpenAI-compatible servers (SGLang, vLLM,
 * llama.cpp) only accept a system/developer message at position 0 and raise
 * "System message must be at the beginning" for any later occurrence. `user`
 * is accepted by every chat template, so trailing system context is re-roled
 * here at the transport boundary instead of changing the loop's logical
 * message model or the prompt-cache prefix.
 */
export function normalizeProviderMessages(
	messages: Record<string, unknown>[],
): Record<string, unknown>[] {
	let sawNonSystem = false;
	return messages.map(message => {
		if (message.role === "system" && sawNonSystem) {
			return { ...message, role: "user" };
		}
		if (message.role !== "system") sawNonSystem = true;
		return message;
	});
}

/** Streaming callbacks for a generate() call. All optional. */
export interface GenerateCallbacks {
	onDelta?: (delta: string) => void;
	onThinking?: (delta: string) => void;
	onTextStart?: () => void;
	onTextEnd?: () => void;
	// Fired once per streamed tool call, the moment its name is first known
	// (placeholder/empty-name chunks are skipped). Gives the UI an early
	// "running" state while the model is still emitting the call's arguments.
	onToolCallStart?: (toolCallId: string, name: string, args: string) => void;
	onToolCallDelta?: (toolCallId: string, delta: string) => void;
	onToolCallIdUpdate?: (previousToolCallId: string, toolCallId: string) => void;
}

/** Options for a generate() call. */
export interface GenerateOptions {
	tools?: Record<string, unknown>[];
	temperature?: number;
	maxTokens?: number;
	// Additional sampling params (populated when an inference mode is active).
	topP?: number;
	topK?: number;
	minP?: number;
	presencePenalty?: number;
	repetitionPenalty?: number;
	signal?: AbortSignal;
	thinkingLevel?: ThinkingLevel;
	callbacks?: GenerateCallbacks;
	// Per-request extras supplied by the loop's provider-boundary hooks
	// (beforeProviderRequest / beforeProviderPayload).
	headers?: Record<string, string>;
	/** Per-request deadline. Combined with the caller's cancellation signal. */
	timeoutMs?: number;
	transformPayload?: (
		payload: Record<string, unknown>,
	) => Promise<Record<string, unknown>> | Record<string, unknown>;
	// Max retry attempts for this request (overrides config default).
	maxRetries?: number;
	// Cache retention hint forwarded to providers supporting it.
	cacheRetention?: string;
	// Provider metadata forwarded with requests.
	metadata?: Record<string, unknown>;
}

export interface LLMBackend {
	generate(
		messages: Record<string, unknown>[],
		options?: GenerateOptions,
	): Promise<LLMResponse>;

	/** Return a backend identical to this one but bound to a different model. */
	withModel(model: string): LLMBackend;
	/** Clone the backend with both model and provider endpoint when supported. */
	withEndpoint?(model: string, baseUrl: string): LLMBackend;

	/** The model this backend currently targets. */
	readonly model: string;
}

export class OpenAIBackend implements LLMBackend {
	readonly baseUrl: string;
	readonly model: string;
	private chatTemplate?: string;
	private stop?: string[];
	private defaultThinkingLevel: ThinkingLevel = "off";

	constructor(options: {
		baseUrl: string;
		model: string;
		chatTemplate?: string;
		stop?: string[];
		thinkingLevel?: ThinkingLevel;
	}) {
		this.baseUrl = options.baseUrl.replace(/\/+$/, "");
		this.model = options.model;
		this.chatTemplate = options.chatTemplate;
		this.stop = options.stop;
		this.defaultThinkingLevel = options.thinkingLevel ?? "off";
	}

	/** Clone this backend bound to a different model (LLMBackend.withModel). */
	withModel(model: string): OpenAIBackend {
		return new OpenAIBackend({
			baseUrl: this.baseUrl,
			model,
			chatTemplate: this.chatTemplate,
			stop: this.stop,
			thinkingLevel: this.defaultThinkingLevel,
		});
	}

	withEndpoint(model: string, baseUrl: string): OpenAIBackend {
		return new OpenAIBackend({
			baseUrl,
			model,
			chatTemplate: this.chatTemplate,
			stop: this.stop,
			thinkingLevel: this.defaultThinkingLevel,
		});
	}

	/** Update the default thinking level at runtime. */
	setDefaultThinkingLevel(level: ThinkingLevel): void {
		this.defaultThinkingLevel = level;
	}

	async generate(
		messages: Record<string, unknown>[],
		options: GenerateOptions = {},
	): Promise<LLMResponse> {
		const {
			tools,
			temperature = 0.5,
			maxTokens = 4096,
			topP,
			topK,
			minP,
			presencePenalty,
			repetitionPenalty,
			signal,
			thinkingLevel,
			callbacks = {},
			headers: extraHeaders,
			timeoutMs,
			transformPayload,
		} = options;
		const {
			onDelta,
			onThinking,
			onTextStart,
			onTextEnd,
			onToolCallStart,
			onToolCallDelta,
			onToolCallIdUpdate,
		} = callbacks;

		const providerMessages = normalizeProviderMessages(messages);

		const body: Record<string, unknown> = {
			model: this.model,
			messages: providerMessages,
			temperature,
			max_tokens: maxTokens,
			stream: true,
			// Ask OpenAI-compatible providers to emit a final usage chunk so the
			// loop can report real token counts instead of a local estimate.
			stream_options: { include_usage: true },
			// llama.cpp: reuse KV cache across turns instead of recomputing the prefix.
			cache_prompt: true,
			...(this.stop && { stop: this.stop }),
			// Additional sampling params (populated when an inference mode is active).
			...(topP !== undefined && { top_p: topP }),
			...(topK !== undefined && { top_k: topK }),
			...(minP !== undefined && { min_p: minP }),
			...(presencePenalty !== undefined && {
				presence_penalty: presencePenalty,
			}),
			...(repetitionPenalty !== undefined && {
				repetition_penalty: repetitionPenalty,
			}),
		};

		// Pass reasoning/thinking level to OpenAI-compatible providers.
		// "off" = omit reasoning field entirely.
		const effectiveLevel = thinkingLevel ?? this.defaultThinkingLevel;
		if (effectiveLevel !== "off") {
			body.reasoning = { effort: effectiveLevel };
		}

		if (tools && tools.length > 0) {
			body.tools = tools;
		}

		// Let a provider-payload hook inspect/rewrite the final body.
		const finalBody = transformPayload ? await transformPayload(body) : body;

		const timeoutSignal =
			timeoutMs !== undefined && timeoutMs > 0
				? AbortSignal.timeout(timeoutMs)
				: undefined;
		const requestSignal =
			signal && timeoutSignal
				? AbortSignal.any([signal, timeoutSignal])
				: (signal ?? timeoutSignal);
		let response: Response;
		try {
			response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					...extraHeaders,
				},
				body: JSON.stringify(finalBody),
				signal: requestSignal,
			});
		} catch (e) {
			const error = e as Error;
			// Aborts propagate unchanged so the loop's signal check handles them.
			if (error.name === "AbortError") throw error;
			throw classifyNetworkError(error);
		}

		if (!response.ok) {
			const errorText = await response.text();
			throw classifyHttpError(
				response.status,
				errorText,
				response.headers.get("retry-after"),
			);
		}

		if (!response.body) {
			throw new BackendError({
				category: "transient",
				message: "Response body is unavailable",
			});
		}

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		let fullContent = "";
		let toolCalls: ToolCall[] = [];
		let stopReason: LLMResponse["stopReason"] = "stop";
		let finishReason: string | undefined;
		let usage: LLMResponse["usage"];
		let hasText = false;
		// Tool-call indices whose early start event has already been emitted, so
		// onToolCallStart fires exactly once per streamed call.
		const startedToolIndexes = new Set<number>();
		while (true) {
			const { value, done } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });

			const lines = buffer.split("\n");
			buffer = lines.pop() || "";

			for (const line of lines) {
				if (!line.startsWith("data: ")) continue;
				const data = line.slice(6).trim();
				if (!data || data === "[DONE]") continue;

				try {
					const chunk = JSON.parse(data);
					// finish_reason and usage can arrive on chunks that carry no delta
					// (notably the trailing usage-only chunk), so read them first.
					const chunkFinish = chunk.choices?.[0]?.finish_reason;
					if (chunkFinish) finishReason = chunkFinish;
					if (chunk.usage || chunk.timings) {
						const parsed = parseProviderUsage(chunk.usage, chunk.timings);
						usage = {
							promptTokens: parsed?.promptTokens ?? usage?.promptTokens,
							completionTokens:
								parsed?.completionTokens ?? usage?.completionTokens,
							totalTokens: parsed?.totalTokens ?? usage?.totalTokens,
							cachedTokens: parsed?.cachedTokens ?? usage?.cachedTokens,
						};
					}
					const delta = chunk.choices?.[0]?.delta;
					if (!delta) continue;

					// Emit text_start on first text content
					if (delta.content && !hasText) {
						hasText = true;
						onTextStart?.();
					}

					if (delta.content) {
						onDelta?.(delta.content);
						fullContent += delta.content;
					}

					if (delta.reasoning) {
						onThinking?.(delta.reasoning);
					}
					if (delta.reasoning_content) {
						onThinking?.(delta.reasoning_content);
					}

					if (delta.tool_calls) {
						for (const tc of delta.tool_calls) {
							// Accumulate tool call across chunks.
							if (!toolCalls[tc.index]) {
								toolCalls[tc.index] = {
									id: "",
									name: "",
									arguments: "",
								};
							}
							if (tc.id) {
								const previousId = toolCalls[tc.index].id;
								if (!previousId && startedToolIndexes.has(tc.index)) {
									onToolCallIdUpdate?.(`tool_${tc.index}`, tc.id);
								}
								toolCalls[tc.index].id = tc.id;
							}
							if (tc.function?.name)
								toolCalls[tc.index].name = tc.function.name;
							if (tc.function?.arguments) {
								toolCalls[tc.index].arguments += tc.function.arguments;
							}

							// Emit the early start once, the moment the name is known.
							// Skipping empty-name chunks lets the UI reuse this chunk
							// (by id/name) when the loop emits the authoritative start
							// before execution — no duplicate card.
							const acc = toolCalls[tc.index];
							if (acc.name && !startedToolIndexes.has(tc.index)) {
								startedToolIndexes.add(tc.index);
								onToolCallStart?.(
									acc.id || `tool_${tc.index}`,
									acc.name,
									acc.arguments,
								);
							}
							if (tc.function?.arguments && startedToolIndexes.has(tc.index)) {
								onToolCallDelta?.(
									acc.id || `tool_${tc.index}`,
									tc.function.arguments,
								);
							}
						}
					}
				} catch (_e: unknown) {
					// Skip parse errors (partial JSON is normal in streaming)
				}
			}
		}

		// Emit text_end after streaming completes (before tool_call_end which
		// is emitted by the loop layer after parsing).
		if (hasText) {
			onTextEnd?.();
		}

		toolCalls = toolCalls
			.filter(tc => tc?.name)
			.map((tc, index) => ({
				id: tc.id || `tool_${index}`,
				name: tc.name,
				arguments: tc.arguments || "{}",
			}));

		// Map the provider finish_reason to our stop reason. "length" means the
		// completion was truncated by max_tokens — the loop surfaces this rather
		// than treating it as a clean stop. Tool calls override (handled by the
		// loop, which sets "tool_calls").
		if (finishReason === "length") {
			stopReason = "length";
		} else if (toolCalls.length > 0) {
			stopReason = "stop";
		}

		return {
			content: fullContent || null,
			toolCalls,
			stopReason,
			usage,
		};
	}
}
