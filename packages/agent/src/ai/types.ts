// ── AI transport types ──────────────────────────────────────────────────────
// Provider-neutral message/context/event vocabulary, modeled on pi-ai's
// types.ts (@earendil-works/pi-ai) but scoped to what agent actually
// needs: a single OpenAI-compatible chat-completions API, matching what our
// previous agent-core's backend.ts supported. Extend `KnownApi` and add an
// adapter under `ai/` when another provider is needed.

import type { TSchema } from "@sinclair/typebox";

export type KnownApi = "openai-completions";
export type Api = KnownApi | (string & {});

export type KnownProvider = "openai-compatible";
export type ProviderId = KnownProvider | (string & {});

export type ToolChoice = "auto" | "none";
export type ThinkingLevel =
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh"
	| "max";
export type ModelThinkingLevel = "off" | ThinkingLevel;

export type CacheRetention = "none" | "short" | "long";

export interface TextContent {
	type: "text";
	text: string;
}

export interface ThinkingContent {
	type: "thinking";
	thinking: string;
}

export interface ImageContent {
	type: "image";
	data: string; // base64 encoded image data
	mimeType: string;
}

export interface ToolCall {
	type: "toolCall";
	id: string;
	name: string;
	arguments: Record<string, unknown>;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

export type StopReason =
	| "pending"
	| "stop"
	| "length"
	| "toolUse"
	| "error"
	| "aborted";

export type JsonValue =
	| string
	| number
	| boolean
	| null
	| JsonValue[]
	| { [key: string]: JsonValue };

/**
 * Durable handle for a provider's async/deferred response, letting a caller poll or cancel it
 * later. Not currently produced by ai/openai-completions.ts (no deferred-response support yet) —
 * defined here as a stable shape the session-log/harness layer can reference structurally.
 */
export interface DeferredHandle {
	provider: string;
	modelId: string;
	api: string;
	/** Provider token, such as a response id or batch id plus row id. */
	id: string;
	expiresAt?: number;
	pollAfterMs?: number;
	/** Provider conversion data required to reconstruct the final assistant message. */
	data?: JsonValue;
}

export interface UserMessage {
	role: "user";
	content: string | (TextContent | ImageContent)[];
	timestamp: number;
}

export interface AssistantMessage {
	role: "assistant";
	content: (TextContent | ThinkingContent | ToolCall)[];
	api: Api;
	provider: ProviderId;
	model: string;
	usage: Usage;
	stopReason: StopReason;
	errorMessage?: string;
	/** Set on stopReason "error": the transport error category from ai/errors.ts, for retry classification. */
	errorCategory?: string;
	timestamp: number;
}

export interface ToolResultMessage<TDetails = unknown> {
	role: "toolResult";
	toolCallId: string;
	toolName: string;
	content: (TextContent | ImageContent)[];
	details?: TDetails;
	usage?: Usage;
	isError: boolean;
	timestamp: number;
}

export type Message = UserMessage | AssistantMessage | ToolResultMessage;

export interface Tool<TParameters extends TSchema = TSchema> {
	name: string;
	description: string;
	parameters: TParameters;
}

export interface Context {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
}

export interface ModelCostRates {
	input: number; // $/million tokens
	output: number; // $/million tokens
	cacheRead: number; // $/million tokens
	cacheWrite: number; // $/million tokens
}

// Model interface. Trimmed of the multi-provider `compat` matrix pi-ai carries —
// agent targets one API shape, so provider quirks (if any) live in the
// adapter itself rather than a per-provider compat config.
export interface Model<TApi extends Api = Api> {
	id: string;
	name: string;
	api: TApi;
	provider: ProviderId;
	baseUrl: string;
	reasoning: boolean;
	contextWindow: number;
	maxTokens: number;
	cost: ModelCostRates;
	samplingParams?: Record<string, unknown>;
	headers?: Record<string, string>;
}

export interface StreamOptions {
	signal?: AbortSignal;
	apiKey?: string;
	fetch?: typeof globalThis.fetch;
	headers?: Record<string, string>;
	timeoutMs?: number;
	maxRetries?: number;
	/** Maximum delay in milliseconds to wait for a provider-requested retry. */
	maxRetryDelayMs?: number;
	temperature?: number;
	maxTokens?: number;
	samplingParams?: Record<string, unknown>;
	cacheRetention?: CacheRetention;
	metadata?: Record<string, unknown>;
	onPayload?: (
		payload: Record<string, unknown>,
	) =>
		| Record<string, unknown>
		| undefined
		| Promise<Record<string, unknown> | undefined>;
}

export interface SimpleStreamOptions extends StreamOptions {
	toolChoice?: ToolChoice;
	reasoning?: ThinkingLevel;
}

/**
 * Event protocol for AssistantMessageEventStream.
 *
 * Streams emit `start` before partial updates, then terminate with either:
 * - `done` carrying the final successful AssistantMessage, or
 * - `error` carrying the final AssistantMessage with stopReason "error" or "aborted"
 *   and errorMessage.
 */
export type AssistantMessageEvent =
	| { type: "start"; partial: AssistantMessage }
	| { type: "text_start"; contentIndex: number; partial: AssistantMessage }
	| {
			type: "text_delta";
			contentIndex: number;
			delta: string;
			partial: AssistantMessage;
	  }
	| {
			type: "text_end";
			contentIndex: number;
			content: string;
			partial: AssistantMessage;
	  }
	| { type: "thinking_start"; contentIndex: number; partial: AssistantMessage }
	| {
			type: "thinking_delta";
			contentIndex: number;
			delta: string;
			partial: AssistantMessage;
	  }
	| {
			type: "thinking_end";
			contentIndex: number;
			content: string;
			partial: AssistantMessage;
	  }
	| { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
	| {
			type: "toolcall_delta";
			contentIndex: number;
			delta: string;
			partial: AssistantMessage;
	  }
	| {
			type: "toolcall_end";
			contentIndex: number;
			toolCall: ToolCall;
			partial: AssistantMessage;
	  }
	| {
			type: "done";
			reason: Extract<StopReason, "stop" | "length" | "toolUse">;
			message: AssistantMessage;
	  }
	| {
			type: "error";
			reason: Extract<StopReason, "aborted" | "error">;
			error: AssistantMessage;
	  };

/**
 * Stream returned by a {@link StreamFn}: async-iterable over protocol events, and resolves the
 * final AssistantMessage via `.result()` once a terminal (`done`/`error`) event has been pushed.
 * The concrete implementation lives in ai/event-stream.ts to avoid a value/type import cycle.
 */
export interface AssistantMessageEventStream
	extends AsyncIterable<AssistantMessageEvent> {
	result(): Promise<AssistantMessage>;
}

/**
 * Stream function used by the agent loop.
 *
 * Contract:
 * - Must not throw or return a rejected promise for request/model/runtime failures.
 * - Must return an AssistantMessageEventStream.
 * - Failures must be encoded in the returned stream via protocol events and a
 *   final AssistantMessage with stopReason "error" or "aborted" and errorMessage.
 */
export type StreamFn = (
	model: Model,
	context: Context,
	options?: SimpleStreamOptions,
) => AssistantMessageEventStream | Promise<AssistantMessageEventStream>;
