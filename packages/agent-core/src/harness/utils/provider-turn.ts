/**
 * One provider request.
 *
 * Builds the request (hooks for headers/timeout/retries/cache/metadata,
 * inference mode resolution, `beforeProviderPayload`), calls
 * `backend.generate()`, and returns its response or a terminal outcome.
 * Mirrors pi's agent-loop.ts: any non-cancellation error ends the turn —
 * retry/backoff responsibility belongs to the backend's stream
 * implementation, not the loop (see core/backend.ts).
 */

import type {
	AgentConfig,
	AgentEvent,
	AgentEventSink,
	AgentMessage,
	Message,
} from "../../types/index.ts";
import { getInferenceMode } from "../../types/types-config.ts";
import type { RunOutcomeStatus } from "../../types/types-messages.ts";
import type { AgentSettings } from "../env/agent-settings.ts";
import { applyHeaderPatch } from "../events.ts";
import { convertToChatFormat } from "../messages.ts";
import type { LLMBackend } from "../utils/backend.ts";
import { buildProviderRequestOptions } from "../utils/provider-options.ts";
import { buildStreamingCallbacks } from "../utils/provider-streaming.ts";

export type ProviderTurnState = {};

export function createProviderTurnState(): ProviderTurnState {
	return {};
}

export interface ProviderTurnOutcome {
	status: Extract<RunOutcomeStatus, "cancelled" | "failed">;
	summary?: string;
	source: "runtime";
}

export type ProviderTurnResult =
	| {
			kind: "response";
			response: Awaited<ReturnType<LLMBackend["generate"]>>;
			messages: Message[];
			contextWasCompacted: boolean;
	  }
	| { kind: "finish"; outcome: ProviderTurnOutcome };

/**
 * The subset of RunAgentLoopConfig this module needs. Declared locally
 * (rather than importing RunAgentLoopConfig from core/agent-loop-runner.ts)
 * to avoid a core <-> loop import cycle, since the runner imports this module.
 */
export interface ProviderTurnConfig extends AgentConfig {
	backend: LLMBackend;
	signal?: AbortSignal;
	onContextCompacted?: (messages: Message[]) => void;
}

export interface RequestAssistantTurnInput {
	state: ProviderTurnState;
	messages: Message[];
	config: ProviderTurnConfig;
	settings: AgentSettings;
	registry: { toToolDefinitions(): Record<string, unknown>[] };
	turnId: string;
	iteration: number;
	contextWasCompacted: boolean;
	convertToLlm: (messages: AgentMessage[]) => Message[];
	emit: AgentEventSink;
	isSteeringInterrupt: (signal: AbortSignal | undefined) => boolean;
	steeringInterruptSummary: string;
}

/**
 * Run one provider request for this turn. On success, returns the response
 * and the (unchanged) messages array. On an unrecoverable error —
 * cancellation or a provider error — returns a terminal outcome for the
 * caller to `finish()`.
 */
export async function requestAssistantTurn(
	input: RequestAssistantTurnInput,
): Promise<ProviderTurnResult> {
	const { config } = input;
	const messages = input.messages;
	const contextWasCompacted = input.contextWasCompacted;

	// Provider callbacks are synchronous by contract, while our event sink
	// may persist/forward asynchronously. Keep a per-request chain so SSE
	// deltas cannot be overtaken by message_end/turn_end events.
	const providerEvents: Promise<void>[] = [];
	const queueProviderEvent = (event: AgentEvent): void => {
		// Invoke every sink immediately so live runtime state is current, and
		// retain its settlement so terminal events cannot overtake deltas.
		providerEvents.push(Promise.resolve(input.emit(event)));
	};
	const llmMessages = input.convertToLlm(messages as AgentMessage[]);
	const chatMessages = convertToChatFormat(llmMessages);

	let requestHeaders = config.streamOptions?.headers;
	let requestTimeoutMs =
		config.streamOptions?.timeoutMs ?? config.turnTimeoutMs;
	let requestMaxRetries = config.streamOptions?.maxRetries ?? 3;
	let requestCacheRetention = config.streamOptions?.cacheRetention;
	let requestMetadata = config.streamOptions?.metadata;

	{
		const result = await config.hooks?.beforeProviderRequest?.({
			model: config.model ?? "",
			sessionId: config.hookSessionId ?? "",
			iteration: input.iteration,
			streamOptions: config.streamOptions ?? {},
		});
		if (result?.headers !== undefined) {
			requestHeaders = applyHeaderPatch(requestHeaders, result.headers);
		}
		if (result?.timeoutMs !== undefined) requestTimeoutMs = result.timeoutMs;
		if (result?.maxRetries !== undefined) requestMaxRetries = result.maxRetries;
		if (result?.cacheRetention !== undefined)
			requestCacheRetention = result.cacheRetention;
		if (result?.metadata !== undefined) requestMetadata = result.metadata;
	}

	// Payload hook must receive the backend's transport-ready payload.
	// Building a parallel camelCase payload here used to replace fields
	// such as `stream` and `max_tokens`, silently disabling SSE.

	try {
		// Inference mode params override individual config values. "auto" no
		// longer dynamically selects a preset (tasks/adaptive-mode.ts was
		// dropped) — it now resolves to its own static entry in
		// INFERENCE_MODES like every other mode.
		const modeDef = input.settings.inferenceMode
			? getInferenceMode(input.settings.inferenceMode)
			: undefined;
		const requestOptions = buildProviderRequestOptions({
			chatMessages,
			toolDefinitions: input.registry.toToolDefinitions(),
			settings: input.settings,
			config,
			requestHeaders: requestHeaders as Record<string, string>,
			requestTimeoutMs: requestTimeoutMs as number,
			requestMaxRetries: requestMaxRetries as number,
			requestCacheRetention,
			requestMetadata,
			modeDef,
			signal: config.signal,
			payloadHook: config.hooks?.beforeProviderPayload,
		});
		requestOptions.callbacks = buildStreamingCallbacks(
			input.turnId,
			queueProviderEvent,
		);
		const response = await config.backend.generate(
			chatMessages,
			requestOptions,
		);
		await Promise.all(providerEvents);
		return { kind: "response", response, messages, contextWasCompacted };
	} catch (llmError) {
		// Cancellation wins over provider error classification. Some provider
		// clients replace an AbortSignal cancellation with a generic Error.
		const cancelled =
			config.signal?.aborted ||
			(llmError instanceof Error && llmError.name === "AbortError");
		if (cancelled) {
			const steeringInterrupt = input.isSteeringInterrupt(config.signal);
			if (!steeringInterrupt) {
				await input.emit({ type: "error", message: "Operation aborted" });
			}
			return {
				kind: "finish",
				outcome: {
					status: "cancelled",
					summary: steeringInterrupt
						? input.steeringInterruptSummary
						: "Operation aborted",
					source: "runtime",
				},
			};
		}

		// Mirrors pi's agent-loop.ts: any other provider error ends the turn.
		// Retry/backoff responsibility belongs to the backend's stream
		// implementation, not the loop.
		const message =
			llmError instanceof Error ? llmError.message : String(llmError);
		await input.emit({ type: "error", message, error: llmError });
		return {
			kind: "finish",
			outcome: { status: "failed", summary: message, source: "runtime" },
		};
	}
}
