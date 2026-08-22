/**
 * One provider request, with retry/compaction handling.
 *
 * Builds the request (hooks for headers/timeout/retries/cache/metadata,
 * adaptive inference mode selection, `beforeProviderPayload`), calls
 * `backend.generate()`, and on failure consults the OutputGuard to decide
 * whether to retry, compact-then-retry, or give up. Owns the inner
 * `while (true)` retry loop so the caller only sees a response or a
 * terminal outcome.
 */

import { compactToFit } from "../compaction/engine.ts";
import type { AgentSettings } from "../../control/configuration/agent-settings.ts";
import type { OutputGuard } from "../../control/guards/output-guard.ts";
import type { RunOutcomeStatus } from "../../control/policy/execution-policy.ts";
import type { InterventionInput } from "../../control/policy/intervention-controller.ts";
import type { LLMBackend } from "../../capabilities/provider/backend.ts";
import { convertToChatFormat } from "../../capabilities/provider/messages.ts";
import { getInferenceMode } from "../../system/types/types-config.ts";
import type {
	AgentEvent,
	AgentEventSink,
	AgentMessage,
	CompactableMessage,
	Message,
} from "../../system/types/types-messages.ts";
import { selectAdaptiveMode } from "./adaptive-mode.ts";
import { applyHeaderPatch } from "./callbacks.ts";
import type { AgentLoopConfig } from "./config.ts";
import { buildProviderRequestOptions } from "./provider-options.ts";
import { buildStreamingCallbacks } from "./provider-streaming.ts";

export interface ProviderTurnState {
	/** Last emitted adaptive-mode selection key, so repeats don't re-emit `inference_mode_selected`. */
	lastAdaptiveSelection: string;
}

export function createProviderTurnState(): ProviderTurnState {
	return { lastAdaptiveSelection: "" };
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
export type ProviderTurnConfig = AgentLoopConfig;

export interface RequestAssistantTurnInput {
	state: ProviderTurnState;
	messages: Message[];
	/**
	 * Request-scoped rendering of `messages` (e.g. with transient
	 * transformContext output like memory-retrieval context spliced in). Used
	 * only to build this one outgoing payload — never persisted, never
	 * compacted. Falls back to `messages` when absent.
	 */
	presentationMessages?: Message[];
	config: ProviderTurnConfig;
	settings: AgentSettings;
	registry: { toToolDefinitions(): Record<string, unknown>[] };
	outputGuard: OutputGuard | null | undefined;
	turnId: string;
	iteration: number;
	adaptiveObjective: string;
	performedToolWork: boolean;
	toolFailures: number;
	contextWasCompacted: boolean;
	convertToLlm: (messages: AgentMessage[]) => Message[];
	emit: AgentEventSink;
	intervene: (input: InterventionInput) => Promise<void> | void;
	isSteeringInterrupt: (signal: AbortSignal | undefined) => boolean;
	steeringInterruptSummary: string;
}

/**
 * Run the provider-request retry loop for one turn. On success, returns the
 * response and the (possibly compacted) messages array. On an unrecoverable
 * error — cancellation, guard abort, or a compaction that couldn't shrink
 * the transcript — returns a terminal outcome for the caller to `finish()`.
 */
export async function requestAssistantTurn(
	input: RequestAssistantTurnInput,
): Promise<ProviderTurnResult> {
	const { config, state } = input;
	let messages = input.messages;
	// Only valid for the first attempt — built from the pre-compaction canonical
	// array, so a retry after compact_then_retry must fall back to the (now
	// compacted) canonical `messages` instead of this now-stale rendering.
	let presentationMessages = input.presentationMessages;
	let contextWasCompacted = input.contextWasCompacted;
	let activeRetryAttempt = 0;

	while (true) {
		// Provider callbacks are synchronous by contract, while our event sink
		// may persist/forward asynchronously. Keep a per-request chain so SSE
		// deltas cannot be overtaken by message_end/turn_end events.
		const providerEvents: Promise<void>[] = [];
		const queueProviderEvent = (event: AgentEvent): void => {
			// Invoke every sink immediately so live runtime state is current, and
			// retain its settlement so terminal events cannot overtake deltas.
			providerEvents.push(Promise.resolve(input.emit(event)));
		};
		const llmMessages = input.convertToLlm(
			(presentationMessages ?? messages) as AgentMessage[],
		);
		const chatMessages = convertToChatFormat(llmMessages);

		let requestHeaders = config.streamOptions?.headers;
		let requestTimeoutMs =
			config.streamOptions?.timeoutMs ?? config.turnTimeoutMs;
		let requestMaxRetries =
			config.streamOptions?.maxRetries ?? config.maxRetries ?? 3;
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
			if (result?.maxRetries !== undefined)
				requestMaxRetries = result.maxRetries;
			if (result?.cacheRetention !== undefined)
				requestCacheRetention = result.cacheRetention;
			if (result?.metadata !== undefined) requestMetadata = result.metadata;
		}

		// Payload hook must receive the backend's transport-ready payload.
		// Building a parallel camelCase payload here used to replace fields
		// such as `stream` and `max_tokens`, silently disabling SSE.

		try {
			// Resolve inference mode params — they override individual config values.
			const adaptiveDecision =
				input.settings.inferenceMode === "auto"
					? selectAdaptiveMode({
							objective: input.adaptiveObjective,
							performedToolWork: input.performedToolWork,
							toolFailures: input.toolFailures,
						})
					: undefined;
			const effectiveMode =
				adaptiveDecision?.mode ?? input.settings.inferenceMode;
			if (adaptiveDecision) {
				const selectionKey = `${adaptiveDecision.mode}:${adaptiveDecision.reason}`;
				if (selectionKey !== state.lastAdaptiveSelection) {
					state.lastAdaptiveSelection = selectionKey;
					await input.emit({
						type: "inference_mode_selected",
						configuredMode: "auto",
						effectiveMode: adaptiveDecision.mode,
						reason: adaptiveDecision.reason,
					});
				}
			}
			const modeDef = effectiveMode
				? getInferenceMode(effectiveMode)
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
			if (activeRetryAttempt > 0) {
				await input.emit({
					type: "agent_retry_end",
					attempt: activeRetryAttempt,
					success: true,
				});
			}
			return { kind: "response", response, messages, contextWasCompacted };
		} catch (llmError) {
			// Cancellation wins over provider error classification. Some provider
			// clients replace an AbortSignal cancellation with a generic Error;
			// sending that through OutputGuard would create a fake retry.
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

			const guardResult = input.outputGuard?.handleError(llmError);

			if (!guardResult || guardResult.action === "abort") {
				await input.emit({
					type: "error",
					message: guardResult?.message ?? String(llmError),
					error: llmError,
				});
				if (!cancelled && activeRetryAttempt > 0) {
					await input.emit({
						type: "agent_retry_end",
						attempt: guardResult?.attempt ?? activeRetryAttempt,
						success: false,
					});
				}
				return {
					kind: "finish",
					outcome: {
						status: "failed",
						summary: guardResult?.message ?? String(llmError),
						source: "runtime",
					},
				};
			}

			activeRetryAttempt = guardResult.attempt ?? activeRetryAttempt + 1;
			// Emit retry start event (OutputGuard handles error classification,
			// loop runner emits the event to avoid duplicates).
			await input.emit({
				type: "agent_retry_start",
				attempt: activeRetryAttempt,
				maxRetries: guardResult.maxRetries ?? 3,
				delayMs: undefined,
				error: guardResult.message ?? String(llmError),
			});
			await input.intervene({
				kind: "retry",
				cause: guardResult.action,
				detector: "provider_error_guard",
				message: guardResult.message ?? String(llmError),
				iteration: input.iteration,
				counters: { attempt: activeRetryAttempt },
				limits: { maxRetries: guardResult.maxRetries ?? 3 },
			});

			if (guardResult.action === "compact_then_retry") {
				const compacted = await compactToFit(messages as CompactableMessage[], {
					triggerTokens: 0,
					targetTokens: config.contextWindowTokens
						? Math.floor(config.contextWindowTokens * 0.75)
						: undefined,
				});
				if (!compacted.changed) {
					await input.emit({
						type: "agent_retry_end",
						attempt: activeRetryAttempt,
						success: false,
					});
					await input.emit({
						type: "error",
						message:
							"Context compaction could not reduce the active transcript.",
					});
					return {
						kind: "finish",
						outcome: {
							status: "failed",
							summary:
								"Context compaction could not reduce the active transcript.",
							source: "runtime",
						},
					};
				}
				messages = compacted.messages as unknown as Message[];
				// Stale now — it was rendered from the pre-compaction canonical
				// array. The retry must send the compacted canonical messages.
				presentationMessages = undefined;
				contextWasCompacted = true;
				config.onContextCompacted?.(messages);
				await input.emit({
					type: "context_update",
					tokens: compacted.tokensAfter,
					maxTokens: config.contextWindowTokens,
					compacted: true,
				});
				await input.intervene({
					kind: "compaction",
					cause: "context_full",
					detector: "provider_retry",
					message: `Context compacted from ${compacted.tokensBefore} to ${compacted.tokensAfter} tokens before retrying.`,
					iteration: input.iteration,
					counters: {
						tokensBefore: compacted.tokensBefore,
						tokensAfter: compacted.tokensAfter,
					},
				});
			}
		}
	}
}
