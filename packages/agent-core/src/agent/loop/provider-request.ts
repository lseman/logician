import { compactToFit } from "../../compaction/index.ts";
import type { LLMBackend, LLMResponse } from "../backend.ts";
import { getInferenceMode } from "../configuration/inference-modes.ts";
import type { OutputGuard } from "../guards/output-guard.ts";
import type { InterventionInput } from "../intervention-controller.ts";
import { applyHeaderPatch } from "../loop/callbacks.ts";
import {
    convertToChatFormat,
    convertToLlm as defaultConvertToLlm,
} from "../messages.ts";
import { selectAdaptiveMode } from "../tasks/adaptive-mode.ts";
import type {
    AgentConfig,
    AgentEvent,
    AgentMessage,
    CompactableMessage,
    Message,
} from "../types.ts";

export type ProviderTurnResult =
    | {
        kind: "response";
        response: LLMResponse;
        messages: Message[];
        contextCompacted: boolean;
        lastAdaptiveSelection: string;
    }
    | {
        kind: "terminal";
        status: "cancelled" | "failed";
        summary: string;
    };

export interface ProviderTurnRequest {
    backend: LLMBackend;
    messages: Message[];
    tools: Record<string, unknown>[];
    config: Pick<
        AgentConfig,
        | "convertToLlm"
        | "hookSessionId"
        | "hooks"
        | "internalHooks"
        | "inferenceMode"
        | "maxRetries"
        | "maxTokens"
        | "model"
        | "streamOptions"
        | "temperature"
        | "thinkingLevel"
        | "turnTimeoutMs"
        | "contextWindowTokens"
    > & { signal?: AbortSignal };
    turnId: string;
    iteration: number;
    outputGuard?: OutputGuard | null;
    objective: string;
    performedToolWork: boolean;
    toolFailures: number;
    lastAdaptiveSelection: string;
    isSteeringInterrupt: (signal: AbortSignal | undefined) => boolean;
    steeringInterruptSummary: string;
    emit: (event: AgentEvent) => Promise<void> | void;
    intervene: (input: InterventionInput) => Promise<void> | void;
    onContextCompacted?: (messages: Message[]) => void;
}

export async function requestProviderTurn(
    request: ProviderTurnRequest,
): Promise<ProviderTurnResult> {
    let messages = request.messages;
    let activeRetryAttempt = 0;
    let contextCompacted = false;
    let lastAdaptiveSelection = request.lastAdaptiveSelection;

    while (true) {
        const providerEvents: Promise<void>[] = [];
        const queueProviderEvent = (event: AgentEvent): void => {
            providerEvents.push(Promise.resolve(request.emit(event)));
        };
        const llmMessages = (request.config.convertToLlm ?? defaultConvertToLlm)(
            messages as AgentMessage[],
        );
        const chatMessages = convertToChatFormat(llmMessages);
        const providerRequestHooks = [
            request.config.internalHooks?.beforeProviderRequest,
            request.config.hooks?.beforeProviderRequest,
        ];
        let requestHeaders = request.config.streamOptions?.headers;
        let requestTimeoutMs =
            request.config.streamOptions?.timeoutMs ?? request.config.turnTimeoutMs;
        let requestMaxRetries =
            request.config.streamOptions?.maxRetries ??
            request.config.maxRetries ??
            3;
        let requestCacheRetention = request.config.streamOptions?.cacheRetention;
        let requestMetadata = request.config.streamOptions?.metadata;

        for (const hook of providerRequestHooks) {
            const result = await hook?.({
                model: request.config.model ?? "",
                sessionId: request.config.hookSessionId ?? "",
                iteration: request.iteration,
                streamOptions: request.config.streamOptions ?? {},
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

        const payloadHooks = [
            request.config.internalHooks?.beforeProviderPayload,
            request.config.hooks?.beforeProviderPayload,
        ];

        try {
            const adaptiveDecision =
                request.config.inferenceMode === "auto"
                    ? selectAdaptiveMode({
                        objective: request.objective,
                        performedToolWork: request.performedToolWork,
                        toolFailures: request.toolFailures,
                    })
                    : undefined;
            const effectiveMode =
                adaptiveDecision?.mode ?? request.config.inferenceMode;
            if (adaptiveDecision) {
                const selectionKey = `${adaptiveDecision.mode}:${adaptiveDecision.reason}`;
                if (selectionKey !== lastAdaptiveSelection) {
                    lastAdaptiveSelection = selectionKey;
                    await request.emit({
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
            const useProviderDefaults = modeDef?.useProviderDefaults ?? false;
            const modeParams = useProviderDefaults ? undefined : modeDef?.params;
            const effectiveTemp =
                modeParams?.temperature ?? request.config.temperature ?? 0.5;
            const response = await request.backend.generate(chatMessages, {
                tools: request.tools,
                ...(!useProviderDefaults && { temperature: effectiveTemp }),
                maxTokens: request.config.maxTokens ?? 4096,
                ...(modeParams?.top_p !== undefined && { topP: modeParams.top_p }),
                ...(modeParams?.top_k !== undefined && { topK: modeParams.top_k }),
                ...(modeParams?.min_p !== undefined && { minP: modeParams.min_p }),
                ...(modeParams?.presence_penalty !== undefined && {
                    presencePenalty: modeParams.presence_penalty,
                }),
                ...(modeParams?.repetition_penalty !== undefined && {
                    repetitionPenalty: modeParams.repetition_penalty,
                }),
                signal: request.config.signal,
                thinkingLevel: request.config.thinkingLevel,
                callbacks: {
                    onDelta: delta =>
                        queueProviderEvent({
                            type: "text_delta",
                            turnId: request.turnId,
                            delta,
                        }),
                    onThinking: delta =>
                        queueProviderEvent({
                            type: "thinking_delta",
                            turnId: request.turnId,
                            delta,
                        }),
                    onTextStart: () =>
                        queueProviderEvent({ type: "text_start", turnId: request.turnId }),
                    onTextEnd: () =>
                        queueProviderEvent({ type: "text_end", turnId: request.turnId }),
                    onToolCallStart: (toolCallId, toolName, args) =>
                        queueProviderEvent({
                            type: "tool_call_start",
                            toolCallId,
                            toolName,
                            args,
                        }),
                    onToolCallDelta: (toolCallId, delta) =>
                        queueProviderEvent({ type: "tool_call_delta", toolCallId, delta }),
                    onToolCallIdUpdate: (previousToolCallId, toolCallId) =>
                        queueProviderEvent({
                            type: "tool_call_id_update",
                            previousToolCallId,
                            toolCallId,
                        }),
                },
                headers: requestHeaders,
                timeoutMs: requestTimeoutMs,
                maxRetries: requestMaxRetries,
                cacheRetention: requestCacheRetention,
                metadata: requestMetadata,
                transformPayload: async basePayload => {
                    let payload = basePayload;
                    for (const hook of payloadHooks) {
                        const result = await hook?.({
                            model: request.config.model ?? "",
                            payload,
                        });
                        if (result?.payload) payload = result.payload;
                    }
                    return payload;
                },
            });
            await Promise.all(providerEvents);
            if (activeRetryAttempt > 0) {
                await request.emit({
                    type: "agent_retry_end",
                    attempt: activeRetryAttempt,
                    success: true,
                });
            }
            return {
                kind: "response",
                response,
                messages,
                contextCompacted,
                lastAdaptiveSelection,
            };
        } catch (error) {
            const cancelled =
                request.config.signal?.aborted ||
                (error instanceof Error && error.name === "AbortError");
            if (cancelled) {
                const steeringInterrupt = request.isSteeringInterrupt(
                    request.config.signal,
                );
                if (!steeringInterrupt) {
                    await request.emit({ type: "error", message: "Operation aborted" });
                }
                return {
                    kind: "terminal",
                    status: "cancelled",
                    summary: steeringInterrupt
                        ? request.steeringInterruptSummary
                        : "Operation aborted",
                };
            }

            const guardResult = request.outputGuard?.handleError(error);
            if (!guardResult || guardResult.action === "abort") {
                await request.emit({
                    type: "error",
                    message: guardResult?.message ?? String(error),
                    error,
                });
                if (activeRetryAttempt > 0) {
                    await request.emit({
                        type: "agent_retry_end",
                        attempt: guardResult?.attempt ?? activeRetryAttempt,
                        success: false,
                    });
                }
                return {
                    kind: "terminal",
                    status: "failed",
                    summary: guardResult?.message ?? String(error),
                };
            }

            activeRetryAttempt = guardResult.attempt ?? activeRetryAttempt + 1;
            await request.emit({
                type: "agent_retry_start",
                attempt: activeRetryAttempt,
                maxRetries: guardResult.maxRetries ?? 3,
                delayMs: undefined,
                error: guardResult.message ?? String(error),
            });
            await request.intervene({
                kind: "retry",
                cause: guardResult.action,
                detector: "provider_error_guard",
                message: guardResult.message ?? String(error),
                iteration: request.iteration,
                counters: { attempt: activeRetryAttempt },
                limits: { maxRetries: guardResult.maxRetries ?? 3 },
            });

            if (guardResult.action !== "compact_then_retry") continue;
            const compacted = await compactToFit(messages as CompactableMessage[], {
                triggerTokens: 0,
                targetTokens: request.config.contextWindowTokens
                    ? Math.floor(request.config.contextWindowTokens * 0.75)
                    : undefined,
            });
            if (!compacted.changed) {
                await request.emit({
                    type: "agent_retry_end",
                    attempt: activeRetryAttempt,
                    success: false,
                });
                await request.emit({
                    type: "error",
                    message: "Context compaction could not reduce the active transcript.",
                });
                return {
                    kind: "terminal",
                    status: "failed",
                    summary: "Context compaction could not reduce the active transcript.",
                };
            }
            messages = compacted.messages as unknown as Message[];
            contextCompacted = true;
            request.onContextCompacted?.(messages);
            await request.emit({
                type: "context_update",
                tokens: compacted.tokensAfter,
                maxTokens: request.config.contextWindowTokens,
                compacted: true,
            });
            await request.intervene({
                kind: "compaction",
                cause: "context_full",
                detector: "provider_retry",
                message: `Context compacted from ${compacted.tokensBefore} to ${compacted.tokensAfter} tokens before retrying.`,
                iteration: request.iteration,
                counters: {
                    tokensBefore: compacted.tokensBefore,
                    tokensAfter: compacted.tokensAfter,
                },
            });
        }
    }
}
