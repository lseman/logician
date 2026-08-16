/**
 * Provider request options builder.
 *
 * Assembles the GenerateOptions object for LLMBackend.generate() by
 * composing inference mode parameters, agent settings, config overrides,
 * and hook-mutated request metadata (headers, timeout, retries, etc.).
 */

import type { GenerateOptions } from "../backend.ts";
import type { AgentHooks } from "../types/types-messages.ts";
import type { AgentSettings } from "../agent-settings.ts";
import type { AgentConfig, ThinkingLevel } from "../types.ts";
import type { InferenceModeDef } from "../types/types-config.ts";

export interface ProviderOptionsContext {
	chatMessages: Record<string, unknown>[];
	toolDefinitions: Record<string, unknown>[];
	settings: AgentSettings;
	config: AgentConfig;
	requestHeaders: Record<string, string>;
	requestTimeoutMs: number;
	requestMaxRetries: number;
	requestCacheRetention: string | undefined;
	requestMetadata: Record<string, unknown> | undefined;
	modeDef: InferenceModeDef | undefined;
	signal?: AbortSignal;
	payloadHooks: Array<AgentHooks["beforeProviderPayload"] | undefined>;
}

/**
 * Build the GenerateOptions object from the full request context.
 *
 * Handles inference mode parameter resolution: when the mode requests
 * provider defaults (e.g., "none"), all sampling params are omitted so
 * the provider uses its built-in defaults. Otherwise, mode params take
 * precedence over config-level overrides.
 */
export function buildProviderRequestOptions(
	ctx: ProviderOptionsContext,
): GenerateOptions {
	const {
		chatMessages,
		toolDefinitions,
		settings,
		config,
		requestHeaders,
		requestTimeoutMs,
		requestMaxRetries,
		requestCacheRetention,
		requestMetadata,
		modeDef,
		signal,
		payloadHooks,
	} = ctx;

	const useProviderDefaults = modeDef?.useProviderDefaults ?? false;
	const modeParams = useProviderDefaults ? undefined : modeDef?.params;
	const effectiveTemp = modeParams?.temperature ?? config.temperature;

	const options: GenerateOptions = {
		tools: toolDefinitions,
		...(effectiveTemp !== undefined && { temperature: effectiveTemp }),
		maxTokens: config.maxTokens ?? 4096,
		...(modeParams?.top_p !== undefined && { topP: modeParams.top_p }),
		...(modeParams?.top_k !== undefined && { topK: modeParams.top_k }),
		...(modeParams?.min_p !== undefined && { minP: modeParams.min_p }),
		...(modeParams?.presence_penalty !== undefined && {
			presencePenalty: modeParams.presence_penalty,
		}),
		...(modeParams?.repetition_penalty !== undefined && {
			repetitionPenalty: modeParams.repetition_penalty,
		}),
		signal,
		thinkingLevel: settings.thinkingLevel as ThinkingLevel,
		callbacks: undefined, // set by caller via buildStreamingCallbacks
		headers: requestHeaders,
		timeoutMs: requestTimeoutMs,
		maxRetries: requestMaxRetries,
		cacheRetention: requestCacheRetention,
		metadata: requestMetadata,
		transformPayload: async basePayload => {
			let payload = basePayload;
			for (const hook of payloadHooks) {
				const result = await hook?.({
					model: config.model ?? "",
					payload,
				});
				if (result?.payload) payload = result.payload;
			}
			return payload;
		},
	};

	return options;
}
