import type {
	ThinkingFormat,
	ThinkingLevel,
} from "../../system/types/types-config.ts";

export interface ProviderRequestContext {
	model: string;
	messages: Record<string, unknown>[];
	tools?: Record<string, unknown>[];
	temperature: number;
	maxTokens: number;
	topP?: number;
	topK?: number;
	minP?: number;
	presencePenalty?: number;
	repetitionPenalty?: number;
	stop?: string[];
	thinkingLevel: ThinkingLevel;
	thinkingFormat?: ThinkingFormat;
}

/**
 * Provider-specific request semantics behind one transport seam. Response
 * decoding is identified explicitly so incompatible protocols cannot be
 * accidentally fed through the chat-completions parser.
 */
export interface ProviderAdapter {
	readonly id: string;
	readonly streamProtocol: "openai-chat-sse";
	endpoint(baseUrl: string): string;
	buildPayload(context: ProviderRequestContext): Record<string, unknown>;
}

export class OpenAIChatCompletionsAdapter implements ProviderAdapter {
	readonly id = "openai-chat-completions";
	readonly streamProtocol = "openai-chat-sse" as const;

	endpoint(baseUrl: string): string {
		return `${baseUrl}/v1/chat/completions`;
	}

	buildPayload(context: ProviderRequestContext): Record<string, unknown> {
		const body: Record<string, unknown> = {
			model: context.model,
			messages: context.messages,
			temperature: context.temperature,
			max_tokens: context.maxTokens,
			stream: true,
			stream_options: { include_usage: true },
			cache_prompt: true,
			...(context.stop && { stop: context.stop }),
			...(context.topP !== undefined && { top_p: context.topP }),
			...(context.topK !== undefined && { top_k: context.topK }),
			...(context.minP !== undefined && { min_p: context.minP }),
			...(context.presencePenalty !== undefined && {
				presence_penalty: context.presencePenalty,
			}),
			...(context.repetitionPenalty !== undefined && {
				repetition_penalty: context.repetitionPenalty,
			}),
			...(context.tools?.length && { tools: context.tools }),
		};

		if (context.thinkingFormat === "qwen") {
			body.enable_thinking = context.thinkingLevel !== "off";
			if (context.thinkingLevel !== "off")
				body.reasoning_effort = context.thinkingLevel;
		} else if (context.thinkingFormat === "qwen-chat-template") {
			body.chat_template_kwargs = {
				enable_thinking: context.thinkingLevel !== "off",
				preserve_thinking: true,
			};
		} else if (context.thinkingLevel !== "off") {
			body.reasoning_effort = context.thinkingLevel;
		}

		return body;
	}
}
