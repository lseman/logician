// ── LLM Backend ──────────────────────────────────────────────────────────────────
// OpenAI-compatible HTTP client for streaming LLM responses.
// Mirrors Python LlamaCppClient/VLLMClient but simplified for TS.

import type { ThinkingLevel, ToolCall } from "./types.ts";

export interface LLMResponse {
	content: string | null;
	toolCalls: ToolCall[];
	stopReason: "stop" | "length" | "error";
	errorMessage?: string;
}

/** Streaming callbacks for a generate() call. All optional. */
export interface GenerateCallbacks {
	onDelta?: (delta: string) => void;
	onThinking?: (delta: string) => void;
	onTextStart?: () => void;
	onTextEnd?: () => void;
	onToolCallStart?: (toolCallId: string, name: string, args: string) => void;
	onToolCallDelta?: (toolCallId: string, delta: string) => void;
}

/** Options for a generate() call. */
export interface GenerateOptions {
	tools?: Record<string, unknown>[];
	temperature?: number;
	maxTokens?: number;
	signal?: AbortSignal;
	thinkingLevel?: ThinkingLevel;
	callbacks?: GenerateCallbacks;
}

export interface LLMBackend {
	generate(
		messages: Record<string, unknown>[],
		options?: GenerateOptions,
	): Promise<LLMResponse>;

	/** Return a backend identical to this one but bound to a different model. */
	withModel(model: string): LLMBackend;

	/** The model this backend currently targets. */
	readonly model: string;
}

export class OpenAIBackend implements LLMBackend {
	readonly baseUrl: string;
	readonly model: string;
	private chatTemplate?: string;
	private stop?: string[];
	private defaultThinkingLevel: ThinkingLevel = "medium";

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
		this.defaultThinkingLevel = options.thinkingLevel ?? "medium";
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
			signal,
			thinkingLevel,
			callbacks = {},
		} = options;
		const {
			onDelta,
			onThinking,
			onTextStart,
			onTextEnd,
			onToolCallStart,
			onToolCallDelta,
		} = callbacks;

		const body: Record<string, unknown> = {
			model: this.model,
			messages,
			temperature,
			max_tokens: maxTokens,
			stream: true,
			...(this.stop && { stop: this.stop }),
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

		const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(body),
			signal,
		});

		if (!response.ok) {
			const errorText = await response.text();
			throw new Error(`LLM request failed: ${response.status} ${errorText}`);
		}

		if (!response.body) {
			throw new Error("Response body is unavailable");
		}

		const reader = response.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		let fullContent = "";
		let toolCalls: ToolCall[] = [];
		let stopReason: LLMResponse["stopReason"] = "stop";
		let hasText = false;
		let hasToolCalls = false;

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

					if (delta.reasoning_content) {
						onThinking?.(delta.reasoning_content);
					}

					if (delta.tool_calls) {
						for (const tc of delta.tool_calls) {
							// Accumulate tool call
							if (!toolCalls[tc.index]) {
								toolCalls[tc.index] = {
									id: "",
									name: "",
									arguments: "",
								};
								// Emit tool_call_start on first chunk of a new tool call
								if (!hasToolCalls || !toolCalls[tc.index].id) {
									hasToolCalls = true;
									const callId = tc.id || `tool_${tc.index}`;
									const callName = tc.function?.name || "";
									const callArgs = tc.function?.arguments || "";
									onToolCallStart?.(callId, callName, callArgs);
								}
							}
							if (tc.id) toolCalls[tc.index].id = tc.id;
							if (tc.function?.name)
								toolCalls[tc.index].name = tc.function.name;
							if (tc.function?.arguments) {
								toolCalls[tc.index].arguments += tc.function.arguments;
								// Emit delta for accumulated arguments
								onToolCallDelta?.(
									tc.id || `tool_${tc.index}`,
									tc.function.arguments,
								);
							}
						}
					}
				} catch {
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
			.filter((tc) => tc?.name)
			.map((tc, index) => ({
				id: tc.id || `tool_${index}`,
				name: tc.name,
				arguments: tc.arguments || "{}",
			}));

		if (toolCalls.length > 0) {
			stopReason = "stop";
		}

		return {
			content: fullContent || null,
			toolCalls,
			stopReason,
		};
	}
}
