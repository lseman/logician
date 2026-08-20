// ── OpenAI-compatible chat-completions streaming adapter ───────────────────
// Speaks the OpenAI /v1/chat/completions streaming protocol (also served by
// llama.cpp, vLLM, SGLang, and other self-hosted/proxy backends). This is the
// only API our previous agent-core supported, so agent starts here
// too rather than porting pi-ai's full multi-provider surface.

import { classifyHttpError, classifyNetworkError } from "./errors.ts";
import { createAssistantMessageEventStream } from "./event-stream.ts";
import type {
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Context,
	Message,
	Model,
	SimpleStreamOptions,
	ToolCall,
	Usage,
} from "./types.ts";

const EMPTY_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

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

function tokenCount(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0
		? Math.floor(value)
		: 0;
}

/** Normalize OpenAI-compatible (including llama.cpp) usage telemetry into our Usage shape. */
export function parseProviderUsage(raw: unknown, rawTimings?: unknown): Usage {
	const usage = (raw && typeof raw === "object" ? raw : {}) as ProviderUsage;
	const timings = (
		rawTimings && typeof rawTimings === "object" ? rawTimings : {}
	) as ProviderTimings;
	const cacheRead = tokenCount(
		usage.prompt_tokens_details?.cached_tokens ??
			usage.input_tokens_details?.cached_tokens ??
			usage.cache_read_input_tokens ??
			usage.cached_tokens ??
			timings.cache_n,
	);
	const input = tokenCount(usage.prompt_tokens);
	const output = tokenCount(usage.completion_tokens);
	return {
		input,
		output,
		cacheRead,
		cacheWrite: 0,
		totalTokens: tokenCount(usage.total_tokens) || input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * Re-role any system message that is not at the start of the array to `user`.
 *
 * Chat templates behind many OpenAI-compatible servers (SGLang, vLLM, llama.cpp) only accept
 * a system/developer message at position 0 and raise "System message must be at the beginning"
 * for any later occurrence. `user` is accepted by every chat template, so trailing system
 * context is re-roled here at the transport boundary instead of changing the caller's message model.
 */
export function normalizeProviderMessages(
	messages: Record<string, unknown>[],
): Record<string, unknown>[] {
	let sawNonSystem = false;
	return messages.map(message => {
		if (message.role === "system" && sawNonSystem)
			return { ...message, role: "user" };
		if (message.role !== "system") sawNonSystem = true;
		return message;
	});
}

function toProviderMessages(context: Context): Record<string, unknown>[] {
	const result: Record<string, unknown>[] = [];
	if (context.systemPrompt)
		result.push({ role: "system", content: context.systemPrompt });
	for (const message of context.messages) {
		result.push(...toProviderMessage(message));
	}
	return normalizeProviderMessages(result);
}

function toProviderMessage(message: Message): Record<string, unknown>[] {
	if (message.role === "user") {
		return [{ role: "user", content: message.content }];
	}
	if (message.role === "toolResult") {
		const text = message.content
			.filter(
				(c): c is Extract<typeof c, { type: "text" }> => c.type === "text",
			)
			.map(c => c.text)
			.join("\n");
		return [{ role: "tool", tool_call_id: message.toolCallId, content: text }];
	}
	// assistant
	const text = message.content
		.filter((c): c is Extract<typeof c, { type: "text" }> => c.type === "text")
		.map(c => c.text)
		.join("");
	const toolCalls = message.content.filter(
		(c): c is ToolCall => c.type === "toolCall",
	);
	return [
		{
			role: "assistant",
			content: text || null,
			...(toolCalls.length > 0 && {
				tool_calls: toolCalls.map(tc => ({
					id: tc.id,
					type: "function",
					function: { name: tc.name, arguments: JSON.stringify(tc.arguments) },
				})),
			}),
		},
	];
}

function toProviderTools(
	context: Context,
): Record<string, unknown>[] | undefined {
	if (!context.tools || context.tools.length === 0) return undefined;
	return context.tools.map(tool => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			parameters: tool.parameters,
		},
	}));
}

function makePartial(model: Model, timestamp: number): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: { ...EMPTY_USAGE },
		stopReason: "pending",
		timestamp,
	};
}

/** Stream a chat completion from an OpenAI-compatible `/v1/chat/completions` endpoint. */
export async function* streamOpenAiCompletions(
	model: Model,
	context: Context,
	options: SimpleStreamOptions = {},
): AsyncGenerator<AssistantMessageEvent> {
	const {
		signal,
		apiKey,
		fetch: fetchImpl = globalThis.fetch,
		headers: extraHeaders,
		timeoutMs,
		temperature,
		maxTokens,
		samplingParams,
		reasoning,
		onPayload,
	} = options;

	const timestamp = Date.now();
	const partial = makePartial(model, timestamp);
	yield { type: "start", partial };

	const body: Record<string, unknown> = {
		model: model.id,
		messages: toProviderMessages(context),
		temperature: temperature ?? 0.5,
		max_tokens: maxTokens ?? model.maxTokens,
		stream: true,
		stream_options: { include_usage: true },
		...model.samplingParams,
		...samplingParams,
	};
	if (reasoning && reasoning !== ("off" as string))
		body.reasoning = { effort: reasoning };
	const tools = toProviderTools(context);
	if (tools) body.tools = tools;

	const finalBody = onPayload ? ((await onPayload(body)) ?? body) : body;

	const timeoutSignal =
		timeoutMs && timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : undefined;
	const requestSignal =
		signal && timeoutSignal
			? AbortSignal.any([signal, timeoutSignal])
			: (signal ?? timeoutSignal);

	let response: Response;
	try {
		response = await fetchImpl(
			`${model.baseUrl.replace(/\/+$/, "")}/v1/chat/completions`,
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					...(apiKey && { Authorization: `Bearer ${apiKey}` }),
					...model.headers,
					...extraHeaders,
				},
				body: JSON.stringify(finalBody),
				signal: requestSignal,
			},
		);
	} catch (e) {
		const error = e as Error;
		if (error.name === "AbortError") {
			const aborted: AssistantMessage = {
				...partial,
				stopReason: "aborted",
				errorMessage: "Request aborted",
			};
			yield { type: "error", reason: "aborted", error: aborted };
			return;
		}
		const transportError = classifyNetworkError(error);
		const failed: AssistantMessage = {
			...partial,
			stopReason: "error",
			errorMessage: transportError.message,
			errorCategory: transportError.category,
		};
		yield { type: "error", reason: "error", error: failed };
		return;
	}

	if (!response.ok) {
		const errorText = await response.text();
		const transportError = classifyHttpError(
			response.status,
			errorText,
			response.headers.get("retry-after"),
		);
		const failed: AssistantMessage = {
			...partial,
			stopReason: "error",
			errorMessage: transportError.message,
			errorCategory: transportError.category,
		};
		yield { type: "error", reason: "error", error: failed };
		return;
	}

	if (!response.body) {
		const failed: AssistantMessage = {
			...partial,
			stopReason: "error",
			errorMessage: "Response body is unavailable",
		};
		yield { type: "error", reason: "error", error: failed };
		return;
	}

	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	let fullText = "";
	let fullThinking = "";
	const toolCallAcc: { id: string; name: string; arguments: string }[] = [];
	let finishReason: string | undefined;
	let usage: Usage = { ...EMPTY_USAGE };
	let hasText = false;
	let hasThinking = false;
	const startedToolIndexes = new Set<number>();

	function currentContent(): AssistantMessage["content"] {
		const content: AssistantMessage["content"] = [];
		if (fullThinking)
			content.push({ type: "thinking", thinking: fullThinking });
		if (fullText) content.push({ type: "text", text: fullText });
		for (const tc of toolCallAcc) {
			if (tc.name)
				content.push({
					type: "toolCall",
					id: tc.id,
					name: tc.name,
					arguments: safeParseArgs(tc.arguments),
				});
		}
		return content;
	}

	try {
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

				let chunk: Record<string, unknown>;
				try {
					chunk = JSON.parse(data);
				} catch {
					continue; // partial JSON is normal in streaming
				}

				const choice = (
					chunk.choices as Record<string, unknown>[] | undefined
				)?.[0];
				const chunkFinish = choice?.finish_reason as string | undefined;
				if (chunkFinish) finishReason = chunkFinish;
				if (chunk.usage || chunk.timings)
					usage = parseProviderUsage(chunk.usage, chunk.timings);

				const delta = choice?.delta as Record<string, unknown> | undefined;
				if (!delta) continue;

				if (delta.content && !hasText) {
					hasText = true;
					yield {
						type: "text_start",
						contentIndex: 0,
						partial: { ...partial, content: currentContent() },
					};
				}
				if (typeof delta.content === "string" && delta.content) {
					fullText += delta.content;
					yield {
						type: "text_delta",
						contentIndex: 0,
						delta: delta.content,
						partial: { ...partial, content: currentContent() },
					};
				}

				const reasoningDelta =
					(delta.reasoning as string | undefined) ??
					(delta.reasoning_content as string | undefined);
				if (reasoningDelta && !hasThinking) {
					hasThinking = true;
					yield {
						type: "thinking_start",
						contentIndex: 0,
						partial: { ...partial, content: currentContent() },
					};
				}
				if (reasoningDelta) {
					fullThinking += reasoningDelta;
					yield {
						type: "thinking_delta",
						contentIndex: 0,
						delta: reasoningDelta,
						partial: { ...partial, content: currentContent() },
					};
				}

				const deltaToolCalls = delta.tool_calls as
					| Record<string, unknown>[]
					| undefined;
				if (deltaToolCalls) {
					for (const tc of deltaToolCalls) {
						const index = tc.index as number;
						if (!toolCallAcc[index])
							toolCallAcc[index] = { id: "", name: "", arguments: "" };
						const acc = toolCallAcc[index];
						if (tc.id) acc.id = tc.id as string;
						const fn = tc.function as Record<string, unknown> | undefined;
						if (fn?.name) acc.name = fn.name as string;
						if (fn?.arguments) acc.arguments += fn.arguments as string;

						if (acc.name && !startedToolIndexes.has(index)) {
							startedToolIndexes.add(index);
							yield {
								type: "toolcall_start",
								contentIndex: index + 1,
								partial: { ...partial, content: currentContent() },
							};
						}
						if (fn?.arguments && startedToolIndexes.has(index)) {
							yield {
								type: "toolcall_delta",
								contentIndex: index + 1,
								delta: fn.arguments as string,
								partial: { ...partial, content: currentContent() },
							};
						}
					}
				}
			}
		}
	} catch (e) {
		const error = e as Error;
		if (error.name === "AbortError" || requestSignal?.aborted) {
			const aborted: AssistantMessage = {
				...partial,
				content: currentContent(),
				stopReason: "aborted",
				errorMessage: "Request aborted",
				usage,
			};
			yield { type: "error", reason: "aborted", error: aborted };
			return;
		}
		const failed: AssistantMessage = {
			...partial,
			content: currentContent(),
			stopReason: "error",
			errorMessage: error.message,
			usage,
		};
		yield { type: "error", reason: "error", error: failed };
		return;
	}

	if (hasText)
		yield {
			type: "text_end",
			contentIndex: 0,
			content: fullText,
			partial: { ...partial, content: currentContent() },
		};
	if (hasThinking)
		yield {
			type: "thinking_end",
			contentIndex: 0,
			content: fullThinking,
			partial: { ...partial, content: currentContent() },
		};
	for (let index = 0; index < toolCallAcc.length; index++) {
		const tc = toolCallAcc[index];
		if (!tc?.name) continue;
		const toolCall: ToolCall = {
			type: "toolCall",
			id: tc.id || `tool_${index}`,
			name: tc.name,
			arguments: safeParseArgs(tc.arguments),
		};
		yield {
			type: "toolcall_end",
			contentIndex: index + 1,
			toolCall,
			partial: { ...partial, content: currentContent() },
		};
	}

	const hasToolCalls = toolCallAcc.some(tc => tc?.name);
	const stopReason =
		finishReason === "length" ? "length" : hasToolCalls ? "toolUse" : "stop";
	const final: AssistantMessage = {
		...partial,
		content: currentContent(),
		stopReason,
		usage,
	};
	yield { type: "done", reason: stopReason, message: final };
}

function safeParseArgs(raw: string): Record<string, unknown> {
	try {
		return raw ? JSON.parse(raw) : {};
	} catch {
		return {};
	}
}

/** StreamFn-compatible entry point: pumps the async generator into an AssistantMessageEventStream. */
export function streamSimple(
	model: Model,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	void (async () => {
		try {
			for await (const event of streamOpenAiCompletions(
				model,
				context,
				options,
			)) {
				stream.push(event);
			}
		} finally {
			stream.end();
		}
	})();
	return stream;
}
