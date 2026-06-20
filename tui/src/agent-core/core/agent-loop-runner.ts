// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.

import type { LLMBackend, BackendError } from "./backend.ts";
import {
	convertToChatFormat,
	createAssistantMessage,
	createSystemMessage,
	createToolResultMessage,
	convertToLlm as defaultConvertToLlm,
} from "./messages.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHooks,
	AgentMessage,
	EventHandler,
	Message,
	StopReason,
	Tool,
	ToolCall,
} from "./types.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import { OutputGuard } from "./output-guard.ts";
import { ExtensionEventBus } from "../hooks/extension-event-bus.ts";
import type {
	BeforeAgentStartEvent,
	TurnStartEvent,
	TurnEndEvent,
	MessageStartEvent,
	MessageUpdateEvent,
	MessageEndEvent,
	ToolExecutionStartEvent,
	ToolExecutionEndEvent,
	BeforeProviderRequestEvent,
	AfterProviderResponseEvent,
} from "../hooks/extension-events.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/** Emit a typed extension event if the bus is available. */
async function emitTyped(
	emitter: ExtensionEventBus | undefined,
	event: { type: string; [key: string]: unknown },
): Promise<void> {
	if (!emitter) return;
	await emitter.emit(event as any);
}

export interface RunAgentLoopContext {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
	cwd?: string;
}

export interface RunAgentLoopConfig extends AgentConfig {
	backend: LLMBackend;
	signal?: AbortSignal;
	maxIterations?: number;
	getSteeringMessages?: (ctx: {
		messages: Message[];
		iteration: number;
	}) => Promise<Message[] | undefined> | Message[] | undefined;
	getFollowUpMessages?: (ctx: {
		messages: Message[];
		iteration: number;
		assistantText: string;
		stopReason?: StopReason;
	}) => Promise<Message[] | undefined> | Message[] | undefined;
	prepareNextTurn?: (ctx: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
	}) => Promise<{ messages?: Message[] } | undefined> | { messages?: Message[] } | undefined;
	shouldStopAfterTurn?: (ctx: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
		message?: Message;
		toolResults: Message[];
	}) => Promise<boolean | undefined> | boolean | undefined;
	transformContext?: (ctx: {
		messages: AgentMessage[];
		iteration: number;
		signal?: AbortSignal;
	}) => Promise<{ messages?: AgentMessage[] } | undefined> | { messages?: AgentMessage[] } | undefined;
	// Output guard config (optional). When provided, the loop uses OutputGuard
	// to handle context_full, retryable errors, and empty responses.
	outputGuard?: OutputGuard | null;
	/** Typed extension event bus for structured extension subscriptions. */
	extensionBus?: ExtensionEventBus;
}

const DEFAULT_MAX_ITERATIONS = 30;
const DEFAULT_MAX_CONTINUATIONS = 12;

function withSystemPrompt(context: RunAgentLoopContext): Message[] {
	const nonSystem = context.messages.filter((message) => message.role !== "system");
	return [createSystemMessage(context.systemPrompt ?? "You are a helpful assistant."), ...nonSystem];
}

function assistantText(message: Message | undefined): string {
	return message?.role === "assistant" && typeof message.content === "string"
		? message.content
		: "";
}

async function firstMessages(
	callbacks: Array<(() => Promise<Message[] | undefined> | Message[] | undefined) | undefined>,
): Promise<Message[]> {
	for (const callback of callbacks) {
		const messages = await drainMessages(callback);
		if (messages.length > 0) return messages;
	}
	return [];
}

async function transformMessages(
	callbacks: Array<AgentHooks["transformContext"] | RunAgentLoopConfig["transformContext"] | undefined>,
	ctx: { messages: AgentMessage[]; iteration: number; signal?: AbortSignal },
): Promise<AgentMessage[] | undefined> {
	for (const callback of callbacks) {
		const result = await callback?.(ctx);
		if (result?.messages) return result.messages;
	}
	return undefined;
}

async function prepareMessages(
	callbacks: Array<AgentHooks["prepareNextTurn"] | RunAgentLoopConfig["prepareNextTurn"] | undefined>,
	ctx: { messages: Message[]; iteration: number; hadToolCalls: boolean },
): Promise<Message[] | undefined> {
	for (const callback of callbacks) {
		const result = await callback?.(ctx);
		if (result?.messages) return result.messages;
	}
	return undefined;
}

async function shouldStop(
	callbacks: Array<AgentHooks["shouldStopAfterTurn"] | RunAgentLoopConfig["shouldStopAfterTurn"] | undefined>,
	ctx: {
		messages: Message[];
		iteration: number;
		hadToolCalls: boolean;
		message?: Message;
		toolResults: Message[];
	},
): Promise<boolean> {
	for (const callback of callbacks) {
		const result = await callback?.(ctx);
		if (result === true) return true;
	}
	return false;
}

function stopReasonFor(responseStopReason: "stop" | "length" | "error", toolCalls: ToolCall[]): StopReason {
	if (toolCalls.length > 0) return "tool_calls";
	return responseStopReason === "length" ? "length" : responseStopReason === "error" ? "error" : "stop";
}

async function emitMessagePair(emit: AgentEventSink, turnId: string, message: Message): Promise<void> {
	await emit({ type: "message_start", turnId, role: message.role });
	await emit({ type: "message_end", turnId, message });
}

async function drainMessages(
	drain:
		| (() => Promise<Message[] | undefined> | Message[] | undefined)
		| undefined,
): Promise<Message[]> {
	if (!drain) return [];
	const messages = await drain();
	return messages?.length ? messages : [];
}

export async function runAgentLoop(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	let messages = [...withSystemPrompt(context), ...prompts];
	const newMessages: Message[] = [...prompts];
	const maxIterations = config.maxIterations ?? DEFAULT_MAX_ITERATIONS;
	const maxContinuations = config.maxContinuations ?? DEFAULT_MAX_CONTINUATIONS;
	const registry = new ToolRegistry({
		cwd: context.cwd ?? config.cwd,
		signal: config.signal,
		onQuestionRequest: config.onQuestionRequest,
	});
	registry.registerMany(context.tools ?? config.tools ?? []);

			const outputGuard = config.outputGuard;
	let iteration = 0;
	let continuationCount = 0;
	let pendingMessages = await firstMessages([
		() => config.getSteeringMessages?.({ messages, iteration }),
		() => config.internalHooks?.getSteeringMessages?.({ messages, iteration }),
		() => config.hooks?.getSteeringMessages?.({ messages, iteration }),
	]);

	// Emit typed before_agent_start event
	await emitTyped(config.extensionBus, {
		type: "before_agent_start",
		prompt: prompts.map((p) => p.content).join("\n"),
		systemPrompt: context.systemPrompt ?? "",
	});

	await emit({ type: "agent_start" });
	const promptTurnId = "turn_0";
	for (const prompt of prompts) {
		await emitMessagePair(emit, promptTurnId, prompt);
	}

	while (iteration < maxIterations) {
		if (config.signal?.aborted) {
			await emit({ type: "error", message: "Operation aborted" });
			break;
		}

		let hasMoreToolCalls = true;
		while ((hasMoreToolCalls || pendingMessages.length > 0) && iteration < maxIterations) {
			iteration++;
			const turnId = `turn_${iteration}`;
			await emitTyped(config.extensionBus, { type: "turn_start", turnIndex: iteration });
			await emit({ type: "turn_start", turnId });

			if (pendingMessages.length > 0) {
				for (const pending of pendingMessages) {
					messages.push(pending);
					newMessages.push(pending);
					await emitMessagePair(emit, turnId, pending);
				}
				pendingMessages = [];
			}

			const transformed = await transformMessages(
				[config.transformContext, config.internalHooks?.transformContext, config.hooks?.transformContext],
				{ messages: messages as AgentMessage[], iteration, signal: config.signal },
			);
			if (transformed) messages = transformed as Message[];

			const llmMessages = (config.convertToLlm ?? defaultConvertToLlm)(
				messages as AgentMessage[],
			);
			const chatMessages = convertToChatFormat(llmMessages);

			// Output guard: check if response is empty (proactive, before next iteration)

			// Attempt LLM call with output guard error handling.
			let response: Awaited<ReturnType<LLMBackend["generate"]>>;
			let llmError: unknown = null;

			try {
				response = await config.backend.generate(chatMessages, {
					tools: registry.toToolDefinitions(),
					temperature: config.temperature ?? 0.5,
					maxTokens: config.maxTokens ?? 4096,
					signal: config.signal,
					thinkingLevel: config.thinkingLevel,
					callbacks: {
						onDelta: (delta) => emit({ type: "text_delta", turnId, delta }),
						onThinking: (delta) => emit({ type: "thinking_delta", turnId, delta }),
						onTextStart: () => emit({ type: "text_start", turnId }),
						onTextEnd: () => emit({ type: "text_end", turnId }),
						onToolCallStart: (toolCallId, toolName, args) =>
							emit({ type: "tool_call_start", toolCallId, toolName, args }),
						onToolCallDelta: (toolCallId, delta) =>
							emit({ type: "tool_call_delta", toolCallId, delta }),
					},
				});
			} catch (err) {
				llmError = err;
				response = null as any;
			}

			// Process LLM error through output guard
			if (llmError) {
				const guardResult = outputGuard?.handleError(llmError);
				if (guardResult) {
					// Emit error event
					emit({
						type: "error",
						message: guardResult.message ?? String(llmError),
						error: llmError,
					});

					if (guardResult.action === "abort") {
						emit({ type: "auto_retry_end", attempt: guardResult.attempt ?? 0, success: false });
						await emitTyped(config.extensionBus, { type: "agent_end", messages: newMessages });
			await emit({ type: "agent_end", messages: newMessages });
						return newMessages;
					}

					if (guardResult.action === "retry" || guardResult.action === "compact_then_retry") {
						emit({
							type: "auto_retry_start",
							attempt: guardResult.attempt ?? 1,
							maxRetries: guardResult.maxRetries ?? 3,
							delayMs: guardResult.retryDelayMs ?? 0,
							error: guardResult.message ?? String(llmError),
						});

						if (guardResult.retryDelayMs && guardResult.retryDelayMs > 0) {
							await new Promise((r) => setTimeout(r, guardResult.retryDelayMs));
						}

						emit({ type: "auto_retry_end", attempt: guardResult.attempt ?? 1, success: true });
						emit({ type: "context_update", tokens: 0, maxTokens: config.contextWindowTokens, compacted: true });
						// Continue to next iteration (retry the LLM call)
						continue;
					}

					// Unknown action — fall through to normal turn end
				}
			}

			const toolCalls = response?.toolCalls ?? [];
			const stopReason = stopReasonFor((response?.stopReason as "stop" | "length" | "error") ?? "stop", toolCalls);

			// Output guard: check for empty/degenerate responses
			if (outputGuard) {
				const guardCheck = outputGuard.checkResponse(
					response?.content ?? null,
					toolCalls.length,
				);
				if (guardCheck.action === "abort") {
					emit({ type: "error", message: guardCheck.message ?? "Model returned empty response." });
					await emitTyped(config.extensionBus, { type: "agent_end", messages: newMessages });
			await emit({ type: "agent_end", messages: newMessages });
					return newMessages;
				}
			}

			const assistant = createAssistantMessage(response?.content as string ?? "", toolCalls);
			messages.push(assistant);
			newMessages.push(assistant);
			await emitTyped(config.extensionBus, { type: "message_start", message: assistant });
			await emit({ type: "message_start", turnId, role: "assistant" });
			await emit({ type: "message_update", turnId, message: assistant });
			await emitTyped(config.extensionBus, { type: "message_update", message: assistant });
			await emit({ type: "message_end", turnId, message: assistant });
			await emitTyped(config.extensionBus, { type: "message_end", message: assistant });

			if (stopReason === "error") {
				await emit({ type: "error", message: response.errorMessage ?? "Model request failed" });
				await emit({ type: "turn_end", turnId, stopReason, message: assistant, toolResults: [] });
				await emitTyped(config.extensionBus, { type: "agent_end", messages: newMessages });
			await emit({ type: "agent_end", messages: newMessages });
				return newMessages;
			}

			const toolResults: Message[] = [];
			hasMoreToolCalls = false;
			for (const toolCall of toolCalls) {
				const prepared = registry.prepare(toolCall);
				// Emit typed tool execution start
				await emitTyped(config.extensionBus, {
					type: "tool_execution_start",
					toolCallId: prepared.call.id,
					toolName: prepared.call.name,
					args: prepared.args,
				});
				const before = await (config.internalHooks?.beforeToolCall?.({
					toolCall: prepared.call,
					args: prepared.args,
					iteration,
				}) ?? config.hooks?.beforeToolCall?.({
					toolCall: prepared.call,
					args: prepared.args,
					iteration,
				}));
				let resultText = before?.content;
				let isError = before?.isError === true;
				const args = before?.args ?? prepared.args;
				if (resultText === undefined) {
					if (prepared.error) {
						resultText = prepared.error;
						isError = true;
					} else {
						const result = await registry.execute(prepared.call, { signal: config.signal }, args);
						resultText = result.content;
					}
				}
				const after = await (config.internalHooks?.afterToolCall?.({
					toolCall: prepared.call,
					args,
					result: resultText,
					isError,
					iteration,
				}) ?? config.hooks?.afterToolCall?.({
					toolCall: prepared.call,
					args,
					result: resultText,
					isError,
					iteration,
				}));
				if (after?.content !== undefined) resultText = after.content;
				if (after?.isError !== undefined) isError = after.isError;
				await emit({
					type: "tool_call_end",
					toolName: prepared.call.name,
					toolCallId: prepared.call.id,
					result: resultText,
					isError,
				});
				// Emit typed tool execution end
				await emitTyped(config.extensionBus, {
					type: "tool_execution_end",
					toolCallId: prepared.call.id,
					toolName: prepared.call.name,
					result: resultText,
					isError,
				});
				const toolResult = createToolResultMessage(
					prepared.call.id,
					prepared.call.name,
					resultText,
					isError,
				);
				messages.push(toolResult);
				newMessages.push(toolResult);
				toolResults.push(toolResult);
				await emitMessagePair(emit, turnId, toolResult);
				hasMoreToolCalls = after?.terminate !== true;
			}

			await emitTyped(config.extensionBus, {
				type: "turn_end",
				turnIndex: iteration,
				stopReason,
				message: assistant,
				toolResults,
			});
			await emit({ type: "turn_end", turnId, stopReason, message: assistant, toolResults });

		// Reset output guard after each completed turn
		outputGuard?.reset();

			const prepared = await prepareMessages(
				[config.prepareNextTurn, config.internalHooks?.prepareNextTurn, config.hooks?.prepareNextTurn],
				{
				messages,
				iteration,
				hadToolCalls: toolCalls.length > 0,
				},
			);
			if (prepared) messages = prepared;

			const stop = await shouldStop(
				[config.shouldStopAfterTurn, config.internalHooks?.shouldStopAfterTurn, config.hooks?.shouldStopAfterTurn],
				{
				messages,
				iteration,
				hadToolCalls: toolCalls.length > 0,
				message: assistant,
				toolResults,
				},
			);
			if (stop) {
				await emitTyped(config.extensionBus, { type: "agent_end", messages: newMessages });
			await emit({ type: "agent_end", messages: newMessages });
				return newMessages;
			}

			pendingMessages = await firstMessages([
				() => config.getSteeringMessages?.({ messages, iteration }),
				() => config.internalHooks?.getSteeringMessages?.({ messages, iteration }),
				() => config.hooks?.getSteeringMessages?.({ messages, iteration }),
			]);
		}

		const followUps = await firstMessages([
			() => config.getFollowUpMessages?.({
				messages,
				iteration,
				assistantText: assistantText(newMessages.at(-1)),
				stopReason: "stop",
			}),
			() => config.internalHooks?.getFollowUpMessages?.({
				messages,
				iteration,
				assistantText: assistantText(newMessages.at(-1)),
				maxContinuations,
				stopReason: "stop",
			}),
			() => config.hooks?.getFollowUpMessages?.({
				messages,
				iteration,
				assistantText: assistantText(newMessages.at(-1)),
				maxContinuations,
				stopReason: "stop",
			}),
		]);
		if (followUps.length > 0 && continuationCount < maxContinuations) {
			continuationCount++;
			pendingMessages = followUps;
			continue;
		}
		break;
	}

	if (iteration >= maxIterations) {
		await emit({ type: "max_iterations", iterations: iteration, limit: maxIterations });
	}

	// Final output guard reset when agent ends
	outputGuard?.reset();
	await emit({ type: "agent_end", messages: newMessages });
	return newMessages;
}

export function forwardEvents(handler?: EventHandler): AgentEventSink {
	return (event) => handler?.(event);
}

/**
 * Resume an agent loop from an existing conversation without adding a new user prompt.
 * Used for retries and continuations where the last message is already a user/tool-result.
 * Throws if the context is empty or ends on an assistant message (nothing to continue from).
 */
export async function runAgentLoopContinue(
	context: RunAgentLoopContext,
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}
	const last = context.messages[context.messages.length - 1];
	if (last?.role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}
	// Re-enter the loop with empty prompts — the existing messages already contain the user turn.
	return runAgentLoop(context, [], config, emit);
}
