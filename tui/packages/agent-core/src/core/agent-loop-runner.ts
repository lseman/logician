// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.

import type { LLMBackend } from "./backend.ts";
import {
	convertToChatFormat,
	createAssistantMessage,
	createSystemMessage,
	convertToLlm as defaultConvertToLlm,
	compactMessagesForContext,
	estimateChatPayloadTokens,
} from "./messages.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentHooks,
	AgentMessage,
	Message,
	StopReason,
	Tool,
	ToolCall,
} from "./types.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import { ToolResultCache } from "./tool-cache.ts";
import {
	getTaskStatus,
	resetTaskStatus,
} from "./task-status-state.ts";
import type { OutputGuard } from "./output-guard.ts";
import type { ExtensionEventBus } from "../hooks/extensions/event-bus.ts";
import {
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	formatAcceptancePrompt,
	parseAcceptanceReport,
	type AcceptanceConfig,
	type ResolvedAcceptance,
	type AcceptanceReport,
} from "./acceptance-contract.ts";
import {
	emitConclusion,
	lastAssistantContent,
	lastHadToolCalls,
	looksComplete,
	looksNonCommittal,
} from "../runtime/conclusion-policy.ts";
import { executeToolBatch } from "../runtime/tool-batch-controller.ts";

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

export interface ReflectionConfig {
	/** Whether to run a self-evaluation step before final conclusion. */
	enabled?: boolean;
	/** Maximum reflection turns allowed. */
	maxReflections?: number;
	/** Reflection prompt template. $task is replaced with the original task description. */
	prompt?: string;
}

export interface RunAgentLoopConfig extends AgentConfig {
	backend: LLMBackend;
	signal?: AbortSignal;
	maxIterations?: number;
	/** Called when in-loop context-full recovery replaces the active transcript. */
	onContextCompacted?: (messages: Message[]) => void;
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
	}) =>
		| Promise<{ messages?: Message[] } | undefined>
		| { messages?: Message[] }
		| undefined;
	/** Refresh mutable harness configuration after a completed turn. */
	refreshNextTurnConfig?: () => Promise<AgentConfig> | AgentConfig;
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
	}) =>
		| Promise<{ messages?: AgentMessage[] } | undefined>
		| { messages?: AgentMessage[] }
		| undefined;
	// Output guard config (optional). When provided, the loop uses OutputGuard
	// to handle context_full, retryable errors, and empty responses.
	outputGuard?: OutputGuard | null;
	/** Typed extension event bus for structured extension subscriptions. */
	extensionBus?: ExtensionEventBus;
	/** Resolve the acceptance config at call time (allows harness to update it). */
	getAcceptanceConfig?: () => AcceptanceConfig | undefined;
}

const DEFAULT_MAX_ITERATIONS = 30;

function withSystemPrompt(context: RunAgentLoopContext): Message[] {
	const nonSystem = context.messages.filter(
		(message) => message.role !== "system",
	);
	return [
		createSystemMessage(context.systemPrompt ?? "You are a helpful assistant."),
		...nonSystem,
	];
}

function assistantText(message: Message | undefined): string {
	return message?.role === "assistant" && typeof message.content === "string"
		? message.content
		: "";
}

async function firstMessages(
	callbacks: Array<
		(() => Promise<Message[] | undefined> | Message[] | undefined) | undefined
	>,
): Promise<Message[]> {
	for (const callback of callbacks) {
		const messages = await drainMessages(callback);
		if (messages.length > 0) return messages;
	}
	return [];
}

async function transformMessages(
	callbacks: Array<
		| AgentHooks["transformContext"]
		| RunAgentLoopConfig["transformContext"]
		| undefined
	>,
	ctx: { messages: AgentMessage[]; iteration: number; signal?: AbortSignal },
): Promise<AgentMessage[] | undefined> {
	for (const callback of callbacks) {
		const result = await callback?.(ctx);
		if (result?.messages) return result.messages;
	}
	return undefined;
}

async function prepareMessages(
	callbacks: Array<
		| AgentHooks["prepareNextTurn"]
		| RunAgentLoopConfig["prepareNextTurn"]
		| undefined
	>,
	ctx: { messages: Message[]; iteration: number; hadToolCalls: boolean },
): Promise<Message[] | undefined> {
	for (const callback of callbacks) {
		const result = await callback?.(ctx);
		if (result?.messages) return result.messages;
	}
	return undefined;
}

async function shouldStop(
	callbacks: Array<
		| AgentHooks["shouldStopAfterTurn"]
		| RunAgentLoopConfig["shouldStopAfterTurn"]
		| undefined
	>,
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

function stopReasonFor(
	responseStopReason: "stop" | "length" | "error",
	toolCalls: ToolCall[],
): StopReason {
	if (toolCalls.length > 0) return "tool_calls";
	if (responseStopReason === "length") return "length";
	if (responseStopReason === "error") return "error";
	return "stop";
}

async function emitMessagePair(
	emit: AgentEventSink,
	turnId: string,
	message: Message,
): Promise<void> {
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

function waitForRetryDelay(ms: number, signal?: AbortSignal): Promise<boolean> {
	if (signal?.aborted) return Promise.resolve(false);
	return new Promise((resolve) => {
		const timer = setTimeout(() => {
			signal?.removeEventListener("abort", onAbort);
			resolve(true);
		}, ms);
		const onAbort = () => {
			clearTimeout(timer);
			resolve(false);
		};
		signal?.addEventListener("abort", onAbort, { once: true });
	});
}

function applyHeaderPatch(
	current: Record<string, string> | undefined,
	patch: Record<string, string | undefined>,
): Record<string, string> {
	const next = { ...(current ?? {}) };
	for (const [name, value] of Object.entries(patch)) {
		if (value === undefined) delete next[name];
		else next[name] = value;
	}
	return next;
}

// ── Reflection / Self-Evaluation ───────────────────────────────────────────

/** Default reflection prompt asking the model to self-critique. */
const DEFAULT_REFLECTION_PROMPT = `
You have just completed a task. Before finalizing, perform a structured self-evaluation.

Review your work against these criteria:
1. **Completeness**: Did you fully address the task? Are there any loose ends?
2. **Correctness**: Are your changes/code/logic sound? Any bugs or mistakes?
3. **Edge cases**: Did you consider error handling, edge cases, or failure modes?
4. **Quality**: Is the output clean, well-structured, and production-ready?
5. **Next steps**: If incomplete, what specific steps are needed?

Respond with a JSON report in a reflection-report fence:
\
\
\
reflection-report
{
  "assessment": "complete" | "incomplete",
  "reasoning": "Brief explanation of your assessment",
  "issues": ["List any issues found, or empty array if none"],
  "needsMoreWork": boolean,
  "suggestedSteps": ["Steps needed if incomplete, or empty array"]
}
\
\
\
If "assessment" is "complete" and "needsMoreWork" is false, the task is done.
If "assessment" is "incomplete" or "needsMoreWork" is true, you will be asked to continue.`;

interface ReflectionResult {
	assessment: "complete" | "incomplete";
	reasoning: string;
	issues: string[];
	needsMoreWork: boolean;
	suggestedSteps: string[];
}

/** Run a self-evaluation step on the current conversation. */
async function runReflection(
	currentMessages: Message[],
	backend: LLMBackend,
	reflectionConfig: ReflectionConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
): Promise<{
	result: ReflectionResult;
	turnId: string;
	messages: Message[];
}> {
	const prompt = reflectionConfig.prompt ?? DEFAULT_REFLECTION_PROMPT;
	const turnId = "reflection";

	const reflectionPrompt: Message = {
		role: "user",
		content: prompt,
	};

	// Add reflection prompt to messages for the LLM call
	const llmMessages = [...currentMessages, reflectionPrompt];

	await emit({ type: "reflection_start", turnId });

	// Generate reflection response
	const response = await backend.generate(
		convertToChatFormat(llmMessages) as unknown as Record<string, unknown>[],
		{
			tools: [],
			temperature: 0.1, // Low temp for structured reasoning
			maxTokens: 2048,
			signal,
		},
	);

	const reflectionContent = (response?.content as string) ?? "";

	// Parse reflection report
	let reflectionReport: ReflectionResult | undefined;
	const fenceStart = reflectionContent.indexOf("```");
	const fenceEnd = reflectionContent.indexOf("```", fenceStart + 3);
	if (fenceStart >= 0 && fenceEnd > fenceStart) {
		const jsonStr = reflectionContent
			.slice(fenceStart + 3, fenceEnd)
			.trim()
			.replace(/^reflection-report\s*/, "");
		try {
			reflectionReport = JSON.parse(jsonStr) as ReflectionResult;
		} catch {
			// Fallback: parse as free text
			const needsWork =
				/needsMoreWork|incomplete|not done|continue|more work/i.test(
					reflectionContent,
				);
			reflectionReport = {
				assessment: needsWork ? "incomplete" : "complete",
				reasoning: reflectionContent.slice(0, 200),
				issues: [],
				needsMoreWork: needsWork,
				suggestedSteps: [],
			};
		}
	}

	const finalReport: ReflectionResult = reflectionReport ?? {
		assessment: "complete",
		reasoning: "No structured reflection produced; assuming complete.",
		issues: [],
		needsMoreWork: false,
		suggestedSteps: [],
	};

	// Add the reflection turn to messages
	const assistantMsg: Message = {
		role: "assistant",
		content: reflectionContent,
		timestamp: Date.now(),
	};

	await emit({
		type: "reflection_end",
		turnId,
		assessment: finalReport.assessment,
		needsMoreWork: finalReport.needsMoreWork,
		issues: finalReport.issues,
	});

	return {
		result: finalReport,
		turnId,
		messages: [...currentMessages, reflectionPrompt, assistantMsg],
	};
}

export async function runAgentLoop(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	const downstreamEmit = emit;
	let eventSequence = 0;
	emit = (event) =>
		downstreamEmit({
			...event,
			seq: ++eventSequence,
			ts: Date.now(),
		});
	let messages = [...withSystemPrompt(context), ...prompts];
	const newMessages: Message[] = [...prompts];
	resetTaskStatus();
	const finish = async (outcome: {
		status: "completed" | "needs_input" | "blocked" | "failed" | "cancelled";
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	}): Promise<Message[]> => {
		await emit({ type: "run_outcome", ...outcome });
		await emitTyped(config.extensionBus, {
			type: "agent_end",
			messages: newMessages,
			outcome,
		});
		await emit({ type: "agent_end", messages: newMessages });
		return newMessages;
	};
	const maxIterations = config.maxIterations ?? DEFAULT_MAX_ITERATIONS;
	// ── P0-1: Shared tool result cache ─────────────────────────────────
	const cache = new ToolResultCache(2000, 60_000);
	const createRegistry = (tools: Tool[]): ToolRegistry => {
		const next = new ToolRegistry({
			cwd: context.cwd ?? config.cwd,
			signal: config.signal,
			onQuestionRequest: config.onQuestionRequest,
			cache,
		});
		next.registerMany(tools);
		return next;
	};
	let registry = createRegistry(context.tools ?? config.tools ?? []);

	const outputGuard = config.outputGuard;
	let iteration = 0;
	let consecutiveRunnerNudges = 0;
	let contextWasCompacted = false;

	// ── Acceptance contract tracking ─────────────────────────────────────
	let resolvedAcceptance: ResolvedAcceptance | null = null;
	let acceptanceReported = false;
	let acceptanceFailed = false;

	function resolveAcceptance(): ResolvedAcceptance {
		if (!resolvedAcceptance) {
			const raw = config.getAcceptanceConfig?.() ?? config.acceptance;
			resolvedAcceptance = resolveEffectiveAcceptance({ explicit: raw });
		}
		return resolvedAcceptance;
	}

	function checkStopRules(resolved: ResolvedAcceptance): boolean {
		if (!resolved.stopRules?.length) return false;
		const text = lastAssistantContent(newMessages);
		for (const rule of resolved.stopRules) {
			if (text.includes(rule)) return true;
		}
		return false;
	}

	function runAcceptanceVerification(
		resolved: ResolvedAcceptance,
		signal?: AbortSignal,
	): Promise<
		Array<{ command: string; result: "passed" | "failed"; summary?: string }>
	> {
		if (!resolved.verify?.length) return Promise.resolve([]);
		const { execFile } = require("node:child_process");
		const { promisify } = require("node:util");
		const execFileAsync = promisify(execFile);

		return Promise.all(
			resolved.verify.map(
				(v) =>
					new Promise<
						Array<{
							command: string;
							result: "passed" | "failed";
							summary?: string;
						}>
					>((resolve) => {
						const timeout = v.timeoutMs ?? 30_000;
						const timeoutId = setTimeout(() => {
							resolve([
								{
									command: v.command,
									result: "failed",
									summary: `Timeout after ${timeout}ms`,
								},
							]);
						}, timeout);
						if (signal?.aborted) {
							clearTimeout(timeoutId);
							resolve([
								{ command: v.command, result: "failed", summary: "Aborted" },
							]);
							return;
						}

						execFileAsync("bash", ["-c", v.command], {
							cwd: v.cwd ?? config.cwd,
							timeout,
							maxBuffer: 1024 * 1024,
						}).then(
							(output: { stdout?: string; stderr?: string }) => {
								clearTimeout(timeoutId);
								const summary = (output.stdout ?? "").trim().slice(0, 500);
								resolve([{ command: v.command, result: "passed", summary }]);
							},
							(error: NodeJS.ErrnoException) => {
								clearTimeout(timeoutId);
								if (v.allowFailure) {
									resolve([
										{
											command: v.command,
											result: "passed",
											summary: `Non-zero exit ${error.code ?? "unknown"} (allowed)`,
										},
									]);
								} else {
									resolve([
										{
											command: v.command,
											result: "failed",
											summary: error.message.slice(0, 500),
										},
									]);
								}
							},
						);
					}),
			),
		).then((results) => results.flat());
	}

	let pendingMessages = await firstMessages([
		() => config.getSteeringMessages?.({ messages, iteration }),
		() => config.internalHooks?.getSteeringMessages?.({ messages, iteration }),
		() => config.hooks?.getSteeringMessages?.({ messages, iteration }),
	]);

	// Apply beforeAgentStart hook
	const beforeAgentStartHooks = [
		config.internalHooks?.beforeAgentStart,
		config.hooks?.beforeAgentStart,
	];
	let beforeAgentStartResult:
		| { messages?: AgentMessage[]; systemPrompt?: string }
		| undefined;
	for (const hook of beforeAgentStartHooks) {
		const result = await hook?.({
			prompt: prompts.map((p) => p.content).join("\n"),
			systemPrompt: context.systemPrompt ?? "",
			messages: messages as AgentMessage[],
		});
		if (result?.messages)
			beforeAgentStartResult = {
				...beforeAgentStartResult,
				messages: result.messages,
			};
		if (result?.systemPrompt)
			beforeAgentStartResult = {
				...beforeAgentStartResult,
				systemPrompt: result.systemPrompt,
			};
	}

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

	// Apply beforeAgentStart hook results to messages and system prompt
	if (beforeAgentStartResult?.messages) {
		for (const msg of beforeAgentStartResult.messages) {
			messages.push(msg as Message);
			newMessages.push(msg as Message);
		}
	}
	if (beforeAgentStartResult?.systemPrompt) {
		context.systemPrompt = beforeAgentStartResult.systemPrompt;
	}

	// ── Inject acceptance contract into system prompt ──────────────────
	const resolved = resolveAcceptance();
	if (shouldRunAcceptanceFinalization(resolved)) {
		const accPrompt = formatAcceptancePrompt(resolved);
		if (accPrompt) {
			const existingSystem = messages
				.filter((m) => m.role === "system")
				.map((m) => m.content)
				.join("\n\n");
			messages = [
				{
					role: "system" as const,
					content: existingSystem
						? `${existingSystem}\n\n${accPrompt}`
						: accPrompt,
					timestamp: Date.now(),
				},
				...messages.filter((m) => m.role !== "system"),
			];
		}
	}

	while (iteration < maxIterations) {
		if (config.signal?.aborted) {
			await emit({ type: "error", message: "Operation aborted" });
			return finish({
				status: "cancelled",
				summary: "Operation aborted before the provider request.",
				source: "runtime",
			});
		}

		let hasMoreToolCalls = true;
		while (
			(hasMoreToolCalls || pendingMessages.length > 0) &&
			iteration < maxIterations
		) {
			iteration++;
			const turnId = `turn_${iteration}`;
			await emitTyped(config.extensionBus, {
				type: "turn_start",
				turnIndex: iteration,
			});
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
				[
					config.transformContext,
					config.internalHooks?.transformContext,
					config.hooks?.transformContext,
				],
				{
					messages: messages as AgentMessage[],
					iteration,
					signal: config.signal,
				},
			);
			if (transformed) {
				messages = transformed as Message[];
				if (contextWasCompacted) config.onContextCompacted?.(messages);
			}

			let response: Awaited<ReturnType<LLMBackend["generate"]>>;
			let activeRetryAttempt = 0;
			while (true) {
				// Provider callbacks are synchronous by contract, while our event sink
				// may persist/forward asynchronously. Keep a per-request chain so SSE
				// deltas cannot be overtaken by message_end/turn_end events.
				const providerEvents: Promise<void>[] = [];
				const queueProviderEvent = (event: AgentEvent): void => {
					// Invoke every sink immediately so live runtime state is current, and
					// retain its settlement so terminal events cannot overtake deltas.
					providerEvents.push(Promise.resolve(emit(event)));
				};
				const llmMessages = (config.convertToLlm ?? defaultConvertToLlm)(
					messages as AgentMessage[],
				);
				const chatMessages = convertToChatFormat(llmMessages);

				// Apply beforeProviderRequest hook
				const providerRequestHooks = [
					config.internalHooks?.beforeProviderRequest,
					config.hooks?.beforeProviderRequest,
				];
				let requestHeaders = config.streamOptions?.headers;
				let requestTimeoutMs = config.streamOptions?.timeoutMs;
				let requestMaxRetries =
					config.streamOptions?.maxRetries ?? config.maxRetries ?? 3;
				let requestCacheRetention = config.streamOptions?.cacheRetention;
				let requestMetadata = config.streamOptions?.metadata;

				for (const hook of providerRequestHooks) {
					const result = await hook?.({
						model: config.model ?? "",
						sessionId: config.hookSessionId ?? "",
						iteration,
						streamOptions: config.streamOptions ?? {},
					});
					if (result?.headers !== undefined) {
						requestHeaders = applyHeaderPatch(requestHeaders, result.headers);
					}
					if (result?.timeoutMs !== undefined)
						requestTimeoutMs = result.timeoutMs;
					if (result?.maxRetries !== undefined)
						requestMaxRetries = result.maxRetries;
					if (result?.cacheRetention !== undefined)
						requestCacheRetention = result.cacheRetention;
					if (result?.metadata !== undefined) requestMetadata = result.metadata;
				}

				// Payload hooks must receive the backend's transport-ready payload.
				// Building a parallel camelCase payload here used to replace fields
				// such as `stream` and `max_tokens`, silently disabling SSE.
				const payloadHooks = [
					config.internalHooks?.beforeProviderPayload,
					config.hooks?.beforeProviderPayload,
				];

				try {
					response = await config.backend.generate(chatMessages, {
						tools: registry.toToolDefinitions(),
						temperature: config.temperature ?? 0.5,
						maxTokens: config.maxTokens ?? 4096,
						signal: config.signal,
						thinkingLevel: config.thinkingLevel,
						callbacks: {
							onDelta: (delta) =>
								queueProviderEvent({ type: "text_delta", turnId, delta }),
							onThinking: (delta) =>
								queueProviderEvent({ type: "thinking_delta", turnId, delta }),
							onTextStart: () =>
								queueProviderEvent({ type: "text_start", turnId }),
							onTextEnd: () =>
								queueProviderEvent({ type: "text_end", turnId }),
							onToolCallStart: (toolCallId, toolName, args) =>
								queueProviderEvent({
									type: "tool_call_start",
									toolCallId,
									toolName,
									args,
								}),
							onToolCallDelta: (toolCallId, delta) =>
								queueProviderEvent({ type: "tool_call_delta", toolCallId, delta }),
						},
						headers: requestHeaders,
						timeoutMs: requestTimeoutMs,
						maxRetries: requestMaxRetries,
						cacheRetention: requestCacheRetention,
						metadata: requestMetadata,
						transformPayload: async (basePayload) => {
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
					});
					await Promise.all(providerEvents);
					if (activeRetryAttempt > 0) {
						await emit({
							type: "auto_retry_end",
							attempt: activeRetryAttempt,
							success: true,
						});
					}
					break;
				} catch (llmError) {
					const guardResult = outputGuard?.handleError(llmError);

					if (!guardResult || guardResult.action === "abort") {
						await emit({
							type: "error",
							message: guardResult?.message ?? String(llmError),
							error: llmError,
						});
						await emit({
							type: "auto_retry_end",
							attempt: guardResult?.attempt ?? activeRetryAttempt,
							success: false,
						});
						return finish({
							status: config.signal?.aborted ? "cancelled" : "failed",
							summary: guardResult?.message ?? String(llmError),
							source: "runtime",
						});
					}

					activeRetryAttempt = guardResult.attempt ?? activeRetryAttempt + 1;
					await emit({
						type: "auto_retry_start",
						attempt: activeRetryAttempt,
						maxRetries: guardResult.maxRetries ?? 3,
						delayMs: guardResult.retryDelayMs ?? 0,
						error: guardResult.message ?? String(llmError),
					});

					if (guardResult.action === "compact_then_retry") {
						const compacted = compactMessagesForContext(messages, {
							targetTokens: config.contextWindowTokens
								? Math.floor(config.contextWindowTokens * 0.75)
								: undefined,
						});
						if (!compacted.changed) {
							await emit({
								type: "auto_retry_end",
								attempt: activeRetryAttempt,
								success: false,
							});
							await emit({
								type: "error",
								message:
									"Context compaction could not reduce the active transcript.",
							});
							return finish({
								status: "failed",
								summary:
									"Context compaction could not reduce the active transcript.",
								source: "runtime",
							});
						}
						messages = compacted.messages;
						contextWasCompacted = true;
						config.onContextCompacted?.(messages);
						await emit({
							type: "context_update",
							tokens: compacted.tokensAfter,
							maxTokens: config.contextWindowTokens,
							compacted: true,
						});
					}

					if (guardResult.retryDelayMs && guardResult.retryDelayMs > 0) {
						const completed = await waitForRetryDelay(
							guardResult.retryDelayMs,
							config.signal,
						);
						if (!completed) {
							await emit({ type: "error", message: "Operation aborted" });
							await emit({
								type: "auto_retry_end",
								attempt: activeRetryAttempt,
								success: false,
							});
							return finish({
								status: "cancelled",
								summary: "Operation aborted during provider retry.",
								source: "runtime",
							});
						}
					}
				}
			}

			const toolCalls = response?.toolCalls ?? [];
			const rawStopReason =
				(response?.stopReason as "stop" | "length" | "error") ?? "stop";
			const stopReason = stopReasonFor(rawStopReason, toolCalls);

			// Apply afterProviderResponse hook
			const responseHooks = [
				config.internalHooks?.afterProviderResponse,
				config.hooks?.afterProviderResponse,
			];
			for (const hook of responseHooks) {
				await hook?.({
					model: config.model ?? "",
					content: response?.content ?? "",
					toolCallCount: toolCalls.length,
					stopReason,
					usageTokens: response?.usage?.totalTokens,
					iteration,
				});
			}

			// Output guard: check for empty/degenerate responses
			if (outputGuard) {
				const guardCheck = outputGuard.checkResponse(
					response?.content ?? null,
					toolCalls.length,
				);
				if (guardCheck.action === "abort") {
					await emit({
						type: "error",
						message: guardCheck.message ?? "Model returned empty response.",
					});
					return finish({
						status: "failed",
						summary: guardCheck.message ?? "Model returned empty response.",
						source: "runtime",
					});
				}
			}

			const assistant = createAssistantMessage(
				(response?.content as string) ?? "",
				toolCalls,
			);
			messages.push(assistant);
			newMessages.push(assistant);
			await emitTyped(config.extensionBus, {
				type: "message_start",
				message: assistant,
			});
			await emit({ type: "message_start", turnId, role: "assistant" });
			await emit({ type: "message_update", turnId, message: assistant });
			await emitTyped(config.extensionBus, {
				type: "message_update",
				message: assistant,
			});
			await emit({ type: "message_end", turnId, message: assistant });
			await emitTyped(config.extensionBus, {
				type: "message_end",
				message: assistant,
			});

			if (stopReason === "error") {
				await emit({
					type: "error",
					message: response.errorMessage ?? "Model request failed",
				});
				await emit({
					type: "turn_end",
					turnId,
					stopReason,
					message: assistant,
					toolResults: [],
				});
				return finish({
					status: "failed",
					summary: response.errorMessage ?? "Model request failed",
					source: "runtime",
				});
			}

			hasMoreToolCalls = false;
			const batch = await executeToolBatch({
				registry,
				toolCalls,
				rawStopReason,
				toolExecution: config.toolExecution,
				iteration,
				signal: config.signal,
				internalHooks: config.internalHooks,
				hooks: config.hooks,
				emit,
				emitExtension: (event) => emitTyped(config.extensionBus, event),
			});
			const toolResults = batch.messages;
			const toolTerminated = batch.terminated;
			for (const toolResult of toolResults) {
				messages.push(toolResult);
				newMessages.push(toolResult);
				await emitMessagePair(emit, turnId, toolResult);
				hasMoreToolCalls = true;
			}

			// The final usage-only SSE chunk is optional and many local providers
			// omit it. Estimate the serialized conversation as a reliable fallback
			// so context usage never remains stuck at zero.
			if (config.contextWindowTokens) {
				const contextTokens = Math.max(
					estimateChatPayloadTokens(messages),
					response?.usage?.totalTokens ?? 0,
				);
				outputGuard?.processResponse(contextTokens, config.contextWindowTokens);
				await emit({
					type: "context_update",
					tokens: contextTokens,
					maxTokens: config.contextWindowTokens,
				});
			}

			await emitTyped(config.extensionBus, {
				type: "turn_end",
				turnIndex: iteration,
				stopReason,
				message: assistant,
				toolResults,
			});
			await emit({
				type: "turn_end",
				turnId,
				stopReason,
				message: assistant,
				toolResults,
			});

			// Reset output guard after each completed turn
			outputGuard?.reset();

			const refreshedConfig = await config.refreshNextTurnConfig?.();
			if (refreshedConfig) {
				Object.assign(config, refreshedConfig);
				context.systemPrompt = refreshedConfig.systemPrompt;
				messages = [
					createSystemMessage(
						refreshedConfig.systemPrompt ?? "You are a helpful assistant.",
					),
					...messages.filter((message) => message.role !== "system"),
				];
				registry = createRegistry(refreshedConfig.tools ?? []);
			}

			const prepared = await prepareMessages(
				[
					config.prepareNextTurn,
					config.internalHooks?.prepareNextTurn,
					config.hooks?.prepareNextTurn,
				],
				{
					messages,
					iteration,
					hadToolCalls: toolCalls.length > 0,
				},
			);
			if (prepared) {
				messages = prepared;
				if (contextWasCompacted) config.onContextCompacted?.(messages);
			}

			// Fix #4: when a tool signals terminate, still drain followUps before exiting.
			// This prevents skipping queued follow-up messages (e.g. steering injected
			// mid-turn) just because a tool requested termination.
			if (toolTerminated) {
				const followUpsOnTerminate = await firstMessages([
					() =>
						config.getFollowUpMessages?.({
							messages,
							iteration,
							assistantText: assistantText(newMessages.at(-1)),
							stopReason: "stop",
						}),
					() =>
						config.internalHooks?.getFollowUpMessages?.({
							messages,
							iteration,
							assistantText: assistantText(newMessages.at(-1)),
							stopReason: "stop",
						}),
					() =>
						config.hooks?.getFollowUpMessages?.({
							messages,
							iteration,
							assistantText: assistantText(newMessages.at(-1)),
							stopReason: "stop",
						}),
				]);
				if (followUpsOnTerminate.length > 0) {
					pendingMessages = followUpsOnTerminate;
					hasMoreToolCalls = false;
					// Re-enter inner loop with follow-up messages
					continue;
				}
				const declared = getTaskStatus();
				return finish({
					status:
						declared?.status === "done"
							? "completed"
							: (declared?.status ?? "completed"),
					summary: declared?.summary,
					source: declared ? "structured" : "heuristic",
				});
			}

			// Fix #5: only invoke shouldStopAfterTurn when no tool calls ran.
			// Tool turns always continue unless the hook is explicitly wired to stop
			// on tool turns — checking it unconditionally causes premature exits when
			// hooks have stale state from a previous no-tool turn.
			const stop =
				toolCalls.length === 0
					? await shouldStop(
							[
								config.shouldStopAfterTurn,
								config.internalHooks?.shouldStopAfterTurn,
								config.hooks?.shouldStopAfterTurn,
							],
							{
								messages,
								iteration,
								hadToolCalls: false,
								message: assistant,
								toolResults,
							},
						)
					: false;
			// Acceptance stop rules take priority
			let acceptanceStop = false;
			if (!stop && shouldRunAcceptanceFinalization(resolved)) {
				acceptanceStop = checkStopRules(resolved);
			}
			if (stop || acceptanceStop) {
				return finish({ status: "completed", source: "heuristic" });
			}

			pendingMessages = await firstMessages([
				() => config.getSteeringMessages?.({ messages, iteration }),
				() =>
					config.internalHooks?.getSteeringMessages?.({ messages, iteration }),
				() => config.hooks?.getSteeringMessages?.({ messages, iteration }),
			]);
		}

		// Agent would stop here — drain followUp queue (pi-style outer loop).
		const followUps = await firstMessages([
			() =>
				config.getFollowUpMessages?.({
					messages,
					iteration,
					assistantText: assistantText(newMessages.at(-1)),
					stopReason: "stop",
				}),
			() =>
				config.internalHooks?.getFollowUpMessages?.({
					messages,
					iteration,
					assistantText: assistantText(newMessages.at(-1)),
					stopReason: "stop",
				}),
			() =>
				config.hooks?.getFollowUpMessages?.({
					messages,
					iteration,
					assistantText: assistantText(newMessages.at(-1)),
					stopReason: "stop",
				}),
		]);

		// Runner-level continuation nudge: fires when no follow-ups from hooks
		// and the model didn't signal completion. This is the FINAL safety net
		// — it fires when no hook returned follow-ups AND the model has no
		// explicit completion signal AND no structured stop was issued.
		// Capped to MAX_CONSECUTIVE_RUNNER_NUDGES to prevent infinite loops.
		const MAX_CONSECUTIVE_RUNNER_NUDGES = 3;
		if (config.continuationEnabled === true && followUps.length === 0) {
			const text = lastAssistantContent(newMessages);
			const hadTools = lastHadToolCalls(newMessages);
			const hasStructuredStop = getTaskStatus() !== null;

			if (
				!hadTools &&
				!looksComplete(text) &&
				!hasStructuredStop &&
				consecutiveRunnerNudges < MAX_CONSECUTIVE_RUNNER_NUDGES
			) {
				followUps.push({
					role: "user" as const,
					content:
						"Continue with the next step. If the task is fully complete, " +
						"say so explicitly. Otherwise keep working — do not stop prematurely.",
				});
				consecutiveRunnerNudges++;
			} else {
				// Model signaled completion, has structured stop, or cap reached — reset.
				consecutiveRunnerNudges = 0;
			}
		}

		if (followUps.length > 0) {
			pendingMessages = followUps;
			continue;
		}
		break;
	}

	// ── Reflection step ─────────────────────────────────────────────
	// Reflection is an additional provider call and must be explicitly enabled.
	// Keeping it opt-in makes the core loop deterministic and avoids surprising
	// latency/cost for callers that did not request an evaluator pass.
	const reflectionEnabled = config.reflectionConfig?.enabled === true;
	const maxReflections = config.reflectionConfig?.maxReflections ?? 2;
	let reflectionCount = 0;
	let reflectionMessages = [...newMessages];

	while (
		reflectionEnabled &&
		reflectionCount < maxReflections &&
		!looksComplete(lastAssistantContent(reflectionMessages))
	) {
		const reflectionResult = await runReflection(
			reflectionMessages,
			config.backend,
			config.reflectionConfig ?? { enabled: true },
			emit,
			config.signal,
		);

		if (!reflectionResult.result.needsMoreWork) {
			// Model says task is complete — break out
			break;
		}

		// Model wants to continue — inject a user message with suggested steps
		const suggestedSteps = reflectionResult.result.suggestedSteps.join("\n");
		const nudgeMessage: Message = {
			role: "user",
			content:
				reflectionResult.result.issues.length > 0
					? `Reflection found issues: ${reflectionResult.result.issues.join(", ")}. Address these and continue.`
					: `Task incomplete. ${suggestedSteps ? `Suggested: ${suggestedSteps}. ` : ""}Continue with the next step.`,
			timestamp: Date.now(),
		};

		reflectionMessages = [
			...reflectionMessages,
			{ role: "user", content: nudgeMessage.content, timestamp: Date.now() },
			{
				role: "assistant",
				content: reflectionResult.result.reasoning,
				timestamp: Date.now(),
			},
		];

		reflectionCount++;
	}

	if (reflectionCount >= maxReflections) {
		// Max reflections reached without clear completion
		await emit({
			type: "task_failed",
			reason: `Agent reached the ${maxReflections}-reflection safety limit without completing the task.`,
			iteration: reflectionCount,
			lastContent: lastAssistantContent(reflectionMessages),
		});
	}

	// Use reflection-enriched messages for final conclusion
	const finalMessagesForConclusion =
		reflectionMessages.length > newMessages.length
			? reflectionMessages
			: newMessages;

	if (iteration >= maxIterations) {
		const lastText = lastAssistantContent(newMessages);
		await emit({
			type: "task_failed",
			reason: `Agent reached the ${maxIterations}-turn safety limit without finishing`,
			iteration,
			lastContent: lastText,
		});
		await emit({
			type: "max_iterations",
			iterations: iteration,
			limit: maxIterations,
		});
	}

	// Emit conclusion / task_failed before agent_end
	const hadFollowUps = iteration < maxIterations;
	await emitConclusion(
		emit,
		finalMessagesForConclusion,
		iteration,
		maxIterations,
		hadFollowUps,
	);

	// ── Acceptance finalization ────────────────────────────────────────
	if (shouldRunAcceptanceFinalization(resolved) && !acceptanceReported) {
		const finalText = lastAssistantContent(finalMessagesForConclusion);

		// Run verification commands
		const verificationResults = await runAcceptanceVerification(
			resolved,
			config.signal,
		);

		// Validate criteria and build ledger
		const report = parsedReportOrReview(
			finalText,
			resolved,
			verificationResults,
			config,
			emit,
		);

		acceptanceReported = true;
		await emit({
			type: "acceptance_complete",
			status: report.status,
			report: report.ledger as unknown as Record<string, unknown>,
		});

		if (report.status === "failed") {
			acceptanceFailed = true;
		}
	}

	function parsedReportOrReview(
		finalText: string,
		resolved: ResolvedAcceptance,
		verificationResults: Array<{
			command: string;
			result: string;
			summary?: string;
		}>,
		_loopConfig: RunAgentLoopConfig,
		_emitFn: AgentEventSink,
	): {
		status: "passed" | "failed" | "timeout";
		ledger: {
			status: string;
			report?: AcceptanceReport;
			verification?: string[];
		};
	} {
		// Check for model-produced report
		const parsed = parseAcceptanceReport(finalText);
		if (!parsed.report && !parsed.error) {
			return {
				status: "failed",
				ledger: { status: "failed", verification: [] },
			};
		}

		// Build verification summary
		const verificationSummary = verificationResults.map(
			(v) =>
				`[${v.result.toUpperCase()}] ${v.command}${v.summary ? ` → ${v.summary.slice(0, 100)}` : ""}`,
		);

		// Validate criteria against report
		const report = parsed.report;
		const criteriaResults = resolved.criteria.map((c) => {
			const satisfied = report?.criteriaSatisfied?.some(
				(cs) =>
					cs.id === c.id &&
					(cs.status === "satisfied" ||
						(c.severity === "recommended" && cs.status === "partial")),
			);
			return {
				id: c.id,
				status: (satisfied ? "satisfied" : "failed") as "satisfied" | "failed",
				evidence: c.must,
			};
		});

		// Check for required review
		let reviewStatus = "not-required";
		if (resolved.review?.required) {
			reviewStatus = report?.criteriaSatisfied?.every(
				(cs) => cs.status === "satisfied",
			)
				? "passed"
				: "failed";
		}

		const allCriteriaPass = criteriaResults.every(
			(c) => c.status === "satisfied",
		);
		const allVerificationsPass = verificationResults.every(
			(v) => v.result === "passed",
		);

		return {
			status:
				allCriteriaPass && allVerificationsPass && reviewStatus !== "failed"
					? "passed"
					: "failed",
			ledger: {
				status: allCriteriaPass && allVerificationsPass ? "passed" : "failed",
				report: {
					...report,
					criteriaSatisfied: criteriaResults,
				} as AcceptanceReport,
				verification:
					verificationSummary.length > 0 ? verificationSummary : undefined,
			},
		};
	}

	// Final output guard reset when agent ends
	outputGuard?.reset();
	const declared = getTaskStatus();
	if (declared) {
		return finish({
			status: declared.status === "done" ? "completed" : declared.status,
			summary: declared.summary,
			source: "structured",
		});
	}
	if (config.signal?.aborted) {
		return finish({
			status: "cancelled",
			summary: "Operation aborted.",
			source: "runtime",
		});
	}

	// If acceptance was required but not satisfied, mark as failed
	if (acceptanceFailed) {
		return finish({
			status: "failed",
			summary:
				"Acceptance contract not satisfied: no valid acceptance report produced.",
			source: "runtime",
		});
	}

	// Replace newMessages with reflection-enriched messages so finish() returns them
	newMessages.splice(0, newMessages.length, ...finalMessagesForConclusion);

	const finalText = lastAssistantContent(finalMessagesForConclusion);
	return finish({
		status:
			iteration >= maxIterations || looksNonCommittal(finalText)
				? "failed"
				: "completed",
		summary: finalText || undefined,
		source: iteration >= maxIterations ? "runtime" : "heuristic",
	});
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
