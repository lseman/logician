// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.

import { compactToFit } from "../../compaction/index.ts";
import {
	emitConclusion,
	lastAssistantContent,
	lastHadToolCalls,
} from "../../runtime/conclusion-policy.ts";
import { executeToolBatch } from "../../runtime/tool-batch-controller.ts";
import { ToolRegistry } from "../../tools/shared/registry.ts";
import {
	parseTextToolCalls,
	stripTextToolCalls,
} from "../../tools/shared/text-to-tool-calls.ts";
import {
	type AcceptanceConfig,
	evaluateAcceptanceReport,
	formatAcceptancePrompt,
	parseAcceptanceReport,
	type ResolvedAcceptance,
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	verifyAcceptanceCommands,
} from "../guards/acceptance-contract.ts";
import type { OutputGuard } from "../guards/output-guard.ts";
import { awaitsUserInput, looksComplete } from "../guards/response-patterns.ts";
import {
	applyHeaderPatch,
	assistantText,
	emitMessagePair,
	firstMessages,
	type LoopCallbacks,
	prepareMessages,
	shouldStop,
	stopReasonFor,
	transformMessages,
	withSystemPrompt,
} from "../loop/callbacks.ts";
import { buildProviderRequestOptions } from "../loop/provider-options.ts";
import { processProviderResponse } from "../loop/provider-response.ts";
import { buildStreamingCallbacks } from "../loop/provider-streaming.ts";
import { runReflection } from "../loop/reflection.ts";
import {
	isToolFailureResult,
	selectAdaptiveMode,
	taskObjectiveFromMessages,
} from "../tasks/adaptive-mode.ts";
import { resolveOutcome } from "../tasks/outcome-resolution.ts";
import { runWithTaskState } from "../tasks/run-task-state.ts";
import { getTaskStatus, resetTaskStatus } from "../tasks/task-status-state.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentEventSink,
	AgentMessage,
	CompactableMessage,
	Message,
	Tool,
	ToolCall,
} from "../types/index.ts";
import { getInferenceMode } from "../types/types-config.ts";
import { resolveAgentSettings } from "./agent-settings.ts";
import type { LLMBackend } from "./backend.ts";
import {
	evaluateStopPolicies,
	type RunOutcomeStatus,
	resolveExecutionPolicy,
} from "./execution-policy.ts";
import { checkBudget } from "./exit-path.ts";
import {
	type HarnessIntervention,
	HarnessInterventionController,
	type InterventionInput,
} from "./intervention-controller.ts";
import {
	convertToChatFormat,
	createSystemMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "./messages.ts";
import { RunBudgetController, type RunBudgetDecision } from "./run-budget.ts";
import { ToolResultCache } from "./tool-cache.ts";

export type { ReflectionConfig } from "../loop/reflection.ts";

export { STEERING_INTERRUPT_SUMMARY } from "./run-kernel-events.ts";

import { STEERING_INTERRUPT_SUMMARY } from "./run-kernel-events.ts";

export const STEERING_INTERRUPT_NAME = "SteeringInterruptError";

export function createSteeringInterruptReason(): Error {
	const error = new Error(STEERING_INTERRUPT_SUMMARY);
	error.name = STEERING_INTERRUPT_NAME;
	return error;
}

function isSteeringInterrupt(signal: AbortSignal | undefined): boolean {
	return (
		signal?.aborted === true &&
		signal.reason instanceof Error &&
		signal.reason.name === STEERING_INTERRUPT_NAME
	);
}

export interface RunAgentLoopContext {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
	cwd?: string;
}

export interface RunAgentLoopConfig extends AgentConfig, LoopCallbacks {
	backend: LLMBackend;
	signal?: AbortSignal;
	maxIterations?: number;
	/** Called when in-loop context-full recovery replaces the active transcript. */
	onContextCompacted?: (messages: Message[]) => void;
	/** Refresh mutable harness configuration after a completed turn. */
	refreshNextTurnConfig?: () => Promise<AgentConfig> | AgentConfig;
	// Output guard config (optional). When provided, the loop uses OutputGuard
	// to handle context_full, retryable errors, and empty responses.
	outputGuard?: OutputGuard | null;
	/** Resolve the acceptance config at call time (allows harness to update it). */
	getAcceptanceConfig?: () => AcceptanceConfig | undefined;

	/** Prior intervention incidents restored from the typed Run Kernel projection. */
	initialInterventions?: HarnessIntervention[];
	/** Task-spanning counters restored from the Run Kernel. */
	durableBudgetState?: {
		providerCalls: number;
		toolCalls: number;
		tokens: number;
		startedAt?: number;
	};
	/** Persist accepted budget consumption before the corresponding work starts. */
	onBudgetConsumed?: (
		resource: "provider_call" | "tool_call" | "token",
		amount: number,
	) => void;
	onToolIntent?: NonNullable<
		import("../../runtime/tool-batch-controller.ts").ToolBatchControllerOptions["onToolIntent"]
	>;
	onToolResult?: NonNullable<
		import("../../runtime/tool-batch-controller.ts").ToolBatchControllerOptions["onToolResult"]
	>;
	onPermissionDecision?: NonNullable<
		import("../../runtime/tool-batch-controller.ts").ToolBatchControllerOptions["onPermissionDecision"]
	>;
	onToolCommit?: (toolCallId: string) => void | Promise<void>;
}

async function runAgentLoopInTaskScope(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	const downstreamEmit = emit;
	let eventSequence = 0;
	emit = event =>
		downstreamEmit({
			...event,
			seq: ++eventSequence,
			ts: Date.now(),
		});
	let messages = [
		...withSystemPrompt(context.systemPrompt, context.messages),
		...prompts,
	];
	const newMessages: Message[] = [...prompts];
	resetTaskStatus();
	const finish = async (outcome: {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	}): Promise<Message[]> => {
		await emit({ type: "run_outcome", ...outcome });
		await emit({ type: "agent_end", messages: newMessages });
		return newMessages;
	};
	let settings = resolveAgentSettings(config);
	const maxIterations = settings.maxIterations;
	const executionPolicy = resolveExecutionPolicy(settings.executionProfile);
	const interventionController = new HarnessInterventionController();
	interventionController.replay(config.initialInterventions ?? []);
	const intervene = (input: InterventionInput): Promise<void> | void =>
		emit({
			type: "harness_intervention",
			...interventionController.record(input),
		});
	// ── P0-1: Shared tool result cache ─────────────────────────────────
	const cache = new ToolResultCache(
		config.cacheSize ?? 2000,
		config.cacheTtlMs ?? 60_000,
	);
	const createRegistry = (tools: Tool[]): ToolRegistry => {
		const next = new ToolRegistry({
			cwd: context.cwd ?? config.cwd,
			allowedPaths: config.allowedPaths,
			allowAllPaths: config.allowAllPaths,
			signal: config.signal,
			onQuestionRequest: config.onQuestionRequest,
			cache,
			maxResultChars: config.truncation?.toolResultMaxChars,
		});
		next.registerMany(tools);
		return next;
	};
	let registry = createRegistry(context.tools ?? config.tools ?? []);

	const outputGuard = config.outputGuard;
	let iteration = 0;
	let consecutiveRunnerNudges = 0;
	let lastRunnerNudgeIteration = -1;
	let lastToolWorkIteration = -1;
	let performedToolWork = false;
	let toolFailures = 0;
	const adaptiveObjective = taskObjectiveFromMessages([
		...context.messages,
		...prompts,
	]);
	let contextWasCompacted = false;
	const reflectionEnabled =
		executionPolicy.embeddedPoliciesEnabled &&
		config.reflectionConfig?.enabled === true;
	const maxReflections = config.reflectionConfig?.maxReflections ?? 2;
	let reflectionCount = 0;
	let reflectionFailed = false;
	let lastAdaptiveSelection = "";
	const runBudget = new RunBudgetController(
		{
			maxElapsedMs: 30 * 60_000,
			maxTokens: config.maxTotalTokens,
			...config.runBudget,
		},
		Date.now,
		config.durableBudgetState,
		consumption =>
			config.onBudgetConsumed?.(consumption.resource, consumption.amount),
	);

	async function finishForBudgetExhaustion(
		decision: RunBudgetDecision,
	): Promise<Message[]> {
		await intervene({
			kind: "budget",
			cause: "run_budget",
			detector: "run_budget",
			message: decision.reason ?? "Run budget exhausted.",
			iteration,
			counters: {
				providerCalls: decision.snapshot.providerCalls,
				toolCalls: decision.snapshot.toolCalls,
				elapsedMs: decision.snapshot.elapsedMs,
			},
		});
		return finish({
			status: "blocked",
			summary: decision.reason,
			source: "runtime",
		});
	}

	// ── Acceptance contract tracking ─────────────────────────────────────
	let resolvedAcceptance: ResolvedAcceptance | null = null;
	let acceptanceReported = false;
	let acceptanceFailed = false;
	let acceptanceFinalizationTurns = 0;

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

	function drainSteering(): Promise<Message[]> {
		return firstMessages([
			() => config.getSteeringMessages?.({ messages, iteration }),
			() =>
				config.internalHooks?.getSteeringMessages?.({ messages, iteration }),
			() => config.hooks?.getSteeringMessages?.({ messages, iteration }),
		]);
	}

	function drainFollowUps(): Promise<Message[]> {
		return firstMessages([
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
	}

	let pendingMessages = await drainSteering();

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
			prompt: prompts.map(p => p.content).join("\n"),
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
	const resolved = executionPolicy.embeddedPoliciesEnabled
		? resolveAcceptance()
		: resolveEffectiveAcceptance({ explicit: undefined });
	if (shouldRunAcceptanceFinalization(resolved)) {
		const accPrompt = formatAcceptancePrompt(resolved);
		if (accPrompt) {
			const existingSystem = messages
				.filter(m => m.role === "system")
				.map(m => m.content)
				.join("\n\n");
			messages = [
				{
					role: "system" as const,
					content: existingSystem
						? `${existingSystem}\n\n${accPrompt}`
						: accPrompt,
					timestamp: Date.now(),
				},
				...messages.filter(m => m.role !== "system"),
			];
		}
	}

	while (iteration < maxIterations) {
		if (config.signal?.aborted) {
			const steeringInterrupt = isSteeringInterrupt(config.signal);
			if (!steeringInterrupt) {
				await emit({ type: "error", message: "Operation aborted" });
			}
			return finish({
				status: "cancelled",
				summary: steeringInterrupt
					? STEERING_INTERRUPT_SUMMARY
					: "Operation aborted before the provider request.",
				source: "runtime",
			});
		}

		let hasMoreToolCalls = true;
		while (
			(hasMoreToolCalls || pendingMessages.length > 0) &&
			iteration < maxIterations
		) {
			const providerBudget = checkBudget(runBudget, "provider_call");
			if (!providerBudget.allowed) {
				return finishForBudgetExhaustion(providerBudget);
			}
			iteration++;
			const turnId = `turn_${iteration}`;
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
				let requestTimeoutMs =
					config.streamOptions?.timeoutMs ?? config.turnTimeoutMs;
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
					// Resolve inference mode params — they override individual config values.
					const adaptiveDecision =
						settings.inferenceMode === "auto"
							? selectAdaptiveMode({
									objective: adaptiveObjective,
									performedToolWork,
									toolFailures,
								})
							: undefined;
					const effectiveMode =
						adaptiveDecision?.mode ?? settings.inferenceMode;
					if (adaptiveDecision) {
						const selectionKey = `${adaptiveDecision.mode}:${adaptiveDecision.reason}`;
						if (selectionKey !== lastAdaptiveSelection) {
							lastAdaptiveSelection = selectionKey;
							await emit({
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
						toolDefinitions: registry.toToolDefinitions(),
						settings,
						config,
						requestHeaders: requestHeaders as Record<string, string>,
						requestTimeoutMs: requestTimeoutMs as number,
						requestMaxRetries: requestMaxRetries as number,
						requestCacheRetention,
						requestMetadata,
						modeDef,
						signal: config.signal,
						payloadHooks,
					});
					requestOptions.callbacks = buildStreamingCallbacks(
						turnId,
						queueProviderEvent,
					);
					response = await config.backend.generate(
						chatMessages,
						requestOptions,
					);
					await Promise.all(providerEvents);
					if (activeRetryAttempt > 0) {
						await emit({
							type: "agent_retry_end",
							attempt: activeRetryAttempt,
							success: true,
						});
					}
					break;
				} catch (llmError) {
					// Cancellation wins over provider error classification. Some provider
					// clients replace an AbortSignal cancellation with a generic Error;
					// sending that through OutputGuard would create a fake retry.
					const cancelled =
						config.signal?.aborted ||
						(llmError instanceof Error && llmError.name === "AbortError");
					if (cancelled) {
						const steeringInterrupt = isSteeringInterrupt(config.signal);
						if (!steeringInterrupt) {
							await emit({ type: "error", message: "Operation aborted" });
						}
						return finish({
							status: "cancelled",
							summary: steeringInterrupt
								? STEERING_INTERRUPT_SUMMARY
								: "Operation aborted",
							source: "runtime",
						});
					}

					const guardResult = outputGuard?.handleError(llmError);

					if (!guardResult || guardResult.action === "abort") {
						await emit({
							type: "error",
							message: guardResult?.message ?? String(llmError),
							error: llmError,
						});
						if (!cancelled && activeRetryAttempt > 0) {
							await emit({
								type: "agent_retry_end",
								attempt: guardResult?.attempt ?? activeRetryAttempt,
								success: false,
							});
						}
						return finish({
							status: "failed",
							summary: guardResult?.message ?? String(llmError),
							source: "runtime",
						});
					}

					activeRetryAttempt = guardResult.attempt ?? activeRetryAttempt + 1;
					// Emit retry start event (OutputGuard handles error classification,
					// loop runner emits the event to avoid duplicates).
					await emit({
						type: "agent_retry_start",
						attempt: activeRetryAttempt,
						maxRetries: guardResult.maxRetries ?? 3,
						delayMs: undefined,
						error: guardResult.message ?? String(llmError),
					});
					await intervene({
						kind: "retry",
						cause: guardResult.action,
						detector: "provider_error_guard",
						message: guardResult.message ?? String(llmError),
						iteration,
						counters: { attempt: activeRetryAttempt },
						limits: { maxRetries: guardResult.maxRetries ?? 3 },
					});

					if (guardResult.action === "compact_then_retry") {
						const compacted = await compactToFit(
							messages as CompactableMessage[],
							{
								triggerTokens: 0,
								targetTokens: config.contextWindowTokens
									? Math.floor(config.contextWindowTokens * 0.75)
									: undefined,
							},
						);
						if (!compacted.changed) {
							await emit({
								type: "agent_retry_end",
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
						messages = compacted.messages as unknown as Message[];
						contextWasCompacted = true;
						config.onContextCompacted?.(messages);
						await emit({
							type: "context_update",
							tokens: compacted.tokensAfter,
							maxTokens: config.contextWindowTokens,
							compacted: true,
						});
						await intervene({
							kind: "compaction",
							cause: "context_full",
							detector: "provider_retry",
							message: `Context compacted from ${compacted.tokensBefore} to ${compacted.tokensAfter} tokens before retrying.`,
							iteration,
							counters: {
								tokensBefore: compacted.tokensBefore,
								tokensAfter: compacted.tokensAfter,
							},
						});
					}
				}
			}

			const tokenBudget = checkBudget(
				runBudget,
				"tokens",
				response?.usage?.totalTokens ?? 0,
			);
			if (!tokenBudget.allowed) {
				return finishForBudgetExhaustion(tokenBudget);
			}
			const processResult = processProviderResponse({
				response,
				registry,
				outputGuard: outputGuard ?? null,
				messages,
				newMessages,
				turnId,
				iteration,
				emit,
				config,
			});

			let toolCalls: ToolCall[];
			let assistant: Message;
			let assistantContent: string;
			if (processResult.success) {
				toolCalls = processResult.toolCalls;
				assistantContent = processResult.assistantContent;
				assistant = processResult.assistant;
				if (toolCalls.some(call => call.name !== "task_status")) {
					performedToolWork = true;
					lastToolWorkIteration = iteration;
				}
			} else {
				return finish({
					status: "failed",
					summary:
						processResult.errorMessage ?? "Model returned empty response.",
					source: "runtime",
				});
			}
			const rawStopReason =
				(response?.stopReason as "stop" | "length" | "error") ?? "stop";
			const stopReason = stopReasonFor(rawStopReason, toolCalls);

			hasMoreToolCalls = false;
			const toolBudget = checkBudget(runBudget, "tool_batch", toolCalls.length);
			if (!toolBudget.allowed) {
				return finishForBudgetExhaustion(toolBudget);
			}
			const batch = await executeToolBatch({
				registry,
				toolCalls,
				rawStopReason,
				toolExecution: settings.toolExecution,
				iteration,
				signal: config.signal,
				internalHooks: config.internalHooks,
				hooks: config.hooks,
				permissions: config.permissions,
				onPermissionRequest: config.onPermissionRequest,
				onPermissionDecision: config.onPermissionDecision,
				onToolIntent: config.onToolIntent,
				onToolResult: config.onToolResult,
				emit,
			});
			const toolResults = batch.messages;
			const toolTerminated = batch.terminated;
			for (const toolResult of toolResults) {
				if (isToolFailureResult(String(toolResult.content ?? ""))) {
					toolFailures++;
				}
				messages.push(toolResult);
				newMessages.push(toolResult);
				await emitMessagePair(emit, turnId, toolResult);
				hasMoreToolCalls = true;
			}
			for (const toolCallId of batch.executedToolCallIds)
				await config.onToolCommit?.(toolCallId);

			// The final usage-only SSE chunk is optional and many local providers
			// omit it. Estimate the serialized conversation as a reliable fallback
			// so context usage never remains stuck at zero.
			const contextTokens = Math.max(
				estimateChatPayloadTokens(messages, registry.toToolDefinitions()),
				response?.usage?.totalTokens ?? 0,
			);
			await emit({
				type: "context_update",
				tokens: contextTokens,
				maxTokens: config.contextWindowTokens,
				cachedTokens: response?.usage?.cachedTokens ?? null,
				promptTokens: response?.usage?.promptTokens ?? null,
				completionTokens: response?.usage?.completionTokens ?? null,
			});
			if (config.contextWindowTokens) {
				const budgetResult = outputGuard?.processResponse(
					contextTokens,
					config.contextWindowTokens,
				);
				// budget_exhausted is a harder threshold than proactive compaction's
				// (95% vs 80%) — if we're here, proactive compaction already failed
				// to keep up (e.g. cooldown window, or a single oversized turn).
				// Compact immediately rather than waiting for the next request to
				// fail with context_full.
				if (budgetResult?.action === "budget_exhausted") {
					const compacted = await compactToFit(
						messages as CompactableMessage[],
						{
							triggerTokens: 0,
							targetTokens: Math.floor(config.contextWindowTokens * 0.75),
						},
					);
					if (compacted.changed) {
						messages = compacted.messages as unknown as Message[];
						contextWasCompacted = true;
						config.onContextCompacted?.(messages);
						await emit({
							type: "context_update",
							tokens: compacted.tokensAfter,
							maxTokens: config.contextWindowTokens,
							compacted: true,
						});
						await intervene({
							kind: "compaction",
							cause: "budget_exhausted",
							detector: "context_budget",
							message: `Context compacted from ${compacted.tokensBefore} to ${compacted.tokensAfter} tokens.`,
							iteration,
							counters: {
								tokensBefore: compacted.tokensBefore,
								tokensAfter: compacted.tokensAfter,
							},
						});
					}
				}
			}

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
				settings = resolveAgentSettings(config);
				context.systemPrompt = refreshedConfig.systemPrompt;
				messages = [
					createSystemMessage(
						refreshedConfig.systemPrompt ?? "You are a helpful assistant.",
					),
					...messages.filter(message => message.role !== "system"),
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
				const followUpsOnTerminate = await drainFollowUps();
				if (followUpsOnTerminate.length > 0) {
					if (
						!followUpsOnTerminate.some(message =>
							String(message.content).startsWith("[continuation-nudge:"),
						)
					) {
						await intervene({
							kind: "continuation",
							cause: "follow_up_after_termination",
							detector: "follow_up_queue",
							message: `Harness scheduled ${followUpsOnTerminate.length} follow-up message(s) after tool termination.`,
							iteration,
						});
					}
					pendingMessages = followUpsOnTerminate;
					hasMoreToolCalls = false;
					// Re-enter inner loop with follow-up messages
					continue;
				}
				return finish(
					resolveOutcome({
						declared: getTaskStatus(),
						structuredOutcomeRequired:
							performedToolWork && registry.has("task_status"),
					}),
				);
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
				return finish(
					resolveOutcome({
						declared: getTaskStatus(),
						structuredOutcomeRequired:
							performedToolWork && registry.has("task_status"),
					}),
				);
			}

			pendingMessages = await drainSteering();
		}

		// Agent would stop here — drain followUp queue (pi-style outer loop).
		const followUps = await drainFollowUps();
		if (
			followUps.length > 0 &&
			!followUps.some(message =>
				String(message.content).startsWith("[continuation-nudge:"),
			)
		) {
			await intervene({
				kind: "continuation",
				cause: "follow_up",
				detector: "follow_up_queue",
				message: `Harness scheduled ${followUps.length} follow-up message(s).`,
				iteration,
				action: "continue",
			});
		}

		// Runner-level continuation nudge: fires when no follow-ups from hooks
		// and the model didn't signal completion. This is the FINAL safety net
		// — it fires when no hook returned follow-ups AND the model has no
		// explicit completion signal AND no structured stop was issued.
		// Capped to MAX_CONSECUTIVE_RUNNER_NUDGES to prevent infinite loops.
		const MAX_CONSECUTIVE_RUNNER_NUDGES = 3;
		let continuationExhausted = false;
		if (
			executionPolicy.embeddedPoliciesEnabled &&
			config.continuationEnabled === true &&
			followUps.length === 0
		) {
			const text = lastAssistantContent(newMessages);
			const hadTools = lastHadToolCalls(newMessages);
			const waitingForUser = awaitsUserInput(text);
			const hasStructuredStop = getTaskStatus() !== null;
			const hasAcceptanceReport =
				shouldRunAcceptanceFinalization(resolved) &&
				parseAcceptanceReport(text).report !== undefined;

			const requiresStructuredConclusion =
				performedToolWork && registry.has("task_status");

			// Real tool work since the last nudge means the run is actually
			// progressing, not stalled — give it a fresh nudge budget rather than
			// counting this stall toward the same cap as the last one.
			if (lastToolWorkIteration > lastRunnerNudgeIteration) {
				consecutiveRunnerNudges = 0;
			}

			const eligibleForNudge =
				!hadTools &&
				!waitingForUser &&
				!hasAcceptanceReport &&
				requiresStructuredConclusion &&
				!hasStructuredStop;
			if (
				eligibleForNudge &&
				consecutiveRunnerNudges < MAX_CONSECUTIVE_RUNNER_NUDGES
			) {
				const nudgeTag = "[continuation-nudge:structured-conclusion]";
				const nudgeContent =
					`${nudgeTag} Do not stop yet without a structured conclusion. Verify that every requested step is complete. ` +
					"If work remains, continue with the next step. If the task is complete, blocked, failed, or needs user input, " +
					"call task_status with the accurate status as your final action.";
				followUps.push({ role: "user" as const, content: nudgeContent });
				await intervene({
					kind: "continuation",
					cause: "missing_structured_conclusion",
					detector: "runner_continuation",
					message: nudgeContent,
					iteration,
					counters: { consecutiveRunnerNudges },
					limits: { maxConsecutiveNudges: MAX_CONSECUTIVE_RUNNER_NUDGES },
				});
				consecutiveRunnerNudges++;
				lastRunnerNudgeIteration = iteration;
			} else if (
				eligibleForNudge &&
				consecutiveRunnerNudges >= MAX_CONSECUTIVE_RUNNER_NUDGES
			) {
				continuationExhausted = true;
				await intervene({
					kind: "continuation",
					cause: "continuation_exhausted",
					detector: "runner_continuation",
					message: `Continuation stopped after ${MAX_CONSECUTIVE_RUNNER_NUDGES} consecutive nudges without observable tool progress.`,
					iteration,
					counters: { consecutiveRunnerNudges },
					limits: { maxConsecutiveNudges: MAX_CONSECUTIVE_RUNNER_NUDGES },
				});
			} else {
				// Model signaled completion, has structured stop, or cap reached — reset.
				consecutiveRunnerNudges = 0;
			}
		}

		if (followUps.length > 0) {
			pendingMessages = followUps;
			continue;
		}
		if (continuationExhausted) {
			return finish({
				status: "blocked",
				summary: `Continuation exhausted after ${MAX_CONSECUTIVE_RUNNER_NUDGES} nudges without tool progress.`,
				source: "runtime",
			});
		}

		// A final question hands control back to the user. It must beat
		// reflection, acceptance finalization, and all other synthetic turns;
		// otherwise the loop fabricates an answer by prompting the model again.
		if (
			executionPolicy.embeddedPoliciesEnabled &&
			awaitsUserInput(lastAssistantContent(newMessages))
		) {
			return finish({
				status: "needs_input",
				summary: "Agent is waiting for the user's answer.",
				source: "heuristic",
			});
		}

		const policyDecision = await evaluateStopPolicies(config.stopPolicies, {
			messages,
			newMessages,
			iteration,
			signal: config.signal,
		});
		if (policyDecision?.action === "continue") {
			await intervene({
				kind: "continuation",
				cause: "stop_policy",
				detector: "custom_stop_policy",
				message: `A stop policy continued the run with ${policyDecision.messages.length} follow-up message(s).`,
				iteration,
				action: "continue",
			});
			if (policyDecision.messages.length > 0) {
				pendingMessages = policyDecision.messages;
				continue;
			}
		} else if (policyDecision?.action === "finish") {
			return finish({
				status: policyDecision.status,
				summary: policyDecision.summary,
				source: "structured",
			});
		}

		if (!executionPolicy.embeddedPoliciesEnabled) break;

		// A failed acceptance report is actionable feedback, not an immediate
		// terminal failure. Give the provider a bounded number of real turns to
		// correct the report (and, when needed, the underlying work).
		if (shouldRunAcceptanceFinalization(resolved) && !acceptanceReported) {
			const verificationResults = await verifyAcceptanceCommands(resolved, {
				cwd: config.cwd,
				signal: config.signal,
			});
			const report = evaluateAcceptanceReport(
				lastAssistantContent(newMessages),
				resolved,
				verificationResults,
			);
			if (report.status === "passed") {
				acceptanceReported = true;
				await emit({
					type: "acceptance_complete",
					status: report.status,
					report: report.ledger as unknown as Record<string, unknown>,
				});
				break;
			} else if (
				acceptanceFinalizationTurns < (resolved.maxFinalizationTurns ?? 3) &&
				iteration < maxIterations
			) {
				acceptanceFinalizationTurns++;
				const acceptanceRetryContent =
					`[continuation-nudge:acceptance-retry] Acceptance validation failed (attempt ${acceptanceFinalizationTurns}/${resolved.maxFinalizationTurns ?? 3}). ` +
					"Review the acceptance contract, fix any unmet criteria or verification failures, and finish with a valid acceptance-report block.";
				pendingMessages = [{ role: "user", content: acceptanceRetryContent }];
				await intervene({
					kind: "verification",
					cause: "acceptance_failed",
					detector: "acceptance_contract",
					message: acceptanceRetryContent,
					iteration,
					counters: { acceptanceFinalizationTurns },
					limits: {
						maxFinalizationTurns: resolved.maxFinalizationTurns ?? 3,
					},
				});
				continue;
			} else {
				acceptanceReported = true;
				acceptanceFailed = true;
				await emit({
					type: "acceptance_complete",
					status: report.status,
					report: report.ledger as unknown as Record<string, unknown>,
				});
				break;
			}
		}

		// Reflection is a verifier, not a synthetic assistant turn. When it finds
		// unfinished work, feed its findings back through the normal provider/tool
		// loop so the agent can actually correct the result.
		if (
			reflectionEnabled &&
			!looksComplete(lastAssistantContent(newMessages))
		) {
			if (reflectionCount >= maxReflections) {
				reflectionFailed = true;
				await emit({
					type: "task_failed",
					reason: `Agent reached the ${maxReflections}-reflection safety limit without completing the task.`,
					iteration: reflectionCount,
					lastContent: lastAssistantContent(newMessages),
				});
				break;
			}
			const reflection = await runReflection(
				newMessages,
				config.backend,
				config.reflectionConfig ?? { enabled: true },
				emit,
				config.signal,
			);
			reflectionCount++;
			if (reflection.result.needsMoreWork) {
				const suggested = reflection.result.suggestedSteps.join("; ");
				const reflectionRetryContent =
					reflection.result.issues.length > 0
						? `[continuation-nudge:reflection-retry] Reflection found issues: ${reflection.result.issues.join(", ")}. Address them and continue working.`
						: `[continuation-nudge:reflection-retry] Reflection found the task incomplete. ${suggested ? `Suggested next steps: ${suggested}. ` : ""}Continue working.`;
				pendingMessages = [{ role: "user", content: reflectionRetryContent }];
				await intervene({
					kind: "verification",
					cause: "reflection_incomplete",
					detector: "reflection",
					message: reflectionRetryContent,
					iteration,
					counters: { reflectionCount },
					limits: { maxReflections },
				});
				continue;
			}
		}
		break;
	}

	const finalMessagesForConclusion = newMessages;

	if (iteration >= maxIterations) {
		await emit({
			type: "max_iterations",
			iterations: iteration,
			limit: maxIterations,
		});
	}

	// Emit conclusion / task_failed before agent_end
	if (executionPolicy.embeddedPoliciesEnabled && iteration < maxIterations) {
		const hadFollowUps = iteration < maxIterations;
		await emitConclusion(
			emit,
			finalMessagesForConclusion,
			iteration,
			maxIterations,
			hadFollowUps,
		);
	}

	// ── Acceptance finalization ────────────────────────────────────────
	if (shouldRunAcceptanceFinalization(resolved) && !acceptanceReported) {
		const finalText = lastAssistantContent(finalMessagesForConclusion);

		// Run verification commands
		const verificationResults = await verifyAcceptanceCommands(resolved, {
			cwd: config.cwd,
			signal: config.signal,
		});

		// Validate criteria and build ledger
		const report = evaluateAcceptanceReport(
			finalText,
			resolved,
			verificationResults,
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

	// Final output guard reset when agent ends
	outputGuard?.reset();
	// Acceptance failure must take precedence over a model-declared `done`.
	if (acceptanceFailed) {
		return finish({
			status: "failed",
			summary:
				"Acceptance contract not satisfied after the configured finalization turns.",
			source: "runtime",
		});
	}
	const declared = executionPolicy.embeddedPoliciesEnabled
		? getTaskStatus()
		: null;
	if (declared || (performedToolWork && registry.has("task_status"))) {
		return finish(
			resolveOutcome({
				declared,
				structuredOutcomeRequired:
					performedToolWork && registry.has("task_status"),
			}),
		);
	}
	if (config.signal?.aborted) {
		return finish({
			status: "cancelled",
			summary: isSteeringInterrupt(config.signal)
				? STEERING_INTERRUPT_SUMMARY
				: "Operation aborted.",
			source: "runtime",
		});
	}

	// Replace newMessages with reflection-enriched messages so finish() returns them
	newMessages.splice(0, newMessages.length, ...finalMessagesForConclusion);

	const finalText = lastAssistantContent(finalMessagesForConclusion);
	return finish({
		status:
			iteration >= maxIterations || reflectionFailed || false
				? "failed"
				: "completed",
		summary: finalText || undefined,
		source:
			iteration >= maxIterations || !executionPolicy.embeddedPoliciesEnabled
				? "runtime"
				: "heuristic",
	});
}

export function runAgentLoop(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	return runWithTaskState(() =>
		runAgentLoopInTaskScope(context, prompts, config, emit),
	);
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
