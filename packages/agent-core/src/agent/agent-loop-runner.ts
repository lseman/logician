// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.

import { compactToFit } from "../compaction/index.ts";
import type { ExtensionEventBus } from "../hooks/extensions/event-bus.ts";
import type { ExtensionEvent as TypedExtensionEvent } from "../hooks/extensions/events.ts";
import {
	emitConclusion,
	lastAssistantContent,
	lastHadToolCalls,
} from "../runtime/conclusion-policy.ts";
import { executeToolBatch } from "../runtime/tool-batch-controller.ts";
import { ToolRegistry } from "../tools/shared/registry.ts";
import {
	parseTextToolCalls,
	stripTextToolCalls,
} from "../tools/shared/text-to-tool-calls.ts";
import type { LLMBackend } from "./backend.ts";
import { getInferenceMode } from "./configuration/inference-modes.ts";
import {
	evaluateStopPolicies,
	resolveExecutionPolicy,
} from "./execution-policy.ts";
import {
	type AcceptanceConfig,
	type AcceptanceReport,
	formatAcceptancePrompt,
	parseAcceptanceReport,
	type ResolvedAcceptance,
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
} from "./guards/acceptance-contract.ts";
import type { GuardEngine } from "./guards/guard-engine.ts";
import type { OutputGuard } from "./guards/output-guard.ts";
import {
	awaitsUserInput,
	looksComplete,
} from "./guards/response-patterns.ts";
import {
	type HarnessIntervention,
	type HarnessInterventionAction,
	HarnessInterventionController,
	type InterventionInput,
} from "./intervention-controller.ts";
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
	waitForRetryDelay,
	withSystemPrompt,
} from "./loop/callbacks.ts";
import { runReflection } from "./loop/reflection.ts";
import {
	convertToChatFormat,
	createAssistantMessage,
	createSystemMessage,
	createUserMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
	sanitizeToolCallArguments,
} from "./messages.ts";
import { RunBudgetController } from "./run-budget.ts";
import { resolveCompletionGate } from "./tasks/completion-gate.ts";
import { runWithTaskState } from "./tasks/run-task-state.ts";
import {
	type ExplicitTaskState,
	shouldProjectTaskState,
	TaskStateController,
	taskObjectiveFromMessages,
} from "./tasks/task-state-controller.ts";
import { getTaskStatus, resetTaskStatus } from "./tasks/task-status-state.ts";
import { ToolResultCache } from "./tool-cache.ts";
import type {
	AgentConfig,
	AgentEvent,
	AgentMessage,
	CompactableMessage,
	Message,
	Tool,
} from "./types.ts";

export type { ReflectionConfig } from "./loop/reflection.ts";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

export const STEERING_INTERRUPT_NAME = "SteeringInterruptError";
export const STEERING_INTERRUPT_SUMMARY =
	"Current provider response interrupted to apply steering.";

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

// Cheap heuristic for "this objective probably needs a subgoal breakdown" —
// long objectives, or ones with multiple imperative clauses, benefit most
// from decomposition; short single-action asks ("fix the typo") do not.
const MULTI_STEP_WORD_THRESHOLD = 25;
const MULTI_STEP_CONJUNCTION_PATTERN = /\b(?:and then|,\s*then|;\s*|\band\b.*\band\b)/i;

function looksMultiStep(objective: string): boolean {
	if (!objective) return false;
	const wordCount = objective.trim().split(/\s+/).filter(Boolean).length;
	if (wordCount >= MULTI_STEP_WORD_THRESHOLD) return true;
	return MULTI_STEP_CONJUNCTION_PATTERN.test(objective);
}

/** Emit a typed extension event if the bus is available. */
async function emitTyped(
	emitter: ExtensionEventBus | undefined,
	event: TypedExtensionEvent,
): Promise<void> {
	if (!emitter) return;
	await emitter.emit(event);
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
	/** Central guard engine (optional). When provided, the loop uses it for
	 *  composite turn-level loop detection, progress tracking, and graduated
	 *  intervention (nudge/restrict/pause/abort) instead of the narrower
	 *  outputGuard-only loop check. */
	guardEngine?: GuardEngine | null;
	/** Typed extension event bus for structured extension subscriptions. */
	extensionBus?: ExtensionEventBus;
	/** Resolve the acceptance config at call time (allows harness to update it). */
	getAcceptanceConfig?: () => AcceptanceConfig | undefined;
	/** Durable checkpoint restored only for a native continuation/resume. */
	initialTaskState?: ExplicitTaskState;
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
		import("../runtime/tool-batch-controller.ts").ToolBatchControllerOptions["onToolIntent"]
	>;
	onToolResult?: NonNullable<
		import("../runtime/tool-batch-controller.ts").ToolBatchControllerOptions["onToolResult"]
	>;
	onPermissionDecision?: NonNullable<
		import("../runtime/tool-batch-controller.ts").ToolBatchControllerOptions["onPermissionDecision"]
	>;
	onToolCommit?: (toolCallId: string) => void | Promise<void>;
}

const DEFAULT_MAX_ITERATIONS = 30;

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
	const taskState = new TaskStateController(
		taskObjectiveFromMessages([...context.messages, ...prompts]),
		config.initialTaskState,
	);
	resetTaskStatus();
	const finish = async (outcome: {
		status: "completed" | "needs_input" | "blocked" | "failed" | "cancelled";
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	}): Promise<Message[]> => {
		if (outcome.status === "completed") taskState.markHandoff();
		if (outcome.status === "blocked" || outcome.status === "failed") {
			taskState.markBlocked(outcome.summary ?? outcome.status);
		}
		await emit({ type: "task_state_update", state: taskState.snapshot() });
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
	const executionPolicy = resolveExecutionPolicy(config.executionProfile);
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
	const guardEngine = config.guardEngine;
	let iteration = 0;
	// Armed when a restrict-severity guard decision injects a hypothesis
	// prompt; consumed after the next provider response to parse the model's
	// reply into stored hypotheses.
	let awaitingHypothesisParse = false;
	// Armed when the initial goal-decomposition prompt is injected at agent
	// start; consumed after the next provider response to parse the model's
	// reply into a subgoal breakdown.
	let awaitingGoalBreakdownParse = false;
	let consecutiveRunnerNudges = 0;
	let lastRunnerNudgeIteration = -1;
	let lastToolWorkIteration = -1;
	let performedToolWork = false;
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
			maxProviderCalls: maxIterations,
			maxToolCalls: maxIterations * 16,
			maxElapsedMs: 30 * 60_000,
			maxTokens: config.maxTotalTokens,
			...config.runBudget,
		},
		Date.now,
		config.durableBudgetState,
		consumption =>
			config.onBudgetConsumed?.(consumption.resource, consumption.amount),
	);

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
				v =>
					new Promise<
						Array<{
							command: string;
							result: "passed" | "failed";
							summary?: string;
						}>
					>(resolve => {
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
		).then(results => results.flat());
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

	// Typed extensions may augment the prompt just like native hooks.
	const extensionBeforeStart = config.extensionBus
		? await config.extensionBus.emit({
				type: "before_agent_start",
				prompt: prompts.map(p => p.content).join("\n"),
				systemPrompt:
					beforeAgentStartResult?.systemPrompt ?? context.systemPrompt ?? "",
			})
		: undefined;
	if (extensionBeforeStart?.messages) {
		beforeAgentStartResult = {
			...beforeAgentStartResult,
			messages: [
				...(beforeAgentStartResult?.messages ?? []),
				...extensionBeforeStart.messages,
			],
		};
	}
	if (extensionBeforeStart?.systemPrompt) {
		beforeAgentStartResult = {
			...beforeAgentStartResult,
			systemPrompt: extensionBeforeStart.systemPrompt,
		};
	}

	await emit({ type: "agent_start" });
	await emit({ type: "task_state_update", state: taskState.snapshot() });
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
			const providerBudget = runBudget.requestProviderCall();
			if (!providerBudget.allowed) {
				await intervene({
					kind: "budget",
					cause: "run_budget",
					detector: "run_budget",
					message: providerBudget.reason ?? "Run budget exhausted.",
					iteration,
					action: "pause",
					counters: {
						providerCalls: providerBudget.snapshot.providerCalls,
						toolCalls: providerBudget.snapshot.toolCalls,
						elapsedMs: providerBudget.snapshot.elapsedMs,
					},
				});
				return finish({
					status: "blocked",
					summary: providerBudget.reason,
					source: "runtime",
				});
			}
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
					taskState: taskState.snapshot(),
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
				// Trailing system context: OpenAIBackend re-roles it to `user`
				// for chat templates that require a system message at position 0.
				if (shouldProjectTaskState(taskState.snapshot())) {
					chatMessages.push({ role: "system", content: taskState.toContext() });
				}

				// Apply beforeProviderRequest hook
				const providerRequestHooks = [
					config.internalHooks?.beforeProviderRequest,
					config.hooks?.beforeProviderRequest,
				];
				let requestHeaders = config.streamOptions?.headers;
				let requestTimeoutMs =
					config.streamOptions?.timeoutMs ?? config.turnTimeoutMs;
				let requestMaxRetries =
					config.autoRetryEnabled === false
						? 0
						: (config.streamOptions?.maxRetries ?? config.maxRetries ?? 3);
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
						config.inferenceMode === "auto"
							? taskState.selectAdaptiveMode()
							: undefined;
					const effectiveMode = adaptiveDecision?.mode ?? config.inferenceMode;
					if (adaptiveDecision) {
						const selectionKey = `${adaptiveDecision.mode}:${adaptiveDecision.reason}`;
						if (selectionKey !== lastAdaptiveSelection) {
							lastAdaptiveSelection = selectionKey;
							await emit({
								type: "inference_mode_selected",
								configuredMode: "auto",
								effectiveMode: adaptiveDecision.mode,
								reason: adaptiveDecision.reason,
								phase: taskState.snapshot().phase,
							});
						}
					}
					const modeDef = effectiveMode
						? getInferenceMode(effectiveMode)
						: undefined;
					// When the mode asks for provider defaults, omit all sampling
					// params so the provider uses its own built-in defaults.
					const useProviderDefaults = modeDef?.useProviderDefaults ?? false;
					const modeParams = useProviderDefaults ? undefined : modeDef?.params;
					const effectiveTemp =
						modeParams?.temperature ?? config.temperature ?? 0.5;
					response = await config.backend.generate(chatMessages, {
						tools: registry.toToolDefinitions(),
						...(!useProviderDefaults && { temperature: effectiveTemp }),
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
						signal: config.signal,
						thinkingLevel: config.thinkingLevel,
						callbacks: {
							onDelta: delta =>
								queueProviderEvent({ type: "text_delta", turnId, delta }),
							onThinking: delta =>
								queueProviderEvent({ type: "thinking_delta", turnId, delta }),
							onTextStart: () =>
								queueProviderEvent({ type: "text_start", turnId }),
							onTextEnd: () => queueProviderEvent({ type: "text_end", turnId }),
							onToolCallStart: (toolCallId, toolName, args) =>
								queueProviderEvent({
									type: "tool_call_start",
									toolCallId,
									toolName,
									args,
								}),
							onToolCallDelta: (toolCallId, delta) =>
								queueProviderEvent({
									type: "tool_call_delta",
									toolCallId,
									delta,
								}),
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
						delayMs: guardResult.retryDelayMs ?? 0,
						error: guardResult.message ?? String(llmError),
					});
					await intervene({
						kind: "retry",
						cause: guardResult.action,
						detector: "provider_error_guard",
						message: guardResult.message ?? String(llmError),
						iteration,
						action: "recover",
						counters: { attempt: activeRetryAttempt },
						limits: { maxRetries: guardResult.maxRetries ?? 3 },
						nextAction:
							guardResult.action === "compact_then_retry"
								? "Compact context, then retry the provider request."
								: "Retry the provider request after backoff.",
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
							action: "continue",
							counters: {
								tokensBefore: compacted.tokensBefore,
								tokensAfter: compacted.tokensAfter,
							},
						});
					}

					if (guardResult.retryDelayMs && guardResult.retryDelayMs > 0) {
						const completed = await waitForRetryDelay(
							guardResult.retryDelayMs,
							config.signal,
						);
						if (!completed) {
							const steeringInterrupt = isSteeringInterrupt(config.signal);
							if (!steeringInterrupt) {
								await emit({ type: "error", message: "Operation aborted" });
							}
							return finish({
								status: "cancelled",
								summary: steeringInterrupt
									? STEERING_INTERRUPT_SUMMARY
									: "Operation aborted during provider retry.",
								source: "runtime",
							});
						}
					}
				}
			}

			const tokenBudget = runBudget.recordTokens(
				response?.usage?.totalTokens ?? 0,
			);
			if (!tokenBudget.allowed) {
				await intervene({
					kind: "budget",
					cause: "run_budget",
					detector: "run_budget",
					message: tokenBudget.reason ?? "Token budget exhausted.",
					iteration,
					action: "pause",
					counters: {
						providerCalls: tokenBudget.snapshot.providerCalls,
						toolCalls: tokenBudget.snapshot.toolCalls,
						elapsedMs: tokenBudget.snapshot.elapsedMs,
					},
				});
				return finish({
					status: "blocked",
					summary: tokenBudget.reason,
					source: "runtime",
				});
			}
			let toolCalls = response?.toolCalls ?? [];
			let assistantContent = response?.content ?? "";
			// Fallback: when LLM emits tool calls as text instead of structured
			// tool_calls array, extract them from the response content.
			if (toolCalls.length === 0 && response?.content) {
				const textCalls = parseTextToolCalls(response.content, name =>
					registry.has(name),
				);
				if (textCalls.length > 0) {
					toolCalls = textCalls;
					assistantContent = stripTextToolCalls(response.content);
				}
			}
			if (toolCalls.some(call => call.name !== "task_status")) {
				performedToolWork = true;
				lastToolWorkIteration = iteration;
			}
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
					content: assistantContent,
					toolCallCount: toolCalls.length,
					stopReason,
					usageTokens: response?.usage?.totalTokens,
					iteration,
				});
			}

			// Output guard: check for empty/degenerate responses
			if (outputGuard) {
				const guardCheck = outputGuard.checkResponse(
					assistantContent || null,
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

			// Persist sanitized arguments (invalid JSON replaced with "{}") so a
			// call truncated by the output token limit never poisons history with
			// a tool_call the backend can never re-parse on a later turn. The
			// *execution* path below still uses the original `toolCalls`, whose
			// ids the executor's own truncation handling (tool-batch-controller's
			// "length" branch) depends on for tool_call/tool_result pairing.
			const assistant = createAssistantMessage(
				assistantContent,
				sanitizeToolCallArguments(toolCalls),
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
			const toolBudget = runBudget.requestToolBatch(toolCalls.length);
			if (!toolBudget.allowed) {
				await intervene({
					kind: "budget",
					cause: "run_budget",
					detector: "run_budget",
					message: toolBudget.reason ?? "Run budget exhausted.",
					iteration,
					action: "pause",
					counters: {
						providerCalls: toolBudget.snapshot.providerCalls,
						toolCalls: toolBudget.snapshot.toolCalls,
						elapsedMs: toolBudget.snapshot.elapsedMs,
					},
				});
				return finish({
					status: "blocked",
					summary: toolBudget.reason,
					source: "runtime",
				});
			}
			const batch = await executeToolBatch({
				registry,
				toolCalls,
				rawStopReason,
				toolExecution: config.toolExecution,
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
				emitExtension: event => emitTyped(config.extensionBus, event),
			});
			const toolResults = batch.messages;
			const toolTerminated = batch.terminated;
			for (const toolResult of toolResults) {
				messages.push(toolResult);
				newMessages.push(toolResult);
				await emitMessagePair(emit, turnId, toolResult);
				hasMoreToolCalls = true;
			}
			if (toolCalls.length > 0) {
				taskState.recordToolBatch(toolCalls, toolResults, iteration);
				await emit({ type: "task_state_update", state: taskState.snapshot() });
			}
			for (const toolCallId of batch.executedToolCallIds)
				await config.onToolCommit?.(toolCallId);

			// Turn-level loop detection. Runs after every turn when guard
			// engine is available; otherwise falls back to output guard.
			if (
				executionPolicy.embeddedPoliciesEnabled &&
				(guardEngine || outputGuard) &&
				toolCalls.length > 0
			) {
				const turnToolCalls = toolCalls.map((call, index) => ({
					name: call.name,
					args: call.arguments,
					result: String(toolResults[index]?.content ?? ""),
				}));


			}

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
							action: "continue",
							counters: {
								tokensBefore: compacted.tokensBefore,
								tokensAfter: compacted.tokensAfter,
							},
						});
					}
				}
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
							action: "continue",
						});
					}
					pendingMessages = followUpsOnTerminate;
					hasMoreToolCalls = false;
					// Re-enter inner loop with follow-up messages
					continue;
				}
				return finish(
					resolveCompletionGate({
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
					resolveCompletionGate({
						declared: getTaskStatus(),
						structuredOutcomeRequired:
							performedToolWork && registry.has("task_status"),
					}),
				);
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
				const nudgeContent = `${nudgeTag} Do not stop yet without a structured conclusion. Verify that every requested step is complete. ` +
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
					action: "pause",
					counters: { consecutiveRunnerNudges },
					limits: { maxConsecutiveNudges: MAX_CONSECUTIVE_RUNNER_NUDGES },
					nextAction: "Resume from the current checkpoint with user guidance.",
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
			const verificationResults = await runAcceptanceVerification(
				resolved,
				config.signal,
			);
			const report = parsedReportOrReview(
				lastAssistantContent(newMessages),
				resolved,
				verificationResults,
				config,
				emit,
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
			v =>
				`[${v.result.toUpperCase()}] ${v.command}${v.summary ? ` → ${v.summary.slice(0, 100)}` : ""}`,
		);

		// Validate criteria against report
		const report = parsed.report;
		const criteriaResults = resolved.criteria.map(c => {
			const satisfied = report?.criteriaSatisfied?.some(
				cs =>
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
				cs => cs.status === "satisfied",
			)
				? "passed"
				: "failed";
		}

		const allCriteriaPass = criteriaResults.every(
			c => c.status === "satisfied",
		);
		const allVerificationsPass = verificationResults.every(
			v => v.result === "passed",
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
			resolveCompletionGate({
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
			iteration >= maxIterations ||
			reflectionFailed ||
			false
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
