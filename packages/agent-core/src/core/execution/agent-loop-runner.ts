// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.

import { compactToFit } from "../compaction/engine.ts";
import {
	emitConclusion,
	lastAssistantContent,
} from "../policy/conclusion-policy.ts";
import { executeToolBatch } from "./tool-batch-controller.ts";
import { ToolRegistry } from "../../infrastructure/tools/registry.ts";
import {
	parseTextToolCalls,
	stripTextToolCalls,
} from "../../infrastructure/tools/utils/text-to-tool-calls.ts";
import {
	type AcceptanceConfig,
	evaluateAcceptanceReport,
	formatAcceptancePrompt,
	type ResolvedAcceptance,
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	verifyAcceptanceCommands,
} from "../../infrastructure/guards/acceptance-contract.ts";
import type { OutputGuard } from "../../infrastructure/guards/output-guard.ts";
import {
	assistantText,
	emitMessagePair,
	stopReasonFor,
	withSystemPrompt,
} from "../loop/callbacks.ts";
import { processProviderResponse } from "../loop/provider-response.ts";
import {
	createProviderTurnState,
	requestAssistantTurn,
} from "../loop/provider-turn.ts";
// ═══════════════════════════════════════════════════════════════════════════
// Task-aware callbacks — injected by the harness when @logician/agent-blocks is loaded.
// When omitted, the loop runs in "pure" mode: no task nudges, no structured
// outcome resolution, no continuation.  This keeps agent-core minimal like pi.
// ═══════════════════════════════════════════════════════════════════════════

export interface TaskAwareCallbacks {
	/** Current structured task status, or null if none declared. */
	getTaskStatus: () =>
		| { status: string; summary: string; ts: number }
		| null
		| undefined;
	/** Reset the task status at each run start. */
	resetTaskStatus: () => void;
	/** Resolve tool-termination outcome using declared task status. */
	resolveOutcome?: (ctx: {
		declared:
			| { status: string; summary: string; ts: number }
			| null
			| undefined;
		structuredOutcomeRequired: boolean;
		fallbackStatus?: string;
		fallbackSummary?: string;
	}) => {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	};
}

// Continuation callback — injected by the harness when @logician/agent-blocks is loaded.
// When omitted, the loop finishes immediately when no more tool calls are produced.
export interface ContinuationCallback {
	(state: ContinuationState): Promise<ContinuationDecision>;
}

export interface ContinuationState {
	consecutiveRunnerNudges: number;
	lastRunnerNudgeIteration: number;
	acceptanceReported: boolean;
	acceptanceFailed: boolean;
	acceptanceFinalizationTurns: number;
	reflectionCount: number;
	reflectionFailed: boolean;
}

export interface ContinuationDecision {
	action: "continue" | "finish";
	pendingMessages?: Message[];
	outcome?: {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	};
}

// ═══════════════════════════════════════════════════════════════════════════
// Inline utilities — previously imported from tasks/adaptive-mode.ts
// ═══════════════════════════════════════════════════════════════════════════

/** Extract the last meaningful user prompt from messages. */
function taskObjectiveFromMessages(
	messages: Array<{ role: string; content: unknown }>,
): string {
	const prompts = messages
		.filter(
			message => message.role === "user" && typeof message.content === "string",
		)
		.map(message =>
			String(message.content).replace(/\s+/g, " ").trim().slice(0, 1000),
		)
		.filter(Boolean);
	const lastMeaningful = prompts.find(
		p =>
			!/^(?:continue|resume|go on|keep going)[.! ]*$/i.test(p) &&
			!/^\[continuation-nudge:/i.test(p),
	);
	return lastMeaningful ?? prompts.at(-1) ?? "";
}

const FAILURE_PATTERN = /(?:error|failure|exception|unsuccessful|could not)/i;
function isToolFailureResult(result: string): boolean {
	return FAILURE_PATTERN.test(result);
}

// ═══════════════════════════════════════════════════════════════════════════
// Inline outcome resolver — previously imported from tasks/outcome-resolution.ts
// ═══════════════════════════════════════════════════════════════════════════

export function resolveOutcomeDefault(ctx: {
	declared: { status: string; summary: string; ts: number } | null | undefined;
	structuredOutcomeRequired: boolean;
	fallbackStatus?: string;
	fallbackSummary?: string;
}): {
	status: RunOutcomeStatus;
	summary?: string;
	source: "structured" | "heuristic" | "runtime";
} {
	if (ctx.declared) {
		return {
			status: (ctx.declared.status === "done"
				? "completed"
				: ctx.declared.status) as RunOutcomeStatus,
			summary: ctx.declared.summary,
			source: "structured",
		};
	}
	if (ctx.structuredOutcomeRequired) {
		return {
			status: "completed",
			summary:
				"Run completed without a declared task outcome. Tool work was performed but no structured outcome was recorded. Review the final text for correctness.",
			source: "runtime",
		};
	}
	return {
		status: (ctx.fallbackStatus ?? "completed") as RunOutcomeStatus,
		summary: ctx.fallbackSummary,
		source: "heuristic",
	};
}
import type {
	AgentConfig,
	AgentEventSink,
	AgentMessage,
	CompactableMessage,
	Message,
	Tool,
	ToolCall,
} from "../types/index.ts";
import { resolveAgentSettings } from "../configuration/agent-settings.ts";
import type { LLMBackend } from "../provider/backend.ts";
import {
	type RunOutcomeStatus,
	resolveExecutionPolicy,
} from "../policy/execution-policy.ts";
import { checkBudget } from "../policy/exit-path.ts";
import {
	HarnessInterventionController,
	type InterventionInput,
} from "../policy/intervention-controller.ts";
import {
	createSystemMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "../provider/messages.ts";
import {
	RunBudgetController,
	type RunBudgetDecision,
} from "../policy/run-budget.ts";
import { ToolResultCache } from "../state/tool-cache.ts";

// A steering interrupt cancels the in-flight provider call to redirect the
// run, not to stop it — the harness auto-continues with the queued steering
// text right after. Matched by exact summary text so both the loop runner
// (which produces it) and the harness (which decides whether to resume as a
// plain turn vs. an autonomous continuation) agree on what counts as one.
export const STEERING_INTERRUPT_SUMMARY =
	"Current provider response interrupted to apply steering.";

const STEERING_INTERRUPT_NAME = "SteeringInterruptError";

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

export interface RunAgentLoopConfig extends AgentConfig {
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

	/** Escalation state persists here across turns; defaults to a fresh, single-turn controller. */
	interventionController?: HarnessInterventionController;
	/** Task-spanning counters carried over from an earlier turn in this session. */
	durableBudgetState?: {
		providerCalls: number;
		toolCalls: number;
		tokens: number;
		startedAt?: number;
	};
	/** Notified as budget is consumed, so the harness can carry counters into the next turn. */
	onBudgetConsumed?: (
		resource: "provider_call" | "tool_call" | "token",
		amount: number,
	) => void;

	// ── Task-aware callbacks (from @logician/agent-blocks) ────────────────────────
	/** Injected by the harness when blocks are loaded. Provides task status, reset,
	 * and outcome resolution. When omitted, the loop runs in pure mode. */
	callbacks?: TaskAwareCallbacks;

	// ── Continuation callback (from @logician/agent-blocks) ────────────────────────
	/** Custom continuation decision. When omitted, the loop finishes when no more
	 * tool calls are produced (no nudges, no reflection, no acceptance retry). */
	decideContinuation?: ContinuationCallback;
}

async function runAgentLoopInternal(
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
	config.callbacks?.resetTaskStatus?.();
	const finish = async (outcome: {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	}): Promise<Message[]> => {
		await emit({ type: "agent_end", messages: newMessages, status: outcome.status, summary: outcome.summary });
		return newMessages;
	};
	let settings = resolveAgentSettings(config);
	const maxIterations = settings.maxIterations;
	const executionPolicy = resolveExecutionPolicy(settings.executionProfile);
	const interventionController =
		config.interventionController ?? new HarnessInterventionController();
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
	let lastToolWorkIteration = -1;
	let performedToolWork = false;
	let toolFailures = 0;
	const adaptiveObjective = taskObjectiveFromMessages([
		...context.messages,
		...prompts,
	]);
	let contextWasCompacted = false;
	const continuationState: ContinuationState = {
		consecutiveRunnerNudges: 0,
		lastRunnerNudgeIteration: -1,
		acceptanceReported: false,
		acceptanceFailed: false,
		acceptanceFinalizationTurns: 0,
		reflectionCount: 0,
		reflectionFailed: false,
	};
	const providerTurnState = createProviderTurnState();
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

	async function drainSteering(): Promise<Message[]> {
		return (
			(await config.hooks?.getSteeringMessages?.({ messages, iteration })) ?? []
		);
	}

	async function drainFollowUps(): Promise<Message[]> {
		return (
			(await config.hooks?.getFollowUpMessages?.({
				messages,
				iteration,
				assistantText: assistantText(newMessages.at(-1)),
				stopReason: "stop",
			})) ?? []
		);
	}

	let pendingMessages = await drainSteering();

	// Apply beforeAgentStart hook
	const beforeAgentStartResult = await config.hooks?.beforeAgentStart?.({
		prompt: prompts.map(p => p.content).join("\n"),
		systemPrompt: context.systemPrompt ?? "",
		messages: messages as AgentMessage[],
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

			const transformResult = await config.hooks?.transformContext?.({
				messages: messages as AgentMessage[],
				iteration,
				signal: config.signal,
			});
			const transformed = transformResult?.messages;
			if (transformed) {
				messages = transformed as Message[];
				if (contextWasCompacted) config.onContextCompacted?.(messages);
			}

			const turnResult = await requestAssistantTurn({
				state: providerTurnState,
				messages,
				config,
				settings,
				registry,
				outputGuard,
				turnId,
				iteration,
				adaptiveObjective,
				performedToolWork,
				toolFailures,
				contextWasCompacted,
				convertToLlm: config.convertToLlm ?? defaultConvertToLlm,
				emit,
				intervene,
				isSteeringInterrupt,
				steeringInterruptSummary: STEERING_INTERRUPT_SUMMARY,
			});
			if (turnResult.kind === "finish") {
				return finish(turnResult.outcome);
			}
			const response = turnResult.response;
			messages = turnResult.messages;
			contextWasCompacted = turnResult.contextWasCompacted;

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
				hooks: config.hooks,
				permissions: config.permissions,
				onPermissionRequest: config.onPermissionRequest,
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

			const prepareResult = await config.hooks?.prepareNextTurn?.({
				messages,
				iteration,
				hadToolCalls: toolCalls.length > 0,
			});
			const prepared = prepareResult?.messages;
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
					(config.callbacks?.resolveOutcome ?? resolveOutcomeDefault)({
						declared: config.callbacks?.getTaskStatus?.(),
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
					? ((await config.hooks?.shouldStopAfterTurn?.({
							messages,
							iteration,
							hadToolCalls: false,
						})) ?? false)
					: false;
			// Acceptance stop rules take priority
			let acceptanceStop = false;
			if (!stop && shouldRunAcceptanceFinalization(resolved)) {
				acceptanceStop = checkStopRules(resolved);
			}
			if (stop || acceptanceStop) {
				return finish(
					(config.callbacks?.resolveOutcome ?? resolveOutcomeDefault)({
						declared: config.callbacks?.getTaskStatus?.(),
						structuredOutcomeRequired:
							performedToolWork && registry.has("task_status"),
					}),
				);
			}

			pendingMessages = await drainSteering();
		}

		// Agent would stop here. Decide whether it actually should (pi-style
		// outer loop). When @logician/agent-blocks is loaded (decideContinuation provided),
		// the callback handles: follow-up draining, nudges, structured outcome
		// resolution, acceptance retry, and reflection. Otherwise the loop finishes.
		if (config.decideContinuation) {
			const continuationDecision =
				await config.decideContinuation(continuationState);
			if (continuationDecision.action === "continue") {
				pendingMessages = continuationDecision.pendingMessages ?? [];
				continue;
			}
			if (continuationDecision.outcome) {
				return finish(continuationDecision.outcome);
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
	if (
		shouldRunAcceptanceFinalization(resolved) &&
		!continuationState.acceptanceReported
	) {
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

		continuationState.acceptanceReported = true;
		await emit({
			type: "acceptance_complete",
			status: report.status,
			report: report.ledger as unknown as Record<string, unknown>,
		});

		if (report.status === "failed") {
			continuationState.acceptanceFailed = true;
		}
	}

	// Final output guard reset when agent ends
	outputGuard?.reset();
	// Acceptance failure must take precedence over a model-declared `done`.
	if (continuationState.acceptanceFailed) {
		return finish({
			status: "failed",
			summary:
				"Acceptance contract not satisfied after the configured finalization turns.",
			source: "runtime",
		});
	}
	const declared = config.callbacks?.getTaskStatus?.();
	if (
		(declared || (performedToolWork && registry.has("task_status"))) &&
		executionPolicy.embeddedPoliciesEnabled
	) {
		return finish(
			(config.callbacks?.resolveOutcome ?? resolveOutcomeDefault)({
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
			iteration >= maxIterations || continuationState.reflectionFailed
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
	return runAgentLoopInternal(context, prompts, config, emit);
}
