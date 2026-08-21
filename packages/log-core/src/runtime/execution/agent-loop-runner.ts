// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.

import { compactToFit } from "../compaction/engine.ts";
import { resolveAgentSettings } from "../../control/configuration/agent-settings.ts";
import {
	evaluateAcceptanceReport,
	formatAcceptancePrompt,
	formatVerificationRepair,
	type ResolvedAcceptance,
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	verifyAcceptanceCommands,
} from "../../control/guards/acceptance-contract.ts";
import {
	isToolFailureResult,
	taskObjectiveFromMessages,
} from "../loop/adaptive-mode.ts";
import {
	assistantText,
	emitMessagePair,
	lastAssistantContent,
	stopReasonFor,
	withSystemPrompt,
} from "../loop/callbacks.ts";
import type { AgentLoopConfig } from "../loop/config.ts";
import { processProviderResponse } from "../loop/provider-response.ts";
import {
	createProviderTurnState,
	requestAssistantTurn,
} from "../loop/provider-turn.ts";
import {
	type RunOutcomeStatus,
	resolveExecutionPolicy,
} from "../../control/policy/execution-policy.ts";
import { checkBudget } from "../../control/policy/exit-path.ts";
import {
	HarnessInterventionController,
	type InterventionInput,
} from "../../control/policy/intervention-controller.ts";
import {
	RunBudgetController,
	type RunBudgetDecision,
} from "../../control/policy/run-budget.ts";
import { AgentRunController } from "../../control/policy/run-controller.ts";
import {
	createSystemMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "../../capabilities/provider/messages.ts";
import { ToolResultCache } from "../state/tool-cache.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";
import type {
	AgentEventSink,
	AgentMessage,
	CompactableMessage,
	Message,
	Tool,
	ToolCall,
} from "../../system/types/types-messages.ts";
import { executeToolBatch } from "./tool-batch-controller.ts";

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

export type RunAgentLoopConfig = AgentLoopConfig;

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
	const finish = async (outcome: {
		status: RunOutcomeStatus;
		summary?: string;
		source: "structured" | "heuristic" | "runtime";
	}): Promise<Message[]> => {
		await emit({
			type: "agent_end",
			messages: newMessages,
			status: outcome.status,
			summary: outcome.summary,
		});
		return newMessages;
	};
	let settings = resolveAgentSettings(config);
	const maxIterations = settings.maxIterations;
	const executionPolicy = resolveExecutionPolicy(settings.executionProfile);
	const interventionController =
		config.interventionController ?? new HarnessInterventionController();
	const runController = config.runController ?? new AgentRunController();
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
	let performedToolWork = false;
	let toolFailures = 0;
	const adaptiveObjective = taskObjectiveFromMessages([
		...context.messages,
		...prompts,
	]);
	let contextWasCompacted = false;
	let acceptanceReported = false;
	let acceptanceFailed = false;
	let cachedVerificationResults:
		| Awaited<ReturnType<typeof verifyAcceptanceCommands>>
		| undefined;
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
			let _assistantContent: string;
			if (processResult.success) {
				toolCalls = processResult.toolCalls;
				_assistantContent = processResult.assistantContent;
				assistant = processResult.assistant;
				if (toolCalls.length > 0) {
					performedToolWork = true;
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
			const permissionEscalation = runController.recordPermissionBatch({
				denials: batch.permissionDenials,
				executed: batch.executedToolCallIds.length,
			});
			for (const toolResult of toolResults) {
				if (isToolFailureResult(String(toolResult.content ?? ""))) {
					toolFailures++;
				}
				messages.push(toolResult);
				newMessages.push(toolResult);
				await emitMessagePair(emit, turnId, toolResult);
				hasMoreToolCalls = true;
			}
			if (permissionEscalation) {
				await intervene({
					kind: "loop",
					cause: "permission_denials",
					detector: "permission_escalation",
					message:
						"Autonomous execution paused after repeated permission denials. User authorization or a different task scope is required.",
					iteration,
					action: "pause",
					counters: {
						consecutive: permissionEscalation.consecutive,
						total: permissionEscalation.total,
					},
					limits: { consecutive: 3, total: 20 },
				});
				return finish({
					status: "needs_input",
					summary:
						"Repeated permission denials require user authorization or a safer scope.",
					source: "runtime",
				});
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
				return finish({ status: "completed", source: "runtime" });
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
				if (stop) return finish({ status: "completed", source: "runtime" });
				runController.requestAcceptanceStop();
				break;
			}

			pendingMessages = await drainSteering();
		}

		pendingMessages = runController.acceptanceStopRequested
			? []
			: await drainFollowUps();
		if (pendingMessages.length > 0) continue;

		// Deterministic verification gets one bounded repair turn. This happens
		// only after the ordinary autonomous policy considers the work finished.
		if (resolved.verify.length > 0) {
			await emit({
				type: "acceptance_start",
				level: resolved.level,
				criteriaCount: resolved.criteria.length,
			});
			cachedVerificationResults = await verifyAcceptanceCommands(resolved, {
				cwd: config.cwd,
				signal: config.signal,
			});
			for (const result of cachedVerificationResults) {
				await emit({
					type: "acceptance_verify",
					command: result.command,
					result: result.result,
					summary: result.summary,
				});
			}
			if (
				runController.requestVerificationRepair(
					cachedVerificationResults,
					iteration < maxIterations,
				)
			) {
				const content = formatVerificationRepair(cachedVerificationResults);
				await intervene({
					kind: "verification",
					cause: "verification_failed",
					detector: "acceptance_verifier",
					message: content,
					iteration,
					action: "recover",
					limits: { repairAttempts: 1 },
				});
				pendingMessages = [{ role: "user", content, timestamp: Date.now() }];
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

	// ── Acceptance finalization ────────────────────────────────────────
	if (shouldRunAcceptanceFinalization(resolved) && !acceptanceReported) {
		const finalText = lastAssistantContent(finalMessagesForConclusion);

		// Run verification commands
		const verificationResults =
			cachedVerificationResults ??
			(await verifyAcceptanceCommands(resolved, {
				cwd: config.cwd,
				signal: config.signal,
			}));

		// Validate criteria and build ledger
		const report = evaluateAcceptanceReport(
			finalText,
			resolved,
			verificationResults,
		);
		for (const criterion of resolved.criteria) {
			const result = report.ledger.report?.criteriaSatisfied.find(
				item => item.id === criterion.id,
			);
			await emit({
				type: "acceptance_check",
				criterionId: criterion.id,
				status: result?.status ?? "failed",
				severity: criterion.severity ?? "required",
			});
		}

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
	if (config.signal?.aborted) {
		return finish({
			status: "cancelled",
			summary: isSteeringInterrupt(config.signal)
				? STEERING_INTERRUPT_SUMMARY
				: "Operation aborted.",
			source: "runtime",
		});
	}

	// Preserve the finalized transcript returned by the loop.
	newMessages.splice(0, newMessages.length, ...finalMessagesForConclusion);

	const finalText = lastAssistantContent(finalMessagesForConclusion);
	return finish({
		status: iteration >= maxIterations ? "failed" : "completed",
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
