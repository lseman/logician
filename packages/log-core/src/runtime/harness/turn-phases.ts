// ── Turn phases for the agent loop ──────────────────────────────────────────
// The loop body of runAgentLoopInternal is a sequence of per-turn phases over
// a shared TurnContext record. Each phase owns one concern (budget, steering,
// soft tools, context transform, provider call, response handling, loop
// detection, tool batch, permission escalation, context budget, config
// update, stop policy, acceptance) and returns a PhaseOutcome that the loop
// skeleton (agent-harness.ts) interprets:
//   continue — proceed to the next phase (or re-check the loop when the
//              sequence completes)
//   reenter  — skip the rest of the sequence and re-check the loop condition
//              (the original loop body's `continue`)
//   break    — exit the loop level (the original loop body's `break`)
//   finish   — terminate the run with the given outcome
// Phases mutate the context in place; only the skeleton decides control flow.
import type { LLMResponse } from "../../capabilities/provider/backend.ts";
import type { resolveTokenEncoding } from "../../capabilities/provider/messages.ts";
import {
	createSystemMessage,
	convertToLlm as defaultConvertToLlm,
	estimateChatPayloadTokens,
} from "../../capabilities/provider/messages.ts";
import { ToolRegistry } from "../../capabilities/tools/registry.ts";
import type { ToolResultCache } from "../../capabilities/tools/tool-result-cache.ts";
import type { AgentSettings } from "../../control/configuration/agent-settings.ts";
import { resolveAgentSettings } from "../../control/configuration/agent-settings.ts";
import type { ResolvedAcceptance } from "../../control/guards/acceptance-contract.ts";
import {
	formatVerificationRepair,
	resolveEffectiveAcceptance,
	shouldRunAcceptanceFinalization,
	verifyAcceptanceCommands,
} from "../../control/guards/acceptance-contract.ts";
import type { OutputGuard } from "../../control/guards/output-guard.ts";
import type { SoftToolRequirementManager } from "../../control/guards/soft-tool-requirement.ts";
import {
	MAX_ESCALATIONS,
	SoftToolRequirementExceededError,
} from "../../control/guards/soft-tool-requirement.ts";
import type { TextLoopDetector } from "../../control/guards/text-loop-detector.ts";
import type { ResolvedExecutionPolicy } from "../../control/policy/execution-policy.ts";
import { evaluateStopPolicies } from "../../control/policy/execution-policy.ts";
import { checkBudget } from "../../control/policy/exit-path.ts";
import type {
	HarnessInterventionController,
	InterventionInput,
} from "../../control/policy/intervention-controller.ts";
import type { RunBudgetController } from "../../control/policy/run-budget.ts";
import type { AgentRunController } from "../../control/policy/run-controller.ts";
import { createVerifiedStopPolicy } from "../../control/policy/verified-stop-policy.ts";
import type { RunOutcomeStatus } from "../../system/types/execution-policy.ts";
import type { RunBudgetDecision } from "../../system/types/run-budget.ts";
import type {
	AgentEventSink,
	AgentMessage,
	CompactableMessage,
	Message,
	Tool,
	ToolCall,
} from "../../system/types/types-messages.ts";
import { compactToFit, toMessages } from "../compaction/engine.ts";
import type { ToolBatchResult } from "../execution/tool-batch-controller.ts";
import { executeToolBatch } from "../execution/tool-batch-controller.ts";
import { isToolFailureResult } from "../loop/adaptive-mode.ts";
import {
	assistantText,
	emitMessagePair,
	lastAssistantContent,
	stopReasonFor,
} from "../loop/callbacks.ts";
import type { PrefixDivergence } from "../loop/prefix-stability.ts";
import { processProviderResponse } from "../loop/provider-response.ts";
import type { ProviderTurnState } from "../loop/provider-turn.ts";
import { requestAssistantTurn } from "../loop/provider-turn.ts";
import type {
	RunAgentLoopConfig,
	RunAgentLoopContext,
} from "./agent-harness.ts";
import { resolveModelContextWindow } from "./live/model.ts";

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

/**
 * Strategy-change nudge injected on the first text-stagnation hit. The run
 * gets one bounded recovery turn; a second hit blocks (see the phases below).
 */
function formatStagnationNudge(hit: string): string {
	return (
		"[stagnation-nudge] The harness detected that your last response repeats " +
		`prior content without new progress (${hit}). Do not restate the same ` +
		"analysis. Change strategy and take a concrete next step: run a tool, " +
		"make an edit, or state a decision and proceed. If the task is already " +
		"complete, say so in one explicit sentence."
	);
}

// ── Shared record ────────────────────────────────────────────────────────────

/** Per-turn derived state, rebuilt by beginTurn. */
export interface TurnState {
	turnId: string;
	/** Provider response for this turn (set by requestProviderTurn). */
	response?: LLMResponse;
	/** How this request's payload prefix compared to the previous one. */
	prefixStability?: PrefixDivergence | undefined;
	/** Assistant message recorded by processAssistantResponse. */
	assistant?: Message;
	/** Tool calls parsed from the response. */
	toolCalls?: ToolCall[];
	/** Executed tool batch result. */
	batch?: ToolBatchResult;
	/** Effective stop reason for the turn (set by executeToolCalls). */
	stopReason?: ReturnType<typeof stopReasonFor>;
}

/**
 * Shared record for the agent loop's turn phases: run-level wiring, the
 * canonical conversation, loop counters, and per-turn derived state. Phases
 * read and mutate it in place.
 */
export interface TurnContext {
	// Wiring — stable for the run. The `config`/`context` objects may be
	// mutated mid-run (refreshNextTurnConfig, beforeAgentStart); that is
	// preserved from the original loop.
	readonly context: RunAgentLoopContext;
	readonly config: RunAgentLoopConfig;
	readonly emit: AgentEventSink;
	readonly interventionController: HarnessInterventionController;
	readonly runController: AgentRunController;
	readonly runBudget: RunBudgetController;
	readonly textLoopDetector: TextLoopDetector;
	readonly softToolManager: SoftToolRequirementManager;
	readonly providerTurnState: ProviderTurnState;
	readonly cache: ToolResultCache;
	readonly outputGuard: OutputGuard | null | undefined;
	readonly tokenEncoding: ReturnType<typeof resolveTokenEncoding>;
	readonly adaptiveObjective: string;
	readonly maxIterations: number;
	readonly executionPolicy: ResolvedExecutionPolicy;

	// Canonical conversation (phases push to / rebind these).
	messages: Message[];
	newMessages: Message[];

	// Loop counters.
	iteration: number;
	pendingMessages: Message[];
	hasMoreToolCalls: boolean;

	// Re-bound mid-run (config refresh).
	settings: AgentSettings;
	registry: ToolRegistry;

	// Run counters.
	performedToolWork: boolean;
	toolFailures: number;
	stagnationNudges: number;
	contextWasCompacted: boolean;
	acceptanceFailed: boolean;
	cachedVerificationResults:
		| Awaited<ReturnType<typeof verifyAcceptanceCommands>>
		| undefined;

	// Acceptance contract (lazy-resolution cache + this run's resolution).
	resolvedAcceptance: ResolvedAcceptance | null;
	resolved: ResolvedAcceptance;

	// Per-turn derived state (rebuilt by beginTurn).
	turn: TurnState;
}

/** Terminal outcome shared by every `finish` path and `finishRun`. */
export interface RunFinishOutcome {
	status: RunOutcomeStatus;
	summary?: string | undefined;
	source: "structured" | "heuristic" | "runtime";
}

export type PhaseOutcome =
	| { kind: "continue" }
	| { kind: "reenter" }
	| { kind: "break" }
	| ({ kind: "finish" } & RunFinishOutcome);

const CONTINUE: PhaseOutcome = { kind: "continue" };
const REENTER: PhaseOutcome = { kind: "reenter" };
const BREAK: PhaseOutcome = { kind: "break" };

/** One phase of a sequence. */
export type LoopPhase = (ctx: TurnContext) => Promise<PhaseOutcome>;

// ── Shared helpers ───────────────────────────────────────────────────────────

/** Emit a harness intervention for the shared context. */
export function intervene(
	ctx: TurnContext,
	input: InterventionInput,
): Promise<void> | void {
	return ctx.emit({
		type: "harness_intervention",
		...ctx.interventionController.record(input),
	});
}

/** Drain queued steering messages (host hook). */
export async function drainSteering(ctx: TurnContext): Promise<Message[]> {
	return (
		(await ctx.config.hooks?.getSteeringMessages?.({
			messages: ctx.messages,
			iteration: ctx.iteration,
		})) ?? []
	);
}

/** Drain queued follow-up messages (host hook). */
export async function drainFollowUps(ctx: TurnContext): Promise<Message[]> {
	return (
		(await ctx.config.hooks?.getFollowUpMessages?.({
			messages: ctx.messages,
			iteration: ctx.iteration,
			assistantText: assistantText(ctx.newMessages.at(-1)),
			stopReason: "stop",
		})) ?? []
	);
}

/** Lazy-resolve the run's acceptance contract. */
export function resolveAcceptance(ctx: TurnContext): ResolvedAcceptance {
	if (!ctx.resolvedAcceptance) {
		const raw = ctx.config.getAcceptanceConfig?.() ?? ctx.config.acceptance;
		ctx.resolvedAcceptance = resolveEffectiveAcceptance({ explicit: raw });
	}
	return ctx.resolvedAcceptance;
}

/** True when the last assistant content matches a configured stop rule. */
export function checkStopRules(ctx: TurnContext): boolean {
	const resolved = ctx.resolved;
	if (!resolved.stopRules?.length) return false;
	const text = lastAssistantContent(ctx.newMessages);
	for (const rule of resolved.stopRules) {
		if (text.includes(rule)) return true;
	}
	return false;
}

/** Emit the agent_end event and return the finalized transcript. */
export async function finishRun(
	ctx: TurnContext,
	outcome: RunFinishOutcome,
): Promise<Message[]> {
	await ctx.emit({
		type: "agent_end",
		messages: ctx.newMessages,
		status: outcome.status,
		summary: outcome.summary,
		stepCount: ctx.iteration,
	});
	return ctx.newMessages;
}

/** Intervention shared by every budget-exhaustion finish. */
async function recordBudgetExhaustion(
	ctx: TurnContext,
	decision: RunBudgetDecision,
): Promise<void> {
	await intervene(ctx, {
		kind: "budget",
		cause: "run_budget",
		detector: "run_budget",
		message: decision.reason ?? "Run budget exhausted.",
		iteration: ctx.iteration,
		counters: {
			providerCalls: decision.snapshot.providerCalls,
			toolCalls: decision.snapshot.toolCalls,
			elapsedMs: decision.snapshot.elapsedMs,
		},
	});
}

function budgetExhaustedOutcome(
	decision: RunBudgetDecision,
): Extract<PhaseOutcome, { kind: "finish" }> {
	return {
		kind: "finish",
		status: "blocked",
		summary: decision.reason,
		source: "runtime",
	};
}

/** Build (or rebuild after a mid-run refresh) the tool registry. */
export function createToolRegistry(
	context: RunAgentLoopContext,
	config: RunAgentLoopConfig,
	cache: ToolResultCache,
	tools: Tool[],
): ToolRegistry {
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
}

// ── Outer-loop phases ────────────────────────────────────────────────────────

/** Abort gate: a cancelled signal ends the run before any provider work. */
export async function checkRunAbort(ctx: TurnContext): Promise<PhaseOutcome> {
	if (ctx.config.signal?.aborted) {
		const steeringInterrupt = isSteeringInterrupt(ctx.config.signal);
		if (!steeringInterrupt) {
			await ctx.emit({ type: "error", message: "Operation aborted" });
		}
		return {
			kind: "finish",
			status: "cancelled",
			summary: steeringInterrupt
				? STEERING_INTERRUPT_SUMMARY
				: "Operation aborted before the provider request.",
			source: "runtime",
		};
	}
	return CONTINUE;
}

/**
 * After the tool loop drains: queued follow-ups re-enter the outer loop;
 * otherwise the run proceeds to the stop-policy and acceptance phases.
 */
export async function drainPostTurnFollowUps(
	ctx: TurnContext,
): Promise<"continue" | "proceed"> {
	ctx.pendingMessages = ctx.runController.acceptanceStopRequested
		? []
		: await drainFollowUps(ctx);
	return ctx.pendingMessages.length > 0 ? "continue" : "proceed";
}

/** Structured stop policies may finish or steer the run. */
export async function evaluateStopPolicyPhase(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const stopPolicies = [
		...(ctx.config.verifiedStopEnabled === true
			? [createVerifiedStopPolicy()]
			: []),
		...(ctx.config.stopPolicies ?? []),
	];
	const decision = await evaluateStopPolicies(
		stopPolicies,
		{
			messages: ctx.messages,
			newMessages: ctx.newMessages,
			iteration: ctx.iteration,
			signal: ctx.config.signal,
		},
		evaluation => ctx.emit({ type: "policy_evaluation", ...evaluation }),
	);
	if (decision?.action === "finish") {
		return {
			kind: "finish",
			status: decision.status,
			summary: decision.summary,
			source: "structured",
		};
	}
	if (decision?.action === "continue" && decision.messages.length > 0) {
		ctx.pendingMessages = decision.messages;
		return CONTINUE;
	}
	return BREAK;
}

/**
 * Deterministic verification gets one bounded repair turn. This happens only
 * after the ordinary autonomous policy considers the work finished.
 */
export async function runAcceptanceRepairPhase(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	if (ctx.resolved.verify.length > 0) {
		await ctx.emit({
			type: "acceptance_start",
			level: ctx.resolved.level,
			criteriaCount: ctx.resolved.criteria.length,
		});
		ctx.cachedVerificationResults = await verifyAcceptanceCommands(
			ctx.resolved,
			{ cwd: ctx.config.cwd, signal: ctx.config.signal },
		);
		for (const result of ctx.cachedVerificationResults) {
			await ctx.emit({
				type: "acceptance_verify",
				command: result.command,
				result: result.result,
				summary: result.summary,
			});
		}
		if (
			ctx.runController.requestVerificationRepair(
				ctx.cachedVerificationResults,
				ctx.iteration < ctx.maxIterations,
			)
		) {
			const content = formatVerificationRepair(ctx.cachedVerificationResults);
			await intervene(ctx, {
				kind: "verification",
				cause: "verification_failed",
				detector: "acceptance_verifier",
				message: content,
				iteration: ctx.iteration,
				action: "recover",
				limits: { repairAttempts: 1 },
			});
			ctx.pendingMessages = [{ role: "user", content, timestamp: Date.now() }];
			return CONTINUE;
		}
	}
	return BREAK;
}

/** Terminal sequence after the loop exits: finalization plus agent_end. */
export async function finalizeRun(ctx: TurnContext): Promise<Message[]> {
	const finalMessagesForConclusion = ctx.newMessages;

	if (ctx.iteration >= ctx.maxIterations) {
		await ctx.emit({
			type: "max_iterations",
			iterations: ctx.iteration,
			limit: ctx.maxIterations,
		});
	}

	// ── Acceptance finalization ────────────────────────────────────────
	if (shouldRunAcceptanceFinalization(ctx.resolved)) {
		const verificationResults =
			ctx.cachedVerificationResults ??
			(await verifyAcceptanceCommands(ctx.resolved, {
				cwd: ctx.config.cwd,
				signal: ctx.config.signal,
			}));

		// Emit verification events
		for (const result of verificationResults) {
			await ctx.emit({
				type: "acceptance_verify",
				command: result.command,
				result: result.result,
				summary: result.summary,
			});
		}

		const hasFailures = verificationResults.some(
			r =>
				r.result === "failed" &&
				!ctx.resolved.verify.find(v => v.command === r.command)?.allowFailure,
		);

		ctx.acceptanceFailed = hasFailures;

		await ctx.emit({
			type: "acceptance_complete",
			status: hasFailures ? "failed" : "passed",
		});
	}

	// Final output guard reset when agent ends
	ctx.outputGuard?.reset();
	// Acceptance failure must take precedence over a model-declared `done`.
	if (ctx.acceptanceFailed) {
		return finishRun(ctx, {
			status: "failed",
			summary:
				"Acceptance contract not satisfied after the configured finalization turns.",
			source: "runtime",
		});
	}
	if (ctx.config.signal?.aborted) {
		return finishRun(ctx, {
			status: "cancelled",
			summary: isSteeringInterrupt(ctx.config.signal)
				? STEERING_INTERRUPT_SUMMARY
				: "Operation aborted.",
			source: "runtime",
		});
	}

	// Preserve the finalized transcript returned by the loop.
	ctx.newMessages.splice(
		0,
		ctx.newMessages.length,
		...finalMessagesForConclusion,
	);

	const finalText = lastAssistantContent(finalMessagesForConclusion);
	return finishRun(ctx, {
		status: ctx.iteration >= ctx.maxIterations ? "failed" : "completed",
		summary: finalText || undefined,
		source:
			ctx.iteration >= ctx.maxIterations ||
			!ctx.executionPolicy.embeddedPoliciesEnabled
				? "runtime"
				: "heuristic",
	});
}

// ── Inner-loop (per-turn) phases ─────────────────────────────────────────────

/** Budget gate for the upcoming provider call. */
export async function checkProviderBudget(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const decision = checkBudget(ctx.runBudget, "provider_call");
	if (!decision.allowed) {
		await recordBudgetExhaustion(ctx, decision);
		return budgetExhaustedOutcome(decision);
	}
	return CONTINUE;
}

/** Start a turn: advance the counter, emit turn_start, flush pending messages. */
export async function beginTurn(ctx: TurnContext): Promise<PhaseOutcome> {
	ctx.iteration++;
	const turnId = `turn_${ctx.iteration}`;
	ctx.turn = { turnId };
	await ctx.emit({ type: "turn_start", turnId });

	if (ctx.pendingMessages.length > 0) {
		for (const pending of ctx.pendingMessages) {
			ctx.messages.push(pending);
			ctx.newMessages.push(pending);
			await emitMessagePair(ctx.emit, turnId, pending);
		}
		ctx.pendingMessages = [];
	}
	return CONTINUE;
}

/**
 * Soft tool requirement: inject reminders for a newly activated requirement,
 * then let the host set a soft requirement or hard tool choice for this turn.
 * The manager tracks the active requirement and injects reminders on
 * activation.
 */
export async function applySoftToolRequirements(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	// If a new soft requirement activated, inject its reminder messages before
	// the model call. This avoids the cache-invalidating cost of forcing
	// tool_choice up front.
	const reminder = ctx.softToolManager.getReminder();
	if (reminder && reminder.length > 0) {
		for (const msg of reminder) {
			ctx.messages.push(msg);
			ctx.newMessages.push(msg);
			await emitMessagePair(ctx.emit, ctx.turn.turnId, msg);
		}
	}

	const toolNames = ctx.registry.list().map(t => t.name);
	const toolChoice = await ctx.config.hooks?.getToolChoice?.({
		messages: ctx.messages as Message[],
		iteration: ctx.iteration,
		availableTools: toolNames,
	});
	if (toolChoice && "soft" in toolChoice && toolChoice.soft) {
		ctx.softToolManager.setRequirement(toolChoice);
	}
	return CONTINUE;
}

/**
 * One provider request for the turn, including the request-scoped
 * transformContext rendering and the post-response token budget check.
 */
export async function requestProviderTurn(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	// transformContext is request-scoped only (ExtensionHooks, not
	// RunControlHooks) — its result must build this turn's outgoing payload
	// and nothing else. It must never be folded back onto the canonical
	// `messages`, or a transient injection (e.g. memory retrieval context)
	// silently becomes part of durable history. Persistent edits belong in
	// prepareNextTurn/beforeAgentStart instead.
	const transformResult = await ctx.config.hooks?.transformContext?.({
		messages: ctx.messages as AgentMessage[],
		iteration: ctx.iteration,
		signal: ctx.config.signal,
	});
	const requestMessages = transformResult?.messages as Message[] | undefined;

	const turnResult = await requestAssistantTurn({
		state: ctx.providerTurnState,
		messages: ctx.messages,
		presentationMessages: requestMessages,
		config: ctx.config,
		settings: ctx.settings,
		registry: ctx.registry,
		outputGuard: ctx.outputGuard,
		turnId: ctx.turn.turnId,
		iteration: ctx.iteration,
		adaptiveObjective: ctx.adaptiveObjective,
		performedToolWork: ctx.performedToolWork,
		toolFailures: ctx.toolFailures,
		contextWasCompacted: ctx.contextWasCompacted,
		convertToLlm: ctx.config.convertToLlm ?? defaultConvertToLlm,
		emit: ctx.emit,
		intervene: input => intervene(ctx, input),
		isSteeringInterrupt,
		steeringInterruptSummary: STEERING_INTERRUPT_SUMMARY,
	});
	if (turnResult.kind === "finish") {
		return {
			kind: "finish",
			status: turnResult.outcome.status,
			summary: turnResult.outcome.summary,
			source: turnResult.outcome.source,
		};
	}
	ctx.messages = turnResult.messages;
	ctx.contextWasCompacted = turnResult.contextWasCompacted;
	ctx.turn.response = turnResult.response;
	ctx.turn.prefixStability = turnResult.prefixStability;

	const tokenBudget = checkBudget(
		ctx.runBudget,
		"tokens",
		turnResult.response?.usage?.totalTokens ?? 0,
	);
	if (!tokenBudget.allowed) {
		await recordBudgetExhaustion(ctx, tokenBudget);
		return budgetExhaustedOutcome(tokenBudget);
	}
	return CONTINUE;
}

/**
 * Handle the provider response: record the assistant message, run text-level
 * loop detection (one nudge, then a block), or fail the run.
 */
export async function processAssistantResponse(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const response = ctx.turn.response;
	const processResult = processProviderResponse({
		response,
		registry: ctx.registry,
		outputGuard: ctx.outputGuard ?? null,
		messages: ctx.messages,
		newMessages: ctx.newMessages,
		turnId: ctx.turn.turnId,
		iteration: ctx.iteration,
		emit: ctx.emit,
		config: ctx.config,
	});

	if (!processResult.success) {
		return {
			kind: "finish",
			status: "failed",
			summary: processResult.errorMessage ?? "Model returned empty response.",
			source: "runtime",
		};
	}
	ctx.turn.toolCalls = processResult.toolCalls;
	ctx.turn.assistant = processResult.assistant;
	if (processResult.toolCalls.length > 0) {
		ctx.performedToolWork = true;
	}

	// Text-level loop detection on assistant content
	const assistantText = processResult.assistant.content ?? "";
	if (assistantText.length > 200) {
		const loopHit = ctx.textLoopDetector.check(assistantText);
		if (loopHit) {
			// Bounded recovery: the first text-only hit injects one
			// strategy-change nudge (same escalation pattern as the soft-tool
			// manager); a second hit blocks the run. A first hit while tool
			// calls are landing is counted only — the run is making work
			// progress, so the batch runs as usual and the next text hit blocks.
			ctx.stagnationNudges++;
			ctx.textLoopDetector.reset();
			const hasToolWork = processResult.toolCalls.length > 0;
			if (ctx.stagnationNudges > 1) {
				await intervene(ctx, {
					kind: "loop",
					cause: "text_stagnation",
					detector: "text_loop_detector",
					message: `Text stagnation detected: ${loopHit}`,
					iteration: ctx.iteration,
					action: "change_strategy",
				});
				return {
					kind: "finish",
					status: "blocked",
					summary:
						"Agent entered a text stagnation loop — repeating content without progress.",
					source: "runtime",
				};
			}
			if (hasToolWork) {
				await intervene(ctx, {
					kind: "loop",
					cause: "text_stagnation",
					detector: "text_loop_detector",
					message: `Text stagnation detected while tools were running: ${loopHit}`,
					iteration: ctx.iteration,
					action: "change_strategy",
				});
			} else {
				await intervene(ctx, {
					kind: "loop",
					cause: "text_stagnation",
					detector: "text_loop_detector",
					message: `Text stagnation detected: ${loopHit}; strategy-change nudge injected.`,
					iteration: ctx.iteration,
					action: "change_strategy",
				});
				ctx.pendingMessages = [
					{
						role: "user",
						content: formatStagnationNudge(loopHit),
						timestamp: Date.now(),
					},
				];
				return REENTER;
			}
		}
	}
	return CONTINUE;
}

/** Execute the turn's tool batch and fold the results into the conversation. */
export async function executeToolCalls(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const toolCalls = ctx.turn.toolCalls ?? [];
	const rawStopReason =
		(ctx.turn.response?.stopReason as "stop" | "length" | "error") ?? "stop";
	ctx.turn.stopReason = stopReasonFor(rawStopReason, toolCalls);
	ctx.hasMoreToolCalls = false;

	const toolBudget = checkBudget(ctx.runBudget, "tool_batch", toolCalls.length);
	if (!toolBudget.allowed) {
		await recordBudgetExhaustion(ctx, toolBudget);
		return budgetExhaustedOutcome(toolBudget);
	}

	const batch = await executeToolBatch({
		registry: ctx.registry,
		toolCalls,
		rawStopReason,
		toolExecution: ctx.settings.toolExecution,
		iteration: ctx.iteration,
		signal: ctx.config.signal,
		hooks: ctx.config.hooks,
		permissions: ctx.config.permissions,
		onPermissionRequest: ctx.config.onPermissionRequest,
		emit: ctx.emit,
	});
	ctx.turn.batch = batch;
	for (const toolResult of batch.messages) {
		if (isToolFailureResult(String(toolResult.content ?? ""))) {
			ctx.toolFailures++;
		}
		ctx.messages.push(toolResult);
		ctx.newMessages.push(toolResult);
		await emitMessagePair(ctx.emit, ctx.turn.turnId, toolResult);
		ctx.hasMoreToolCalls = true;
	}
	return CONTINUE;
}

/**
 * Soft tool requirement compliance: a requirement unmet after the bounded
 * escalation budget aborts the run.
 */
export async function enforceSoftToolCompliance(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const toolCalls = ctx.turn.toolCalls ?? [];
	const toolCallObjects = toolCalls.map(tc => {
		let args: Record<string, unknown> = {};
		try {
			args = JSON.parse(tc.arguments ?? "{}");
		} catch {
			args = {};
		}
		return { name: tc.name, arguments: args };
	});
	try {
		ctx.softToolManager.checkCompliance(toolCallObjects);
	} catch (err) {
		if (err instanceof SoftToolRequirementExceededError) {
			const toolName = err.message.split("'")[1];
			await intervene(ctx, {
				kind: "loop",
				cause: "soft_tool_requirement_exceeded",
				detector: "soft_tool_requirement",
				message: `Soft tool requirement for '${toolName}' was not satisfied after ${MAX_ESCALATIONS} forced turns; aborting.`,
				iteration: ctx.iteration,
				action: "stop",
			});
			return {
				kind: "finish",
				status: "blocked",
				summary:
					"Soft tool requirement was not met after repeated escalations.",
				source: "runtime",
			};
		}
		throw err;
	}
	return CONTINUE;
}

/** Repeated permission denials pause autonomous execution for the user. */
export async function checkPermissionEscalation(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const batch = ctx.turn.batch;
	if (!batch) return CONTINUE;
	const permissionEscalation = ctx.runController.recordPermissionBatch({
		denials: batch.permissionDenials,
		executed: batch.executedToolCallIds.length,
	});
	if (permissionEscalation) {
		await intervene(ctx, {
			kind: "loop",
			cause: "permission_denials",
			detector: "permission_escalation",
			message:
				"Autonomous execution paused after repeated permission denials. User authorization or a different task scope is required.",
			iteration: ctx.iteration,
			action: "pause",
			counters: {
				consecutive: permissionEscalation.consecutive,
				total: permissionEscalation.total,
			},
			limits: { consecutive: 3, total: 20 },
		});
		return {
			kind: "finish",
			status: "needs_input",
			summary:
				"Repeated permission denials require user authorization or a safer scope.",
			source: "runtime",
		};
	}
	return CONTINUE;
}

/**
 * Context budget: report usage for the turn and compact immediately when the
 * output guard declares the effective window exhausted.
 */
export async function manageContextBudget(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	// The final usage-only SSE chunk is optional and many local providers
	// omit it. Prefer the provider-reported total when present; only fall
	// back to estimating the serialized conversation (an expensive
	// full-history pass) when the provider reported nothing.
	const response = ctx.turn.response;
	const reportedTokens = response?.usage?.totalTokens ?? 0;
	const contextTokens =
		reportedTokens > 0
			? reportedTokens
			: await estimateChatPayloadTokens(
					ctx.messages,
					ctx.registry.toToolDefinitions(),
					ctx.tokenEncoding,
				);
	// The effective window follows the active model: the per-model cap when
	// configured for it (model cycling mutates config mid-run), otherwise
	// the global setting.
	const effectiveContextWindow = resolveModelContextWindow(
		ctx.config.models,
		ctx.config.model,
		ctx.config.contextWindowTokens,
	);
	await ctx.emit({
		type: "context_update",
		tokens: contextTokens,
		maxTokens: effectiveContextWindow,
		cachedTokens: response?.usage?.cachedTokens ?? null,
		promptTokens: response?.usage?.promptTokens ?? null,
		completionTokens: response?.usage?.completionTokens ?? null,
		prefixStable: ctx.turn.prefixStability?.stable,
		prefixDivergedAt: ctx.turn.prefixStability?.divergedAt,
		prefixRewritten: ctx.turn.prefixStability?.rewritten,
	});
	if (effectiveContextWindow) {
		const budgetResult = ctx.outputGuard?.processResponse(
			contextTokens,
			effectiveContextWindow,
		);
		// budget_exhausted is a harder threshold than proactive compaction's
		// (95% vs 80%) — if we're here, proactive compaction already failed
		// to keep up (e.g. cooldown window, or a single oversized turn).
		// Compact immediately rather than waiting for the next request to
		// fail with context_full.
		if (budgetResult?.action === "budget_exhausted") {
			const compacted = await compactToFit(
				ctx.messages as CompactableMessage[],
				{
					triggerTokens: 0,
					targetTokens: Math.floor(effectiveContextWindow * 0.75),
				},
			);
			if (compacted.changed) {
				ctx.messages = toMessages(compacted.messages);
				ctx.contextWasCompacted = true;
				ctx.config.onContextCompacted?.(ctx.messages);
				await ctx.emit({
					type: "context_update",
					tokens: compacted.tokensAfter,
					maxTokens: effectiveContextWindow,
					compacted: true,
				});
				await intervene(ctx, {
					kind: "compaction",
					cause: "budget_exhausted",
					detector: "context_budget",
					message: `Context compacted from ${compacted.tokensBefore} to ${compacted.tokensAfter} tokens.`,
					iteration: ctx.iteration,
					counters: {
						tokensBefore: compacted.tokensBefore,
						tokensAfter: compacted.tokensAfter,
					},
				});
			}
		}
	}
	return CONTINUE;
}

/** Close the turn: emit turn_end and reset the output guard. */
export async function endTurn(ctx: TurnContext): Promise<PhaseOutcome> {
	await ctx.emit({
		type: "turn_end",
		turnId: ctx.turn.turnId,
		stopReason: ctx.turn.stopReason,
		message: ctx.turn.assistant,
		toolResults: ctx.turn.batch?.messages,
	});

	// Reset output guard after each completed turn
	ctx.outputGuard?.reset();
	return CONTINUE;
}

/**
 * Mid-run config refresh (model cycling, permission changes) plus the host's
 * prepareNextTurn hook, which may replace the canonical messages.
 */
export async function prepareNextTurn(ctx: TurnContext): Promise<PhaseOutcome> {
	const refreshedConfig = await ctx.config.refreshNextTurnConfig?.();
	if (refreshedConfig) {
		Object.assign(ctx.config, refreshedConfig);
		ctx.settings = resolveAgentSettings(ctx.config);
		ctx.context.systemPrompt = refreshedConfig.systemPrompt;
		ctx.messages = [
			createSystemMessage(
				refreshedConfig.systemPrompt ?? "You are a helpful assistant.",
			),
			...ctx.messages.filter(message => message.role !== "system"),
		];
		ctx.registry = createToolRegistry(
			ctx.context,
			ctx.config,
			ctx.cache,
			refreshedConfig.tools ?? [],
		);
	}

	const prepareResult = await ctx.config.hooks?.prepareNextTurn?.({
		messages: ctx.messages,
		iteration: ctx.iteration,
		hadToolCalls: (ctx.turn.toolCalls?.length ?? 0) > 0,
	});
	const prepared = prepareResult?.messages;
	if (prepared) {
		ctx.messages = prepared;
		if (ctx.contextWasCompacted) ctx.config.onContextCompacted?.(ctx.messages);
	}
	return CONTINUE;
}

/**
 * Decide whether the inner loop continues: a tool-terminated run still drains
 * follow-ups; stop hooks and acceptance stop rules exit; otherwise queued
 * steering re-enters the loop.
 */
export async function decideTurnContinuation(
	ctx: TurnContext,
): Promise<PhaseOutcome> {
	const toolCalls = ctx.turn.toolCalls ?? [];

	// when a tool signals terminate, still drain followUps before exiting.
	// This prevents skipping queued follow-up messages (e.g. steering
	// injected mid-turn) just because a tool requested termination.
	if (ctx.turn.batch?.terminated) {
		const followUpsOnTerminate = await drainFollowUps(ctx);
		if (followUpsOnTerminate.length > 0) {
			if (
				!followUpsOnTerminate.some(message =>
					String(message.content).startsWith("[continuation-nudge:"),
				)
			) {
				await intervene(ctx, {
					kind: "continuation",
					cause: "follow_up_after_termination",
					detector: "follow_up_queue",
					message: `Harness scheduled ${followUpsOnTerminate.length} follow-up message(s) after tool termination.`,
					iteration: ctx.iteration,
				});
			}
			ctx.pendingMessages = followUpsOnTerminate;
			ctx.hasMoreToolCalls = false;
			// Re-enter inner loop with follow-up messages
			return REENTER;
		}
		return { kind: "finish", status: "completed", source: "runtime" };
	}

	// only invoke shouldStopAfterTurn when no tool calls ran.
	// Tool turns always continue unless the hook is explicitly wired to stop
	// on tool turns — checking it unconditionally causes premature exits when
	// hooks have stale state from a previous no-tool turn.
	const stop =
		toolCalls.length === 0
			? ((await ctx.config.hooks?.shouldStopAfterTurn?.({
					messages: ctx.messages,
					iteration: ctx.iteration,
					hadToolCalls: false,
				})) ?? false)
			: false;
	// Acceptance stop rules take priority
	let acceptanceStop = false;
	if (!stop && shouldRunAcceptanceFinalization(ctx.resolved)) {
		acceptanceStop = checkStopRules(ctx);
	}
	if (stop || acceptanceStop) {
		if (stop) return { kind: "finish", status: "completed", source: "runtime" };
		ctx.runController.requestAcceptanceStop();
		return BREAK;
	}

	ctx.pendingMessages = await drainSteering(ctx);
	return CONTINUE;
}

// ── Phase sequences ──────────────────────────────────────────────────────────

/** Per-turn sequence: one provider round-trip and everything around it. */
export const INNER_PHASES: readonly LoopPhase[] = [
	checkProviderBudget,
	beginTurn,
	applySoftToolRequirements,
	requestProviderTurn,
	processAssistantResponse,
	executeToolCalls,
	enforceSoftToolCompliance,
	checkPermissionEscalation,
	manageContextBudget,
	endTurn,
	prepareNextTurn,
	decideTurnContinuation,
];

/**
 * Run a phase sequence until one phase breaks or finishes. `finish` results
 * are materialized into the run's final transcript; completing the sequence
 * returns `continue`.
 */
export async function runPhaseSequence(
	ctx: TurnContext,
	phases: readonly LoopPhase[],
): Promise<PhaseOutcome | Message[]> {
	for (const phase of phases) {
		const outcome = await phase(ctx);
		if (outcome.kind === "finish") {
			return finishRun(ctx, outcome);
		}
		if (outcome.kind === "break" || outcome.kind === "reenter") {
			return outcome;
		}
	}
	return CONTINUE;
}
