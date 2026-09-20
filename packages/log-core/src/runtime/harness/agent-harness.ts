// ── Functional Agent Loop ─────────────────────────────────────────────────
// Pi-style loop contract for Logician's current backend/tool adapter:
// context + prompts + config + emit => new messages.
//
// The loop body is a sequence of per-turn phases (turn-phases.ts) over a
// shared TurnContext record; this module owns run setup, the loop skeleton,
// and the public contract.

import { resolveTokenEncoding } from "../../capabilities/provider/messages.ts";
import { ToolResultCache } from "../../capabilities/tools/tool-result-cache.ts";
import { resolveAgentSettings } from "../../control/configuration/agent-settings.ts";
import { resolveEffectiveAcceptance } from "../../control/guards/acceptance-contract.ts";
import { SoftToolRequirementManager } from "../../control/guards/soft-tool-requirement.ts";
import { TextLoopDetector } from "../../control/guards/text-loop-detector.ts";
import { resolveExecutionPolicy } from "../../control/policy/execution-policy.ts";
import { HarnessInterventionController } from "../../control/policy/intervention-controller.ts";
import { RunBudgetController } from "../../control/policy/run-budget.ts";
import { AgentRunController } from "../../control/policy/run-controller.ts";
import type {
	AgentEventSink,
	AgentMessage,
	Message,
	Tool,
} from "../../system/types/types-messages.ts";
import { taskObjectiveFromMessages } from "../loop/adaptive-mode.ts";
import { emitMessagePair, withSystemPrompt } from "../loop/callbacks.ts";
import type { AgentLoopConfig } from "../loop/config.ts";
import { createProviderTurnState } from "../loop/provider-turn.ts";
import {
	checkRunAbort,
	createToolRegistry,
	drainPostTurnFollowUps,
	drainSteering,
	evaluateStopPolicyPhase,
	finalizeRun,
	finishRun,
	INNER_PHASES,
	resolveAcceptance,
	runAcceptanceRepairPhase,
	runPhaseSequence,
	type TurnContext,
} from "./turn-phases.ts";

// Re-exported so existing import paths keep working.
export {
	createSteeringInterruptReason,
	STEERING_INTERRUPT_SUMMARY,
} from "./turn-phases.ts";

export interface RunAgentLoopContext {
	systemPrompt?: string | undefined;
	messages: Message[];
	tools?: Tool[] | undefined;
	cwd?: string | undefined;
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
	const stampedEmit: AgentEventSink = event =>
		downstreamEmit({
			...event,
			seq: ++eventSequence,
			ts: Date.now(),
		});
	const messages = [
		...withSystemPrompt(context.systemPrompt, context.messages),
		...prompts,
	];
	const newMessages: Message[] = [...prompts];
	const settings = resolveAgentSettings(config);
	// Token budgets are only as good as the tokenizer family: resolve the
	// model's BPE encoding once so estimates match the model's vocabulary.
	const tokenEncoding = resolveTokenEncoding(config.backend.model);
	const maxIterations = settings.maxIterations;
	const executionPolicy = resolveExecutionPolicy(settings.executionProfile);
	const interventionController =
		config.interventionController ?? new HarnessInterventionController();
	const runController = config.runController ?? new AgentRunController();
	// ── P0-1: Shared tool result cache ─────────────────────────────────
	const cache = new ToolResultCache(
		config.cacheSize ?? 2000,
		config.cacheTtlMs ?? 60_000,
	);
	const outputGuard = config.outputGuard;
	const adaptiveObjective = taskObjectiveFromMessages([
		...context.messages,
		...prompts,
	]);
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
	const textLoopDetector = new TextLoopDetector();
	const softToolManager = new SoftToolRequirementManager();

	const ctx: TurnContext = {
		context,
		config,
		emit: stampedEmit,
		interventionController,
		runController,
		runBudget,
		textLoopDetector,
		softToolManager,
		providerTurnState,
		cache,
		outputGuard,
		tokenEncoding,
		adaptiveObjective,
		maxIterations,
		executionPolicy,
		messages,
		newMessages,
		iteration: 0,
		pendingMessages: [],
		hasMoreToolCalls: true,
		settings,
		registry: createToolRegistry(
			context,
			config,
			cache,
			context.tools ?? config.tools ?? [],
		),
		performedToolWork: false,
		toolFailures: 0,
		stagnationNudges: 0,
		contextWasCompacted: false,
		acceptanceFailed: false,
		cachedVerificationResults: undefined,
		resolvedAcceptance: null,
		resolved: resolveEffectiveAcceptance({ explicit: undefined }),
		turn: { turnId: "" },
	};

	// Apply beforeAgentStart hook
	ctx.pendingMessages = await drainSteering(ctx);
	const beforeAgentStartResult = await ctx.config.hooks?.beforeAgentStart?.({
		prompt: prompts.map(p => p.content).join("\n"),
		systemPrompt: context.systemPrompt ?? "",
		messages: ctx.messages as AgentMessage[],
	});

	await ctx.emit({ type: "agent_start" });
	const promptTurnId = "turn_0";
	for (const prompt of prompts) {
		await emitMessagePair(ctx.emit, promptTurnId, prompt);
	}

	// Apply beforeAgentStart hook results to messages and system prompt
	if (beforeAgentStartResult?.messages) {
		for (const msg of beforeAgentStartResult.messages) {
			ctx.messages.push(msg as Message);
			ctx.newMessages.push(msg as Message);
		}
	}
	if (beforeAgentStartResult?.systemPrompt) {
		context.systemPrompt = beforeAgentStartResult.systemPrompt;
	}
	if (executionPolicy.embeddedPoliciesEnabled) {
		ctx.resolved = resolveAcceptance(ctx);
	}

	while (ctx.iteration < ctx.maxIterations) {
		const abort = await checkRunAbort(ctx);
		if (abort.kind === "finish") return finishRun(ctx, abort);

		ctx.hasMoreToolCalls = true;
		while (
			(ctx.hasMoreToolCalls || ctx.pendingMessages.length > 0) &&
			ctx.iteration < ctx.maxIterations
		) {
			const inner = await runPhaseSequence(ctx, INNER_PHASES);
			if (Array.isArray(inner)) return inner;
			if (inner.kind === "break") break;
		}
		const followUps = await drainPostTurnFollowUps(ctx);
		if (followUps === "continue") continue;

		// Stop policies and the bounded verification repair are sequential,
		// not a phase sequence: a `continue` from either re-enters the outer
		// loop; a `break` from the stop policies falls through to the
		// acceptance phase, and a `break` from the acceptance phase exits the
		// outer loop.
		const stopPolicy = await evaluateStopPolicyPhase(ctx);
		if (stopPolicy.kind === "finish") return finishRun(ctx, stopPolicy);
		if (stopPolicy.kind === "continue") continue;

		const repair = await runAcceptanceRepairPhase(ctx);
		if (repair.kind === "finish") return finishRun(ctx, repair);
		if (repair.kind === "continue") continue;
		break;
	}

	return finalizeRun(ctx);
}

export function runAgentLoop(
	context: RunAgentLoopContext,
	prompts: Message[],
	config: RunAgentLoopConfig,
	emit: AgentEventSink,
): Promise<Message[]> {
	return runAgentLoopInternal(context, prompts, config, emit);
}
