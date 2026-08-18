// ── Post-turn continuation policy ────────────────────────────────────────
// When a turn produces no more tool calls, something has to decide whether
// the run is actually done: drain follow-ups, nudge toward a structured
// conclusion, defer to a user question, run custom stop policies, retry a
// failed acceptance report, or ask reflection for a second pass. This module
// is that decision, pulled out of the loop's control flow so the loop itself
// only has to act on the result.

import {
	lastAssistantContent,
	lastHadToolCalls,
} from "../core/conclusion-policy.ts";
import type { LLMBackend } from "../core/backend.ts";
import type {
	AgentStopPolicy,
	RunOutcomeStatus,
} from "../core/execution-policy.ts";
import { evaluateStopPolicies } from "../core/execution-policy.ts";
import type { InterventionInput } from "../core/intervention-controller.ts";
import type { ResolvedAcceptance } from "../guards/acceptance-contract.ts";
import {
	evaluateAcceptanceReport,
	parseAcceptanceReport,
	shouldRunAcceptanceFinalization,
	verifyAcceptanceCommands,
} from "../guards/acceptance-contract.ts";
import { awaitsUserInput, looksComplete } from "../guards/response-patterns.ts";
import type { ReflectionConfig } from "../loop/reflection.ts";
import { runReflection } from "../loop/reflection.ts";
import type { AgentEventSink, Message } from "../types/index.ts";
import { getTaskStatus } from "./task-status-state.ts";

const MAX_CONSECUTIVE_RUNNER_NUDGES = 3;

export interface ContinuationPolicyState {
	consecutiveRunnerNudges: number;
	lastRunnerNudgeIteration: number;
	acceptanceReported: boolean;
	acceptanceFailed: boolean;
	acceptanceFinalizationTurns: number;
	reflectionCount: number;
	reflectionFailed: boolean;
}

export function createContinuationPolicyState(): ContinuationPolicyState {
	return {
		consecutiveRunnerNudges: 0,
		lastRunnerNudgeIteration: -1,
		acceptanceReported: false,
		acceptanceFailed: false,
		acceptanceFinalizationTurns: 0,
		reflectionCount: 0,
		reflectionFailed: false,
	};
}

export interface ContinuationPolicyInput {
	state: ContinuationPolicyState;
	/** Follow-up messages already drained from the harness's follow-up hook for this stop point. */
	followUps: Message[];
	messages: Message[];
	newMessages: Message[];
	iteration: number;
	maxIterations: number;
	performedToolWork: boolean;
	lastToolWorkIteration: number;
	hasTaskStatusTool: boolean;
	embeddedPoliciesEnabled: boolean;
	continuationEnabled: boolean | undefined;
	resolvedAcceptance: ResolvedAcceptance;
	reflectionEnabled: boolean;
	maxReflections: number;
	stopPolicies: readonly AgentStopPolicy[] | undefined;
	backend: LLMBackend;
	reflectionConfig: ReflectionConfig | undefined;
	cwd: string | undefined;
	signal: AbortSignal | undefined;
	emit: AgentEventSink;
	intervene: (input: InterventionInput) => Promise<void> | void;
}

export type ContinuationDecision =
	| { action: "continue"; pendingMessages: Message[] }
	| { action: "break" }
	| {
			action: "finish";
			outcome: {
				status: RunOutcomeStatus;
				summary?: string;
				source: "structured" | "heuristic" | "runtime";
			};
	  };

/**
 * Decide what happens after a turn with no pending tool calls. Mirrors the
 * pi-style outer loop: drain follow-ups, apply the runner's own
 * continuation nudge, defer to the user, run custom stop policies, then
 * (if embedded policies are enabled) retry acceptance or reflection before
 * finally allowing the run to end.
 */
export async function decidePostTurnContinuation(
	input: ContinuationPolicyInput,
): Promise<ContinuationDecision> {
	const { state, followUps } = input;

	if (
		followUps.length > 0 &&
		!followUps.some(message =>
			String(message.content).startsWith("[continuation-nudge:"),
		)
	) {
		await input.intervene({
			kind: "continuation",
			cause: "follow_up",
			detector: "follow_up_queue",
			message: `Harness scheduled ${followUps.length} follow-up message(s).`,
			iteration: input.iteration,
			action: "continue",
		});
	}

	let continuationExhausted = false;
	if (
		input.embeddedPoliciesEnabled &&
		input.continuationEnabled === true &&
		followUps.length === 0
	) {
		continuationExhausted = await applyRunnerNudge(input, followUps);
	}

	if (followUps.length > 0) {
		return { action: "continue", pendingMessages: followUps };
	}
	if (continuationExhausted) {
		return {
			action: "finish",
			outcome: {
				status: "blocked",
				summary: `Continuation exhausted after ${MAX_CONSECUTIVE_RUNNER_NUDGES} nudges without tool progress.`,
				source: "runtime",
			},
		};
	}

	// A final question hands control back to the user. It must beat
	// reflection, acceptance finalization, and all other synthetic turns;
	// otherwise the loop fabricates an answer by prompting the model again.
	if (
		input.embeddedPoliciesEnabled &&
		awaitsUserInput(lastAssistantContent(input.newMessages))
	) {
		return {
			action: "finish",
			outcome: {
				status: "needs_input",
				summary: "Agent is waiting for the user's answer.",
				source: "heuristic",
			},
		};
	}

	const policyDecision = await evaluateStopPolicies(input.stopPolicies, {
		messages: input.messages,
		newMessages: input.newMessages,
		iteration: input.iteration,
		signal: input.signal,
	});
	if (policyDecision?.action === "continue") {
		await input.intervene({
			kind: "continuation",
			cause: "stop_policy",
			detector: "custom_stop_policy",
			message: `A stop policy continued the run with ${policyDecision.messages.length} follow-up message(s).`,
			iteration: input.iteration,
			action: "continue",
		});
		if (policyDecision.messages.length > 0) {
			return {
				action: "continue",
				pendingMessages: policyDecision.messages as Message[],
			};
		}
	} else if (policyDecision?.action === "finish") {
		return {
			action: "finish",
			outcome: {
				status: policyDecision.status,
				summary: policyDecision.summary,
				source: "structured",
			},
		};
	}

	if (!input.embeddedPoliciesEnabled) return { action: "break" };

	const acceptanceDecision = await applyAcceptanceRetry(input);
	if (acceptanceDecision) return acceptanceDecision;

	const reflectionDecision = await applyReflectionRetry(input);
	if (reflectionDecision) return reflectionDecision;

	return { action: "break" };
}

/** Mutates `state` and appends the nudge into `followUps` when eligible. Returns whether nudges are exhausted. */
async function applyRunnerNudge(
	input: ContinuationPolicyInput,
	followUps: Message[],
): Promise<boolean> {
	const { state } = input;
	const text = lastAssistantContent(input.newMessages);
	const hadTools = lastHadToolCalls(input.newMessages);
	const waitingForUser = awaitsUserInput(text);
	const hasStructuredStop = getTaskStatus() !== null;
	const hasAcceptanceReport =
		shouldRunAcceptanceFinalization(input.resolvedAcceptance) &&
		parseAcceptanceReport(text).report !== undefined;

	const requiresStructuredConclusion =
		input.performedToolWork && input.hasTaskStatusTool;

	// Real tool work since the last nudge means the run is actually
	// progressing, not stalled — give it a fresh nudge budget rather than
	// counting this stall toward the same cap as the last one.
	if (input.lastToolWorkIteration > state.lastRunnerNudgeIteration) {
		state.consecutiveRunnerNudges = 0;
	}

	const eligibleForNudge =
		!hadTools &&
		!waitingForUser &&
		!hasAcceptanceReport &&
		requiresStructuredConclusion &&
		!hasStructuredStop;

	if (!eligibleForNudge) {
		// Model signaled completion, has structured stop, or cap reached — reset.
		state.consecutiveRunnerNudges = 0;
		return false;
	}

	if (state.consecutiveRunnerNudges < MAX_CONSECUTIVE_RUNNER_NUDGES) {
		const nudgeTag = "[continuation-nudge:structured-conclusion]";
		const nudgeContent =
			`${nudgeTag} Do not stop yet without a structured conclusion. Verify that every requested step is complete. ` +
			"If work remains, continue with the next step. If the task is complete, blocked, failed, or needs user input, " +
			"call task_status with the accurate status as your final action.";
		followUps.push({ role: "user" as const, content: nudgeContent });
		await input.intervene({
			kind: "continuation",
			cause: "missing_structured_conclusion",
			detector: "runner_continuation",
			message: nudgeContent,
			iteration: input.iteration,
			counters: { consecutiveRunnerNudges: state.consecutiveRunnerNudges },
			limits: { maxConsecutiveNudges: MAX_CONSECUTIVE_RUNNER_NUDGES },
		});
		state.consecutiveRunnerNudges++;
		state.lastRunnerNudgeIteration = input.iteration;
		return false;
	}

	await input.intervene({
		kind: "continuation",
		cause: "continuation_exhausted",
		detector: "runner_continuation",
		message: `Continuation stopped after ${MAX_CONSECUTIVE_RUNNER_NUDGES} consecutive nudges without observable tool progress.`,
		iteration: input.iteration,
		counters: { consecutiveRunnerNudges: state.consecutiveRunnerNudges },
		limits: { maxConsecutiveNudges: MAX_CONSECUTIVE_RUNNER_NUDGES },
	});
	return true;
}

/** A failed acceptance report is actionable feedback, not an immediate terminal failure. */
async function applyAcceptanceRetry(
	input: ContinuationPolicyInput,
): Promise<ContinuationDecision | undefined> {
	const { state } = input;
	if (
		!shouldRunAcceptanceFinalization(input.resolvedAcceptance) ||
		state.acceptanceReported
	) {
		return undefined;
	}

	const verificationResults = await verifyAcceptanceCommands(
		input.resolvedAcceptance,
		{ cwd: input.cwd, signal: input.signal },
	);
	const report = evaluateAcceptanceReport(
		lastAssistantContent(input.newMessages),
		input.resolvedAcceptance,
		verificationResults,
	);

	if (report.status === "passed") {
		state.acceptanceReported = true;
		await input.emit({
			type: "acceptance_complete",
			status: report.status,
			report: report.ledger as unknown as Record<string, unknown>,
		});
		return { action: "break" };
	}

	const maxTurns = input.resolvedAcceptance.maxFinalizationTurns ?? 3;
	if (
		state.acceptanceFinalizationTurns < maxTurns &&
		input.iteration < input.maxIterations
	) {
		state.acceptanceFinalizationTurns++;
		const acceptanceRetryContent =
			`[continuation-nudge:acceptance-retry] Acceptance validation failed (attempt ${state.acceptanceFinalizationTurns}/${maxTurns}). ` +
			"Review the acceptance contract, fix any unmet criteria or verification failures, and finish with a valid acceptance-report block.";
		await input.intervene({
			kind: "verification",
			cause: "acceptance_failed",
			detector: "acceptance_contract",
			message: acceptanceRetryContent,
			iteration: input.iteration,
			counters: {
				acceptanceFinalizationTurns: state.acceptanceFinalizationTurns,
			},
			limits: { maxFinalizationTurns: maxTurns },
		});
		return {
			action: "continue",
			pendingMessages: [{ role: "user", content: acceptanceRetryContent }],
		};
	}

	state.acceptanceReported = true;
	state.acceptanceFailed = true;
	await input.emit({
		type: "acceptance_complete",
		status: report.status,
		report: report.ledger as unknown as Record<string, unknown>,
	});
	return { action: "break" };
}

/** Reflection is a verifier, not a synthetic assistant turn: feed findings back through the normal loop. */
async function applyReflectionRetry(
	input: ContinuationPolicyInput,
): Promise<ContinuationDecision | undefined> {
	const { state } = input;
	if (
		!input.reflectionEnabled ||
		looksComplete(lastAssistantContent(input.newMessages))
	) {
		return undefined;
	}

	if (state.reflectionCount >= input.maxReflections) {
		state.reflectionFailed = true;
		await input.emit({
			type: "task_failed",
			reason: `Agent reached the ${input.maxReflections}-reflection safety limit without completing the task.`,
			iteration: state.reflectionCount,
			lastContent: lastAssistantContent(input.newMessages),
		});
		return { action: "break" };
	}

	const reflection = await runReflection(
		input.newMessages,
		input.backend,
		input.reflectionConfig ?? { enabled: true },
		input.emit,
		input.signal,
	);
	state.reflectionCount++;
	if (!reflection.result.needsMoreWork) return undefined;

	const suggested = reflection.result.suggestedSteps.join("; ");
	const reflectionRetryContent =
		reflection.result.issues.length > 0
			? `[continuation-nudge:reflection-retry] Reflection found issues: ${reflection.result.issues.join(", ")}. Address them and continue working.`
			: `[continuation-nudge:reflection-retry] Reflection found the task incomplete. ${suggested ? `Suggested next steps: ${suggested}. ` : ""}Continue working.`;
	await input.intervene({
		kind: "verification",
		cause: "reflection_incomplete",
		detector: "reflection",
		message: reflectionRetryContent,
		iteration: input.iteration,
		counters: { reflectionCount: state.reflectionCount },
		limits: { maxReflections: input.maxReflections },
	});
	return {
		action: "continue",
		pendingMessages: [{ role: "user", content: reflectionRetryContent }],
	};
}
