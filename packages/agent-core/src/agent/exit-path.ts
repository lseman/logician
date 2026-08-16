// ── Exit Path Unification ──────────────────────────────────────────────────
// Consolidates all loop exit conditions (budget, stop policies, acceptance,
// task status, reflection, user input) into a single decision function.
// Replaces 20+ scattered "return finish({ ... })" calls.

import type { Message, RunOutcomeStatus } from "./types.ts";
import type {
	AcceptanceEvaluation,
	ResolvedAcceptance,
	AcceptanceVerificationResult,
} from "./guards/acceptance-contract.ts";
import type { StopPolicyDecision } from "./execution-policy.ts";
import type { RunBudgetController } from "./run-budget.ts";
import type { TaskStatusRecord } from "./tasks/task-status-state.ts";
import type { OutcomeDecision } from "./tasks/outcome-resolution.ts";

export type { OutcomeDecision, RunOutcomeStatus };

// ── Budget Decision ────────────────────────────────────────────────────────

export interface BudgetCheckResult {
	allowed: boolean;
	reason?: string;
	snapshot: {
		providerCalls: number;
		toolCalls: number;
		tokens: number;
		elapsedMs: number;
	};
}

export function checkBudget(
	budget: RunBudgetController,
	request: "provider_call" | "tool_batch" | "tokens",
	count: number = 1,
): BudgetCheckResult {
	switch (request) {
		case "provider_call":
			return budget.requestProviderCall();
		case "tool_batch":
			return budget.requestToolBatch(count);
		case "tokens":
			return budget.recordTokens(count);
	}
}

// ── Exit Decision ──────────────────────────────────────────────────────────

export interface ExitDecision {
	status: RunOutcomeStatus;
	summary?: string;
	source: "structured" | "heuristic" | "runtime";
}

export interface ExitInput {
	iteration: number;
	maxIterations: number;
	signal?: AbortSignal;
	isSteeringInterrupt: boolean;
	declaredStatus: TaskStatusRecord | null;
	hasTaskStatusTool: boolean;
	performedToolWork: boolean;
	acceptanceReported: boolean;
	acceptanceFailed: boolean;
	acceptanceReport?: AcceptanceEvaluation;
	stopPolicyDecision?: StopPolicyDecision;
	userInputAwaited: boolean;
	looksComplete: boolean;
	reflectionFailed: boolean;
	reflectionCount: number;
	maxReflections: number;
	followUps: Message[];
}

/**
 * Single exit decision function.
 *
 * Evaluates all stop conditions in priority order:
 * 1. Abort / cancellation
 * 2. Budget exhaustion
 * 3. Acceptance failure
 * 4. Structured task outcome
 * 5. Stop policies
 * 6. User input awaited
 * 7. Reflection failure
 * 8. Max iterations exceeded
 * 9. Heuristic completion
 */
export function resolveExit(input: ExitInput): ExitDecision | null {
	// 1. Abort / cancellation
	if (input.signal?.aborted) {
		return {
			status: "cancelled",
			summary: input.isSteeringInterrupt
				? "Current provider response interrupted to apply steering."
				: "Operation aborted.",
			source: "runtime",
		};
	}

	// 2. Acceptance failure (takes precedence over model-declared done)
	if (input.acceptanceFailed) {
		return {
			status: "failed",
			summary: "Acceptance contract not satisfied after the configured finalization turns.",
			source: "runtime",
		};
	}

	// 3. Structured task outcome
	if (input.declaredStatus || (input.performedToolWork && input.hasTaskStatusTool)) {
		return resolveOutcome({
			declared: input.declaredStatus,
			structuredOutcomeRequired: input.performedToolWork && input.hasTaskStatusTool,
		});
	}

	// 4. Stop policies
	if (input.stopPolicyDecision) {
		if (input.stopPolicyDecision.action === "finish") {
			return {
				status: input.stopPolicyDecision.status,
				summary: input.stopPolicyDecision.summary,
				source: "structured",
			};
		}
		// action === "continue" — caller injects follow-up messages
	}

	// 5. User input awaited
	if (input.userInputAwaited) {
		return {
			status: "needs_input",
			summary: "Agent is waiting for the user's answer.",
			source: "heuristic",
		};
	}

	// 6. Reflection failure
	if (input.reflectionFailed) {
		return {
			status: "failed",
			summary: `Agent reached the ${input.maxReflections}-reflection safety limit without completing the task.`,
			source: "heuristic",
		};
	}

	// 7. Max iterations exceeded
	if (input.iteration >= input.maxIterations) {
		return {
			status: "failed",
			summary: `Reached maximum iterations (${input.maxIterations}).`,
			source: "runtime",
		};
	}

	// 8. Heuristic completion — no structured outcome, no stop signals
	if (!input.looksComplete) {
		// Task incomplete but no reflection was triggered (or reflection passed)
		return {
			status: "completed",
			summary: "Agent completed without a structured conclusion.",
			source: "heuristic",
		};
	}

	// 9. No exit needed — continue loop
	return null;
}

// ── Outcome Resolution ─────────────────────────────────────────────────────

export interface OutcomeResolutionInput {
	declared: TaskStatusRecord | null;
	structuredOutcomeRequired: boolean;
	fallbackStatus?: RunOutcomeStatus;
	fallbackSummary?: string;
}

export function resolveOutcome(
	input: OutcomeResolutionInput,
): ExitDecision {
	if (input.declared) {
		return {
			status: input.declared.status === "done" ? "completed" : input.declared.status,
			summary: input.declared.summary,
			source: "structured",
		};
	}
	if (input.structuredOutcomeRequired) {
		return {
			status: "blocked",
			summary: "The run stopped after tool work without a structured task outcome. Resume to verify completion or declare the blocker.",
			source: "runtime",
		};
	}
	return {
		status: input.fallbackStatus ?? "completed",
		summary: input.fallbackSummary,
		source: "heuristic",
	};
}

// ── Post-Turn Evaluation ───────────────────────────────────────────────────

export interface PostTurnResult {
	/** Set to exit the loop. */
	exit?: ExitDecision;
	/** Follow-up messages to inject into the next iteration. */
	followUps?: Message[];
	/** Updated counters for continuation nudges. */
	consecutiveRunnerNudges: number;
	lastRunnerNudgeIteration: number;
	acceptanceReported: boolean;
	acceptanceFailed: boolean;
	acceptanceFinalizationTurns: number;
	reflectionCount: number;
	reflectionFailed: boolean;
	/** Whether the policy decided to continue (for stop policy action === "continue"). */
	policyContinued?: boolean;
}

export interface PostTurnInput {
	followUps: Message[];
	iteration: number;
	maxIterations: number;
	maxReflections: number;
	consecutiveRunnerNudges: number;
	lastRunnerNudgeIteration: number;
	lastToolWorkIteration: number;
	embeddedPoliciesEnabled: boolean;
	continuationEnabled: boolean;
	acceptanceConfig?: ResolvedAcceptance;
	acceptanceReported: boolean;
	acceptanceFailed: boolean;
	acceptanceFinalizationTurns: number;
	reflectionEnabled: boolean;
	reflectionCount: number;
	reflectionFailed: boolean;
	looksComplete: boolean;
	awaitsUserInput: boolean;
	lastAssistantContent: string;
	lastHadToolCalls: boolean;
	hasStructuredStop: boolean;
	performedToolWork: boolean;
	hasTaskStatusTool: boolean;
	stopPolicyDecision?: StopPolicyDecision;
}

export function evaluatePostTurn(input: PostTurnInput): PostTurnResult {
	let consecutiveRunnerNudges = input.consecutiveRunnerNudges;
	let lastRunnerNudgeIteration = input.lastRunnerNudgeIteration;
	let acceptanceReported = input.acceptanceReported;
	let acceptanceFailed = input.acceptanceFailed;
	let acceptanceFinalizationTurns = input.acceptanceFinalizationTurns;
	let reflectionCount = input.reflectionCount;
	let reflectionFailed = input.reflectionFailed;

	// 1. Continuation nudge (structured-conclusion)
	if (
		input.embeddedPoliciesEnabled &&
		input.continuationEnabled &&
		input.followUps.length === 0
	) {
		const hadTools = input.lastHadToolCalls;
		const waitingForUser = input.awaitsUserInput;
		const hasStructuredStop = input.hasStructuredStop;
		const requiresStructuredConclusion =
			input.performedToolWork && input.hasTaskStatusTool;

		if (input.lastToolWorkIteration > input.lastRunnerNudgeIteration) {
			consecutiveRunnerNudges = 0;
		}

		const eligibleForNudge =
			!hadTools &&
			!waitingForUser &&
			!hasStructuredStop &&
			requiresStructuredConclusion;

		const MAX_NUDGES = 3;
		if (eligibleForNudge && consecutiveRunnerNudges < MAX_NUDGES) {
			consecutiveRunnerNudges++;
			lastRunnerNudgeIteration = input.iteration;
			// Caller will inject the nudge message
		} else if (eligibleForNudge && consecutiveRunnerNudges >= MAX_NUDGES) {
			return {
				exit: {
					status: "blocked",
					summary: `Continuation exhausted after ${MAX_NUDGES} nudges without tool progress.`,
					source: "runtime",
				},
				consecutiveRunnerNudges,
				lastRunnerNudgeIteration,
				acceptanceReported,
				acceptanceFailed,
				acceptanceFinalizationTurns,
				reflectionCount,
				reflectionFailed,
			};
		}
		if (!eligibleForNudge) consecutiveRunnerNudges = 0;
	}

	// 2. Follow-ups from hooks or nudges → continue with them
	if (input.followUps.length > 0) {
		return {
			followUps: input.followUps,
			consecutiveRunnerNudges,
			lastRunnerNudgeIteration,
			acceptanceReported,
			acceptanceFailed,
			acceptanceFinalizationTurns,
			reflectionCount,
			reflectionFailed,
		};
	}

	// 3. Stop policies
	if (input.stopPolicyDecision) {
		if (input.stopPolicyDecision.action === "continue") {
			return {
				followUps: input.stopPolicyDecision.messages,
				consecutiveRunnerNudges,
				lastRunnerNudgeIteration,
				acceptanceReported,
				acceptanceFailed,
				acceptanceFinalizationTurns,
				reflectionCount,
				reflectionFailed,
				policyContinued: true,
			};
		}
		if (input.stopPolicyDecision.action === "finish") {
			return {
				exit: {
					status: input.stopPolicyDecision.status,
					summary: input.stopPolicyDecision.summary,
					source: "structured",
				},
				consecutiveRunnerNudges,
				lastRunnerNudgeIteration,
				acceptanceReported,
				acceptanceFailed,
				acceptanceFinalizationTurns,
				reflectionCount,
				reflectionFailed,
			};
		}
	}

	// 4. Acceptance finalization (in-loop retry)
	if (input.acceptanceConfig && !acceptanceReported) {
		// Note: evaluation happens at the caller level; this is just the control flow
		// The actual evaluation is done in agent-loop-runner.ts where we have
		// access to the acceptance report parsing logic
	}

	// 5. Reflection
	if (input.reflectionEnabled && !input.looksComplete) {
		if (reflectionCount >= input.maxReflections) {
			reflectionFailed = true;
			return {
				exit: {
					status: "failed",
					summary: `Agent reached the ${input.maxReflections}-reflection safety limit without completing the task.`,
					source: "heuristic",
				},
				consecutiveRunnerNudges,
				lastRunnerNudgeIteration,
				acceptanceReported,
				acceptanceFailed,
				acceptanceFinalizationTurns,
				reflectionCount,
				reflectionFailed,
			};
		}
		// Caller handles the actual reflection execution
	}

	return {
		consecutiveRunnerNudges,
		lastRunnerNudgeIteration,
		acceptanceReported,
		acceptanceFailed,
		acceptanceFinalizationTurns,
		reflectionCount,
		reflectionFailed,
	};
}
