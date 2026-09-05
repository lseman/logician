import type { RunOutcomeStatus } from "../../system/types/execution-policy.ts";
import type { Message } from "../../system/types/types-messages.ts";

export interface StopPolicyContext {
	/** Full active transcript, including the system message. */
	messages: readonly Message[];
	/** Messages produced during the current run. */
	newMessages: readonly Message[];
	iteration: number;
	signal?: AbortSignal;
}

export type StopPolicyDecision =
	| {
			action: "continue";
			messages: Message[];
	  }
	| {
			action: "finish";
			status: RunOutcomeStatus;
			summary?: string;
	  };

/**
 * Optional policy evaluated when the mechanism has no pending tool calls,
 * steering messages, or follow-up messages.
 */
export type AgentStopPolicy = (
	context: StopPolicyContext,
) => Promise<StopPolicyDecision | undefined> | StopPolicyDecision | undefined;

export type StopPolicyKind = "deterministic" | "prompt" | "agent";

export interface NamedAgentStopPolicy {
	id: string;
	description: string;
	kind: StopPolicyKind;
	evaluate: AgentStopPolicy;
}

export interface StopPolicyEvaluation {
	policyId: string;
	kind: StopPolicyKind;
	status: "started" | "completed" | "failed";
	durationMs?: number;
	decision?: StopPolicyDecision["action"] | "abstain";
	error?: string;
}

export type StopPolicy = AgentStopPolicy | NamedAgentStopPolicy;

export type ExecutionProfile = "autonomous" | "minimal";

export interface ResolvedExecutionPolicy {
	profile: ExecutionProfile;
	embeddedPoliciesEnabled: boolean;
}

export function resolveExecutionPolicy(
	profile: ExecutionProfile | undefined,
): ResolvedExecutionPolicy {
	const resolvedProfile = profile ?? "autonomous";
	return {
		profile: resolvedProfile,
		embeddedPoliciesEnabled: resolvedProfile === "autonomous",
	};
}

export async function evaluateStopPolicies(
	policies: readonly StopPolicy[] | undefined,
	context: StopPolicyContext,
	onEvaluation?: (evaluation: StopPolicyEvaluation) => Promise<void> | void,
): Promise<StopPolicyDecision | undefined> {
	for (const candidate of policies ?? []) {
		const policy =
			typeof candidate === "function"
				? {
						id: "anonymous-stop-policy",
						kind: "deterministic" as const,
						evaluate: candidate,
					}
				: candidate;
		const started = performance.now();
		await onEvaluation?.({
			policyId: policy.id,
			kind: policy.kind,
			status: "started",
		});
		try {
			const decision = await policy.evaluate(context);
			await onEvaluation?.({
				policyId: policy.id,
				kind: policy.kind,
				status: "completed",
				durationMs: performance.now() - started,
				decision: decision?.action ?? "abstain",
			});
			if (decision) return decision;
		} catch (error) {
			await onEvaluation?.({
				policyId: policy.id,
				kind: policy.kind,
				status: "failed",
				durationMs: performance.now() - started,
				error: error instanceof Error ? error.message : String(error),
			});
		}
	}
	return undefined;
}
