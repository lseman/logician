import type { Message } from "./types/types-messages.ts";

export type RunOutcomeStatus =
	| "completed"
	| "needs_input"
	| "blocked"
	| "failed"
	| "cancelled";

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
	policies: readonly AgentStopPolicy[] | undefined,
	context: StopPolicyContext,
): Promise<StopPolicyDecision | undefined> {
	for (const policy of policies ?? []) {
		const decision = await policy(context);
		if (decision) return decision;
	}
	return undefined;
}
