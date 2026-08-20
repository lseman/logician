import type { RunOutcomeStatus } from "../../core/policy/execution-policy.ts";
import type { TaskStatusRecord } from "./task-status-state.ts";

export interface OutcomeResolutionInput {
	declared: TaskStatusRecord | null;
	fallbackStatus?: RunOutcomeStatus;
	fallbackSummary?: string;
}

export interface OutcomeDecision {
	status: RunOutcomeStatus;
	summary?: string;
	source: "structured" | "heuristic" | "runtime";
}

/** Resolve every normal loop completion through one small terminal contract. */
export function resolveOutcome(input: OutcomeResolutionInput): OutcomeDecision {
	if (input.declared) {
		return {
			status:
				input.declared.status === "done" ? "completed" : input.declared.status,
			summary: input.declared.summary,
			source: "structured",
		};
	}
	return {
		status: input.fallbackStatus ?? "completed",
		summary: input.fallbackSummary,
		source: "heuristic",
	};
}
