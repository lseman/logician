import type { RunOutcomeStatus } from "../../core/policy/execution-policy.ts";
import type { TaskStatusRecord } from "./task-status-state.ts";

export interface OutcomeResolutionInput {
	declared: TaskStatusRecord | null;
	structuredOutcomeRequired: boolean;
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
	if (input.structuredOutcomeRequired) {
		return {
			status: "completed",
			summary:
				"Run completed without a declared task outcome. Tool work was performed but no structured outcome was recorded. Review the final text for correctness.",
			source: "runtime",
		};
	}
	return {
		status: input.fallbackStatus ?? "completed",
		summary: input.fallbackSummary,
		source: "heuristic",
	};
}
