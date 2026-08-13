import type { RunOutcomeStatus } from "../execution-policy.ts";
import type { TaskStatusRecord } from "./task-status-state.ts";

export interface CompletionGateInput {
	declared: TaskStatusRecord | null;
	structuredOutcomeRequired: boolean;
	fallbackStatus?: RunOutcomeStatus;
	fallbackSummary?: string;
}

export interface CompletionGateDecision {
	status: RunOutcomeStatus;
	summary?: string;
	source: "structured" | "heuristic" | "runtime";
}

/** Authoritative terminal resolution shared by every loop exit path. */
export function resolveCompletionGate(
	input: CompletionGateInput,
): CompletionGateDecision {
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
			status: "blocked",
			summary:
				"The run stopped after tool work without a structured task outcome. Resume to verify completion or declare the blocker.",
			source: "runtime",
		};
	}
	return {
		status: input.fallbackStatus ?? "completed",
		summary: input.fallbackSummary,
		source: "heuristic",
	};
}
