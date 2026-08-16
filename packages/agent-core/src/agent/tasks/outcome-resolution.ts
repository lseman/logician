import type { RunOutcomeStatus } from "../execution-policy.ts";
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

// ── Completion gate aliases (backward compat) ────────────────────────────────
// These are re-exported from completion-gate.ts for callers that import the
// gate names instead of the core outcome-resolution names.

export type CompletionGateDecision = OutcomeDecision;
export type CompletionGateInput = OutcomeResolutionInput;

export function resolveCompletionGate(
    input: CompletionGateInput,
): CompletionGateDecision {
    return resolveOutcome(input);
}

/** Resolve every normal loop completion through one small terminal contract. */
export function resolveOutcome(
    input: OutcomeResolutionInput,
): OutcomeDecision {
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
