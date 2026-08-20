import type { AgentEvent } from "../types/types-messages.ts";

export interface TrajectorySummary {
	status?: "completed" | "needs_input" | "blocked" | "failed" | "cancelled";
	turns: number;
	toolCalls: number;
	toolErrors: number;
	permissionDenials: number;
	continuations: number;
	interventions: number;
	compactions: number;
	verificationPassed?: boolean;
}

export interface TrajectoryExpectation {
	status?: TrajectorySummary["status"];
	verificationPassed?: boolean;
	maxTurns?: number;
	maxToolErrors?: number;
	maxPermissionDenials?: number;
	maxContinuations?: number;
}

export interface TrajectoryGrade {
	passed: boolean;
	summary: TrajectorySummary;
	failures: string[];
}

/** Reduce a complete event trajectory to stable, outcome-oriented metrics. */
export function summarizeTrajectory(
	events: readonly AgentEvent[],
): TrajectorySummary {
	const summary: TrajectorySummary = {
		turns: 0,
		toolCalls: 0,
		toolErrors: 0,
		permissionDenials: 0,
		continuations: 0,
		interventions: 0,
		compactions: 0,
	};
	for (const event of events) {
		if (event.type === "turn_start") summary.turns++;
		if (event.type === "tool_execution_start") summary.toolCalls++;
		if (event.type === "tool_execution_end" && event.isError)
			summary.toolErrors++;
		if (event.type === "tool_permission_decision" && event.decision === "deny")
			summary.permissionDenials++;
		if (event.type === "harness_intervention") {
			summary.interventions++;
			if (event.kind === "continuation") summary.continuations++;
			if (event.kind === "compaction") summary.compactions++;
		}
		if (event.type === "acceptance_complete")
			summary.verificationPassed = event.status === "passed";
		if (event.type === "agent_end") summary.status = event.status;
	}
	return summary;
}

/** Grade a recorded run without coupling evaluation tasks to the live harness. */
export function gradeTrajectory(
	events: readonly AgentEvent[],
	expectation: TrajectoryExpectation,
): TrajectoryGrade {
	const summary = summarizeTrajectory(events);
	const failures: string[] = [];
	if (expectation.status && summary.status !== expectation.status)
		failures.push(
			`status: expected ${expectation.status}, got ${summary.status}`,
		);
	if (
		expectation.verificationPassed !== undefined &&
		summary.verificationPassed !== expectation.verificationPassed
	)
		failures.push(
			`verification: expected ${expectation.verificationPassed}, got ${summary.verificationPassed}`,
		);
	for (const [label, actual, maximum] of [
		["turns", summary.turns, expectation.maxTurns],
		["tool errors", summary.toolErrors, expectation.maxToolErrors],
		[
			"permission denials",
			summary.permissionDenials,
			expectation.maxPermissionDenials,
		],
		["continuations", summary.continuations, expectation.maxContinuations],
	] as const) {
		if (maximum !== undefined && actual > maximum)
			failures.push(`${label}: maximum ${maximum}, got ${actual}`);
	}
	return { passed: failures.length === 0, summary, failures };
}
