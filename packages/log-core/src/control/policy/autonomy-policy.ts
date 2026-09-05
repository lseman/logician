export interface AutonomousTask {
	id: string | number;
	subject: string;
	status: string;
}

export interface AutonomousContinuationInput {
	assistantText: string;
	stopReason?: string;
	tasks: readonly AutonomousTask[];
}

export type AutonomousContinuation = {
	reason: "length_truncation" | "unfinished_todos";
	message: string;
};

/**
 * Decide whether an otherwise-finished turn needs one more model call.
 *
 * Handles:
 * - Provider truncation recovery
 * - Unfinished todo continuation
 *
 * User handoff detection is handled by the steering mechanism (live user
 * input pauses the loop). Text-parsing heuristics for "awaitsUserInput"
 * are removed — they produce unreliable results and OMP does not use them.
 */
export function decideAutonomousContinuation(
	input: AutonomousContinuationInput,
): AutonomousContinuation | undefined {
	if (input.stopReason === "length") {
		return {
			reason: "length_truncation",
			message:
				"[continuation-nudge:length] Your previous response was cut off because it reached the output limit. " +
				"Continue exactly where you left off without repeating completed work.",
		};
	}

	const remaining = input.tasks.filter(
		task => task.status !== "completed" && task.status !== "deleted",
	);
	if (remaining.length === 0) return undefined;

	const next =
		remaining.find(task => task.status === "in_progress") ?? remaining[0];
	return {
		reason: "unfinished_todos",
		message:
			`[continuation-nudge:todo] You still have ${remaining.length} unfinished task(s). ` +
			`Continue working on #${next.id} ${next.subject}. ` +
			"Keep the todo list accurate as work progresses. If you need user input or are blocked, say so explicitly and stop.",
	};
}
