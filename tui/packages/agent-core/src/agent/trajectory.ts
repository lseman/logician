export interface TrajectoryEntry {
	version: 1;
	sequence: number;
	timestamp: number;
	sessionId: string;
	runId: string;
	operationId: string;
	kind: "run_start" | "agent_event" | "run_finish";
	payload: Record<string, unknown>;
}

export interface TrajectoryReport {
	events: number;
	durationMs: number;
	providerRetries: number;
	toolCalls: number;
	toolFailures: number;
	loopEscapes: number;
	compactions: number;
	permissionRequests: number;
	permissionDenials: number;
	subagentRuns: number;
	subagentFailures: number;
	acceptancePassed: boolean;
	prematureStop: boolean;
	replayComplete: boolean;
}

/** Pure projection used by diagnostics and evaluation; persistence belongs to RunKernel. */
export function evaluateTrajectory(
	entries: TrajectoryEntry[],
): TrajectoryReport {
	const events = entries.filter(entry => entry.kind === "agent_event");
	const eventTypes = events.map(entry => entry.payload.type);
	const outcome = [...events]
		.reverse()
		.find(entry => entry.payload.type === "run_outcome")?.payload;
	const taskState = [...events]
		.reverse()
		.find(entry => entry.payload.type === "task_state_update")?.payload.state as
		| {
				phase?: string;
				blockers?: string[];
				verification?: Array<{ passed: boolean }>;
		  }
		| undefined;
	const first = entries[0]?.timestamp ?? 0;
	const last = entries.at(-1)?.timestamp ?? first;
	const finished = entries.some(entry => entry.kind === "run_finish");
	const acceptancePassed =
		outcome?.status === "completed" &&
		(taskState?.blockers?.length ?? 0) === 0 &&
		(taskState?.verification?.every(item => item.passed) ?? true);
	return {
		events: events.length,
		durationMs: Math.max(0, last - first),
		providerRetries: eventTypes.filter(type => type === "agent_retry_start")
			.length,
		toolCalls: eventTypes.filter(type => type === "tool_execution_start")
			.length,
		toolFailures: events.filter(
			entry =>
				entry.payload.type === "tool_execution_end" &&
				entry.payload.isError === true,
		).length,
		loopEscapes: eventTypes.filter(
			type => type === "loop_detected" || type === "harness_intervention",
		).length,
		compactions: eventTypes.filter(type => type === "compaction").length,
		permissionRequests: eventTypes.filter(
			type => type === "tool_permission_request",
		).length,
		permissionDenials: events.filter(
			entry =>
				entry.payload.type === "tool_permission_decision" &&
				entry.payload.decision === "deny",
		).length,
		subagentRuns: eventTypes.filter(type => type === "subagent_start").length,
		subagentFailures: events.filter(
			entry =>
				entry.payload.type === "subagent_end" && entry.payload.isError === true,
		).length,
		acceptancePassed,
		prematureStop:
			outcome?.status === "completed" &&
			Boolean(taskState) &&
			taskState?.phase !== "handoff" &&
			!acceptancePassed,
		replayComplete: entries.length === 0 || finished,
	};
}
