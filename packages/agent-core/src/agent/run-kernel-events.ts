import type { RunOutcomeStatus } from "./execution-policy.ts";
import type { HarnessIntervention } from "./intervention-controller.ts";

export const RUN_KERNEL_SCHEMA_VERSION = 1 as const;

/** @deprecated Use `RunOutcomeStatus` from execution-policy.ts directly. */
export type RunTerminalStatus = RunOutcomeStatus;

export type RunOperationRecovery =
	| "pure"
	| "idempotent"
	| "receipt_recoverable"
	| "at_most_once_unknown";

export type RunKernelEvent =
	| {
		type: "task_started";
		rootPrompt: string;
		createdAt: number;
		progressFingerprint?: string;
	}
	| { type: "run_started"; cause: "prompt" | "continue" | "resume" }
	| { type: "lease_acquired"; ownerId: string; expiresAt: number }
	| {
		type: "continuation_requested";
		cause: string;
		progressFingerprint: string;
	}
	| { type: "intervention_recorded"; intervention: HarnessIntervention }
	| {
		type: "budget_consumed";
		resource: "provider_call" | "tool_call" | "token" | "cost_microusd";
		amount: number;
	}
	| {
		type: "operation_intent_recorded";
		operationId: string;
		toolCallId?: string;
		toolName: string;
		arguments?: Record<string, unknown>;
		argumentsDigest: string;
		idempotencyKey: string;
		recovery: RunOperationRecovery;
	}
	| {
		type: "operation_result_recorded";
		operationId: string;
		resultDigest: string;
		result?: string;
		isError: boolean;
		receipt?: string;
	}
	| {
		type: "permission_decided";
		toolCallId: string;
		toolName: string;
		decision: "allow" | "deny";
		source: "rule" | "mode" | "user" | "fail_closed";
		scope?: "once" | "session";
		approvalRule?: string;
	}
	| { type: "operation_committed"; operationId: string }
	| { type: "operation_quarantined"; operationId: string; reason: string }
	| { type: "compaction_committed"; generation: number }
	| {
		type: "queue_updated";
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	}
	| {
		type: "subagent_started";
		agentId: string;
		agent: string;
		task: string;
		taskIndex?: number;
	}
	| { type: "subagent_progressed"; agentId: string; eventType: string }
	| {
		type: "subagent_finished";
		agentId: string;
		agent: string;
		result: string;
		isError: boolean;
		turns?: number;
	}
	| {
		type: "run_finished";
		status: RunTerminalStatus;
		summary?: string;
		source?: "structured" | "heuristic" | "runtime";
	}
	| {
		type: "trajectory_recorded";
		kind: "run_start" | "agent_event" | "run_finish";
		operationId: string;
		payload: Record<string, unknown>;
	}
	| { type: "diagnostic_recorded"; code: string; message: string };

export interface RunEventEnvelope {
	schemaVersion: typeof RUN_KERNEL_SCHEMA_VERSION;
	sequence: number;
	eventId: string;
	sessionId: string;
	taskId: string;
	runId: string;
	operationId?: string;
	leaseEpoch: number;
	timestamp: number;
	event: RunKernelEvent;
}

export interface RunKernelOperation {
	operationId: string;
	toolCallId?: string;
	toolName: string;
	arguments?: Record<string, unknown>;
	argumentsDigest: string;
	idempotencyKey: string;
	recovery: RunOperationRecovery;
	status: "intent_recorded" | "result_recorded" | "committed" | "quarantined";
	resultDigest?: string;
	result?: string;
	isError?: boolean;
	receipt?: string;
	quarantineReason?: string;
}

export interface RunKernelState {
	schemaVersion: typeof RUN_KERNEL_SCHEMA_VERSION;
	sessionId?: string;
	taskId?: string;
	runId?: string;
	rootPrompt?: string;
	createdAt?: number;
	updatedAt?: number;
	lastSequence: number;
	leaseEpoch: number;
	leaseOwnerId?: string;
	leaseExpiresAt?: number;
	status: "idle" | "active" | RunTerminalStatus;
	continuationRuns: number;
	noProgressRuns: number;
	lastProgressFingerprint: string;
	lastCause: string;
	interventions: HarnessIntervention[];
	budgets: Record<
		"provider_call" | "tool_call" | "token" | "cost_microusd",
		number
	>;
	operations: Record<string, RunKernelOperation>;
	permissionDecisions: Array<{
		toolCallId: string;
		toolName: string;
		decision: "allow" | "deny";
		source: "rule" | "mode" | "user" | "fail_closed";
		scope?: "once" | "session";
		approvalRule?: string;
		sequence: number;
	}>;
	compactionGeneration: number;
	queues: { steering: string[]; followUp: string[]; nextTurn: string[] };
	subagents: Record<
		string,
		{
			agentId: string;
			agent: string;
			task: string;
			taskIndex?: number;
			status: "running" | "completed" | "failed";
			lastEventType?: string;
			result?: string;
			turns?: number;
		}
	>;
	outcome?: {
		status: RunTerminalStatus;
		summary?: string;
		source?: "structured" | "heuristic" | "runtime";
	};
	terminalReason?: string;
	trajectory: Array<{
		sequence: number;
		timestamp: number;
		runId: string;
		operationId: string;
		kind: "run_start" | "agent_event" | "run_finish";
		payload: Record<string, unknown>;
	}>;
	diagnostics: Array<{ code: string; message: string; sequence: number }>;
}

export interface RunKernelViolation {
	code:
	| "invalid_envelope"
	| "sequence_gap"
	| "identity_changed"
	| "stale_lease"
	| "lease_not_acquired"
	| "lease_expired"
	| "event_after_terminal"
	| "task_not_started"
	| "duplicate_operation"
	| "operation_not_found"
	| "invalid_operation_transition"
	| "invalid_budget_amount"
	| "invalid_compaction_generation"
	| "subagent_not_found";
	message: string;
	sequence?: number;
}

export interface RunKernelReduction {
	state: RunKernelState;
	violations: RunKernelViolation[];
}

export function initialRunKernelState(): RunKernelState {
	return {
		schemaVersion: RUN_KERNEL_SCHEMA_VERSION,
		lastSequence: 0,
		leaseEpoch: 0,
		status: "idle",
		continuationRuns: 0,
		noProgressRuns: 0,
		lastProgressFingerprint: "",
		lastCause: "user_prompt",
		interventions: [],
		budgets: { provider_call: 0, tool_call: 0, token: 0, cost_microusd: 0 },
		operations: {},
		permissionDecisions: [],
		compactionGeneration: 0,
		queues: { steering: [], followUp: [], nextTurn: [] },
		subagents: {},
		trajectory: [],
		diagnostics: [],
	};
}

function isObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
	return typeof value === "string";
}

function isFiniteNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

export function isRunKernelEvent(value: unknown): value is RunKernelEvent {
	if (!isObject(value) || !isString(value.type)) return false;
	switch (value.type) {
		case "task_started":
			return (
				isString(value.rootPrompt) &&
				isFiniteNumber(value.createdAt) &&
				(value.progressFingerprint === undefined ||
					isString(value.progressFingerprint))
			);
		case "run_started":
			return (
				value.cause === "prompt" ||
				value.cause === "continue" ||
				value.cause === "resume"
			);
		case "lease_acquired":
			return isString(value.ownerId) && isFiniteNumber(value.expiresAt);
		case "continuation_requested":
			return isString(value.cause) && isString(value.progressFingerprint);
		case "intervention_recorded":
			return (
				isObject(value.intervention) &&
				isString(value.intervention.id) &&
				isString(value.intervention.kind) &&
				isString(value.intervention.action) &&
				isFiniteNumber(value.intervention.attempt)
			);
		case "budget_consumed":
			return (
				["provider_call", "tool_call", "token", "cost_microusd"].includes(
					String(value.resource),
				) && isFiniteNumber(value.amount)
			);
		case "operation_intent_recorded":
			return (
				isString(value.operationId) &&
				(value.toolCallId === undefined || isString(value.toolCallId)) &&
				isString(value.toolName) &&
				(value.arguments === undefined || isObject(value.arguments)) &&
				isString(value.argumentsDigest) &&
				isString(value.idempotencyKey) &&
				[
					"pure",
					"idempotent",
					"receipt_recoverable",
					"at_most_once_unknown",
				].includes(String(value.recovery))
			);
		case "operation_result_recorded":
			return (
				isString(value.operationId) &&
				isString(value.resultDigest) &&
				(value.result === undefined || isString(value.result)) &&
				typeof value.isError === "boolean" &&
				(value.receipt === undefined || isString(value.receipt))
			);
		case "permission_decided":
			return (
				isString(value.toolCallId) &&
				isString(value.toolName) &&
				["allow", "deny"].includes(String(value.decision)) &&
				["rule", "mode", "user", "fail_closed"].includes(
					String(value.source),
				) &&
				(value.scope === undefined ||
					["once", "session"].includes(String(value.scope)))
			);
		case "operation_committed":
			return isString(value.operationId);
		case "operation_quarantined":
			return isString(value.operationId) && isString(value.reason);
		case "compaction_committed":
			return (
				Number.isSafeInteger(value.generation) &&
				(value.generation as number) > 0
			);
		case "queue_updated":
			return (
				Array.isArray(value.steering) &&
				value.steering.every(isString) &&
				Array.isArray(value.followUp) &&
				value.followUp.every(isString) &&
				Array.isArray(value.nextTurn) &&
				value.nextTurn.every(isString)
			);
		case "subagent_started":
			return (
				isString(value.agentId) &&
				isString(value.agent) &&
				isString(value.task) &&
				(value.taskIndex === undefined || Number.isSafeInteger(value.taskIndex))
			);
		case "subagent_progressed":
			return isString(value.agentId) && isString(value.eventType);
		case "subagent_finished":
			return (
				isString(value.agentId) &&
				isString(value.agent) &&
				isString(value.result) &&
				typeof value.isError === "boolean" &&
				(value.turns === undefined || Number.isSafeInteger(value.turns))
			);
		case "run_finished":
			return (
				["completed", "needs_input", "blocked", "failed", "cancelled"].includes(
					String(value.status),
				) &&
				(value.summary === undefined || isString(value.summary)) &&
				(value.source === undefined ||
					["structured", "heuristic", "runtime"].includes(String(value.source)))
			);
		case "trajectory_recorded":
			return (
				["run_start", "agent_event", "run_finish"].includes(
					String(value.kind),
				) &&
				isString(value.operationId) &&
				isObject(value.payload)
			);
		case "diagnostic_recorded":
			return isString(value.code) && isString(value.message);
		default:
			return false;
	}
}

export function isRunEventEnvelope(value: unknown): value is RunEventEnvelope {
	if (!isObject(value) || !isRunKernelEvent(value.event)) return false;
	return (
		value.schemaVersion === RUN_KERNEL_SCHEMA_VERSION &&
		Number.isSafeInteger(value.sequence) &&
		(value.sequence as number) > 0 &&
		typeof value.eventId === "string" &&
		typeof value.sessionId === "string" &&
		typeof value.taskId === "string" &&
		typeof value.runId === "string" &&
		Number.isSafeInteger(value.leaseEpoch) &&
		(value.leaseEpoch as number) >= 0 &&
		isFiniteNumber(value.timestamp)
	);
}

const TERMINAL = new Set<RunKernelState["status"]>([
	"completed",
	"needs_input",
	"blocked",
	"failed",
	"cancelled",
]);

/** Pure, non-throwing reducer. Invalid events leave the prior state unchanged. */
export function reduceRunKernel(
	prior: RunKernelState,
	envelope: RunEventEnvelope,
): RunKernelReduction {
	const violations: RunKernelViolation[] = [];
	const reject = (
		code: RunKernelViolation["code"],
		message: string,
	): RunKernelReduction => ({
		state: prior,
		violations: [{ code, message, sequence: envelope.sequence }],
	});
	if (!isRunEventEnvelope(envelope))
		return reject(
			"invalid_envelope",
			"event envelope failed schema validation",
		);
	if (envelope.sequence !== prior.lastSequence + 1)
		return reject(
			"sequence_gap",
			`expected sequence ${prior.lastSequence + 1}, received ${envelope.sequence}`,
		);
	if (prior.sessionId && prior.sessionId !== envelope.sessionId)
		return reject(
			"identity_changed",
			"session identity changed within a stream",
		);
	if (
		prior.taskId &&
		prior.taskId !== envelope.taskId &&
		envelope.event.type !== "task_started"
	)
		return reject(
			"identity_changed",
			"task identity changed without task_started",
		);
	if (envelope.leaseEpoch < prior.leaseEpoch)
		return reject(
			"stale_lease",
			`lease epoch ${envelope.leaseEpoch} is older than ${prior.leaseEpoch}`,
		);
	if (
		envelope.leaseEpoch > prior.leaseEpoch &&
		envelope.event.type !== "task_started" &&
		envelope.event.type !== "lease_acquired"
	)
		return reject(
			"lease_not_acquired",
			"a higher fencing epoch requires lease_acquired",
		);
	if (
		prior.leaseExpiresAt !== undefined &&
		envelope.timestamp > prior.leaseExpiresAt &&
		envelope.event.type !== "lease_acquired" &&
		envelope.event.type !== "task_started" &&
		envelope.event.type !== "diagnostic_recorded"
	)
		return reject("lease_expired", `lease expired at ${prior.leaseExpiresAt}`);
	if (
		TERMINAL.has(prior.status) &&
		envelope.event.type !== "diagnostic_recorded" &&
		envelope.event.type !== "trajectory_recorded" &&
		envelope.event.type !== "queue_updated" &&
		envelope.event.type !== "compaction_committed" &&
		envelope.event.type !== "lease_acquired" &&
		envelope.event.type !== "task_started" &&
		!(
			prior.status === "completed" &&
			envelope.event.type === "continuation_requested"
		)
	)
		return reject(
			"event_after_terminal",
			`cannot apply ${envelope.event.type} after ${prior.status}`,
		);
	if (!prior.taskId && envelope.event.type !== "task_started")
		return reject("task_not_started", "the first event must start the task");

	const state = structuredClone(prior);
	state.sessionId ??= envelope.sessionId;
	state.taskId ??= envelope.taskId;
	if (
		envelope.event.type !== "trajectory_recorded" &&
		envelope.event.type !== "diagnostic_recorded"
	)
		state.runId = envelope.runId;
	state.lastSequence = envelope.sequence;
	state.leaseEpoch = Math.max(state.leaseEpoch, envelope.leaseEpoch);
	state.updatedAt = envelope.timestamp;
	const event = envelope.event;

	if (event.type === "task_started") {
		if (prior.taskId && !TERMINAL.has(prior.status))
			return reject(
				"identity_changed",
				"an active task must finish before another starts",
			);
		state.taskId = envelope.taskId;
		state.rootPrompt = event.rootPrompt;
		state.createdAt = event.createdAt;
		state.status = "active";
		state.continuationRuns = 0;
		state.noProgressRuns = 0;
		state.lastProgressFingerprint = event.progressFingerprint ?? "";
		state.lastCause = "user_prompt";
		state.interventions = [];
		state.budgets = {
			provider_call: 0,
			tool_call: 0,
			token: 0,
			cost_microusd: 0,
		};
		state.operations = {};
		state.permissionDecisions = [];
		state.compactionGeneration = 0;
		state.queues = { steering: [], followUp: [], nextTurn: [] };
		state.subagents = {};
		state.outcome = undefined;
		state.terminalReason = undefined;
	} else if (event.type === "run_started") {
		state.status = "active";
		state.lastCause = event.cause;
	} else if (event.type === "lease_acquired") {
		if (
			prior.leaseOwnerId &&
			prior.leaseOwnerId !== event.ownerId &&
			(prior.leaseExpiresAt ?? 0) >= envelope.timestamp &&
			envelope.leaseEpoch === prior.leaseEpoch
		)
			return reject(
				"lease_not_acquired",
				`live lease is owned by ${prior.leaseOwnerId}`,
			);
		state.leaseOwnerId = event.ownerId;
		state.leaseExpiresAt = event.expiresAt;
	} else if (event.type === "continuation_requested") {
		state.status = "active";
		state.continuationRuns++;
		state.lastCause = event.cause;
		if (
			event.progressFingerprint &&
			event.progressFingerprint !== state.lastProgressFingerprint
		) {
			state.lastProgressFingerprint = event.progressFingerprint;
			state.noProgressRuns = 0;
		} else state.noProgressRuns++;
	} else if (event.type === "intervention_recorded") {
		state.interventions.push(structuredClone(event.intervention));
	} else if (event.type === "budget_consumed") {
		if (!Number.isFinite(event.amount) || event.amount <= 0)
			return reject(
				"invalid_budget_amount",
				"budget consumption must be finite and positive",
			);
		state.budgets[event.resource] += event.amount;
	} else if (event.type === "operation_intent_recorded") {
		if (state.operations[event.operationId])
			return reject(
				"duplicate_operation",
				`operation ${event.operationId} already exists`,
			);
		state.operations[event.operationId] = {
			operationId: event.operationId,
			toolCallId: event.toolCallId,
			toolName: event.toolName,
			arguments: event.arguments ? structuredClone(event.arguments) : undefined,
			argumentsDigest: event.argumentsDigest,
			idempotencyKey: event.idempotencyKey,
			recovery: event.recovery,
			status: "intent_recorded",
		};
	} else if (event.type === "permission_decided") {
		state.permissionDecisions.push({
			toolCallId: event.toolCallId,
			toolName: event.toolName,
			decision: event.decision,
			source: event.source,
			scope: event.scope,
			approvalRule: event.approvalRule,
			sequence: envelope.sequence,
		});
	} else if (event.type === "operation_result_recorded") {
		const operation = state.operations[event.operationId];
		if (!operation)
			return reject(
				"operation_not_found",
				`operation ${event.operationId} was not started`,
			);
		if (operation.status !== "intent_recorded")
			return reject(
				"invalid_operation_transition",
				`cannot record result from ${operation.status}`,
			);
		operation.status = "result_recorded";
		operation.resultDigest = event.resultDigest;
		operation.result = event.result;
		operation.isError = event.isError;
		operation.receipt = event.receipt;
	} else if (event.type === "operation_committed") {
		const operation = state.operations[event.operationId];
		if (!operation)
			return reject(
				"operation_not_found",
				`operation ${event.operationId} was not started`,
			);
		if (operation.status !== "result_recorded")
			return reject(
				"invalid_operation_transition",
				`cannot commit from ${operation.status}`,
			);
		operation.status = "committed";
	} else if (event.type === "operation_quarantined") {
		const operation = state.operations[event.operationId];
		if (!operation)
			return reject(
				"operation_not_found",
				`operation ${event.operationId} was not started`,
			);
		if (operation.status === "committed")
			return reject(
				"invalid_operation_transition",
				"a committed operation cannot be quarantined",
			);
		operation.status = "quarantined";
		operation.quarantineReason = event.reason;
	} else if (event.type === "compaction_committed") {
		if (event.generation !== state.compactionGeneration + 1)
			return reject(
				"invalid_compaction_generation",
				`expected compaction generation ${state.compactionGeneration + 1}`,
			);
		state.compactionGeneration = event.generation;
	} else if (event.type === "queue_updated") {
		state.queues = {
			steering: [...event.steering],
			followUp: [...event.followUp],
			nextTurn: [...event.nextTurn],
		};
	} else if (event.type === "subagent_started") {
		state.subagents[event.agentId] = {
			agentId: event.agentId,
			agent: event.agent,
			task: event.task,
			taskIndex: event.taskIndex,
			status: "running",
		};
	} else if (event.type === "subagent_progressed") {
		const child = state.subagents[event.agentId];
		if (!child)
			return reject(
				"subagent_not_found",
				`subagent ${event.agentId} was not started`,
			);
		child.lastEventType = event.eventType;
	} else if (event.type === "subagent_finished") {
		const child = state.subagents[event.agentId];
		state.subagents[event.agentId] = {
			agentId: event.agentId,
			agent: event.agent,
			task: child?.task ?? "restored delegated task",
			taskIndex: child?.taskIndex,
			status: event.isError ? "failed" : "completed",
			lastEventType: child?.lastEventType,
			result: event.result,
			turns: event.turns,
		};
	} else if (event.type === "run_finished") {
		state.status = event.status;
		state.outcome = {
			status: event.status,
			summary: event.summary,
			source: event.source,
		};
		state.terminalReason = event.summary;
	} else if (event.type === "trajectory_recorded") {
		state.trajectory.push({
			sequence: envelope.sequence,
			timestamp: envelope.timestamp,
			runId: envelope.runId,
			operationId: event.operationId,
			kind: event.kind,
			payload: structuredClone(event.payload),
		});
	} else if (event.type === "diagnostic_recorded") {
		state.diagnostics.push({
			code: event.code,
			message: event.message,
			sequence: envelope.sequence,
		});
	}
	return { state, violations };
}

export function replayRunKernel(
	events: readonly RunEventEnvelope[],
): RunKernelReduction {
	let state = initialRunKernelState();
	const violations: RunKernelViolation[] = [];
	for (const event of events) {
		const next = reduceRunKernel(state, event);
		violations.push(...next.violations);
		state = next.state;
	}
	return { state, violations };
}
