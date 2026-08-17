import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	initialRunKernelState,
	isRunEventEnvelope,
	type RunEventEnvelope,
	type RunKernelEvent,
	reduceRunKernel,
	replayRunKernel,
} from "../agent/core/run-kernel-events.ts";

void test("runtime validation rejects unknown and malformed event payloads", () => {
	assert.equal(
		isRunEventEnvelope({
			schemaVersion: 1,
			sequence: 1,
			eventId: "event",
			sessionId: "session",
			taskId: "task",
			runId: "run",
			leaseEpoch: 0,
			timestamp: 1,
			event: { type: "future_unknown_event" },
		}),
		false,
	);
	assert.equal(
		isRunEventEnvelope({
			schemaVersion: 1,
			sequence: 1,
			eventId: "event",
			sessionId: "session",
			taskId: "task",
			runId: "run",
			leaseEpoch: 0,
			timestamp: 1,
			event: {
				type: "budget_consumed",
				resource: "provider_call",
				amount: "one",
			},
		}),
		false,
	);
});

function stream(events: RunKernelEvent[]): RunEventEnvelope[] {
	return events.map((event, index) => ({
		schemaVersion: 1,
		sequence: index + 1,
		eventId: `event-${index + 1}`,
		sessionId: "session-a",
		taskId: "task-a",
		runId: "run-a",
		operationId: "operationId" in event ? event.operationId : undefined,
		leaseEpoch: 1,
		timestamp: 100 + index,
		event,
	}));
}

void test("kernel deterministically replays task, budgets, operation, and terminal state", () => {
	const replay = replayRunKernel(
		stream([
			{ type: "task_started", rootPrompt: "ship it", createdAt: 100 },
			{ type: "run_started", cause: "prompt" },
			{ type: "budget_consumed", resource: "provider_call", amount: 1 },
			{ type: "budget_consumed", resource: "tool_call", amount: 1 },
			{
				type: "operation_intent_recorded",
				operationId: "op-1",
				toolName: "apply_patch",
				argumentsDigest: "args-sha256",
				idempotencyKey: "task-a:1",
				recovery: "at_most_once_unknown",
			},
			{
				type: "operation_result_recorded",
				operationId: "op-1",
				resultDigest: "result-sha256",
				isError: false,
			},
			{ type: "operation_committed", operationId: "op-1" },
			{
				type: "permission_decided",
				toolCallId: "call-1",
				toolName: "apply_patch",
				decision: "allow",
				source: "user",
				scope: "session",
			},
			{
				type: "subagent_started",
				agentId: "child-1",
				agent: "reviewer",
				task: "review",
			},
			{
				type: "subagent_progressed",
				agentId: "child-1",
				eventType: "tool_execution_end",
			},
			{
				type: "subagent_finished",
				agentId: "child-1",
				agent: "reviewer",
				result: "looks good",
				isError: false,
				turns: 2,
			},
			{ type: "compaction_committed", generation: 1 },
			{ type: "run_finished", status: "completed" },
		]),
	);
	assert.deepEqual(replay.violations, []);
	assert.equal(replay.state.status, "completed");
	assert.equal(replay.state.budgets.provider_call, 1);
	assert.equal(replay.state.budgets.tool_call, 1);
	assert.equal(replay.state.operations["op-1"]?.status, "committed");
	assert.equal(replay.state.permissionDecisions[0]?.scope, "session");
	assert.equal(replay.state.subagents["child-1"]?.status, "completed");
	assert.equal(
		replay.state.subagents["child-1"]?.lastEventType,
		"tool_execution_end",
	);
	assert.equal(replay.state.compactionGeneration, 1);
});

void test("kernel rejects stale leases and preserves the last valid projection", () => {
	const [started] = stream([
		{ type: "task_started", rootPrompt: "safe", createdAt: 100 },
	]);
	assert.ok(started);
	const first = reduceRunKernel(initialRunKernelState(), started);
	const stale: RunEventEnvelope = {
		...started,
		sequence: 2,
		eventId: "event-2",
		leaseEpoch: 0,
		event: { type: "run_started", cause: "resume" },
	};
	const result = reduceRunKernel(first.state, stale);
	assert.equal(result.violations[0]?.code, "stale_lease");
	assert.strictEqual(result.state, first.state);
});

void test("kernel exposes crash frontiers without guessing operation outcomes", () => {
	const replay = replayRunKernel(
		stream([
			{ type: "task_started", rootPrompt: "recover", createdAt: 100 },
			{
				type: "operation_intent_recorded",
				operationId: "op-unknown",
				toolName: "external_write",
				argumentsDigest: "digest",
				idempotencyKey: "key",
				recovery: "at_most_once_unknown",
			},
		]),
	);
	assert.equal(
		replay.state.operations["op-unknown"]?.status,
		"intent_recorded",
	);
	assert.equal(
		replay.state.operations["op-unknown"]?.recovery,
		"at_most_once_unknown",
	);
});

void test("kernel rejects illegal operation transitions and events after terminal state", () => {
	const events = stream([
		{ type: "task_started", rootPrompt: "guard", createdAt: 100 },
		{
			type: "operation_intent_recorded",
			operationId: "op-1",
			toolName: "read",
			argumentsDigest: "digest",
			idempotencyKey: "key",
			recovery: "pure",
		},
		{ type: "operation_committed", operationId: "op-1" },
	]);
	const replay = replayRunKernel(events);
	assert.equal(replay.violations[0]?.code, "invalid_operation_transition");
	assert.equal(replay.state.lastSequence, 2);

	const terminal = replayRunKernel(
		stream([
			{ type: "task_started", rootPrompt: "done", createdAt: 100 },
			{ type: "run_finished", status: "cancelled" },
			{ type: "budget_consumed", resource: "provider_call", amount: 1 },
		]),
	);
	assert.equal(terminal.violations[0]?.code, "event_after_terminal");
	assert.equal(terminal.state.budgets.provider_call, 0);
	assert.equal(terminal.state.lastSequence, 2);
});
