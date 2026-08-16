import { test } from "bun:test";
import assert from "node:assert/strict";
import { appendFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
	createRuntimeState,
	reduceRuntimeState,
	RunKernel,
} from "../agent/run-kernel.ts";

void test("kernel runtime projection tracks streaming and settles cleanly", () => {
	let runtime = createRuntimeState();
	runtime = reduceRuntimeState(runtime, {
		type: "agent_start",
		seq: 1,
		ts: 10,
	});
	runtime = reduceRuntimeState(runtime, {
		type: "message_start",
		turnId: "turn_1",
		role: "assistant",
	});
	runtime = reduceRuntimeState(runtime, {
		type: "text_delta",
		turnId: "turn_1",
		delta: "hello",
	});
	assert.equal(runtime.isStreaming, true);
	assert.equal(runtime.streamingMessage?.content, "hello");

	runtime = reduceRuntimeState(runtime, {
		type: "agent_end",
		messages: [],
		ts: 20,
	});
	assert.equal(runtime.isStreaming, false);
	assert.equal(runtime.streamingMessage, undefined);
	assert.equal(runtime.lastRunDurationMs, 10);
});

void test("kernel journal persists and deterministically restores its projection", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-kernel-"));
	const kernel = new RunKernel(cwd, "session-a");
	const ids = { taskId: "task-a", runId: "run-a", leaseEpoch: 1 };
	kernel.append(
		{ type: "task_started", rootPrompt: "ship", createdAt: 10 },
		ids,
	);
	kernel.append(
		{ type: "budget_consumed", resource: "provider_call", amount: 2 },
		ids,
	);

	const restored = new RunKernel(cwd, "session-a").snapshot();
	assert.equal(restored.state.rootPrompt, "ship");
	assert.equal(restored.state.budgets.provider_call, 2);
	assert.equal(restored.state.lastSequence, 2);
	assert.deepEqual(restored.violations, []);
});

void test("latency-sensitive status reads exclude trajectory payloads", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-kernel-status-"));
	const kernel = new RunKernel(cwd, "session-status");
	const ids = { taskId: "task-status", runId: "run-status", leaseEpoch: 1 };
	kernel.append(
		{ type: "task_started", rootPrompt: "stream", createdAt: 10 },
		ids,
	);
	kernel.recordTrajectory(
		"agent_event",
		"operation-status",
		{ type: "turn_start", large: "x".repeat(10_000) },
		ids.runId,
	);

	const status = kernel.status();
	assert.equal(status.taskId, ids.taskId);
	assert.equal(status.status, "active");
	assert.equal("trajectory" in status, false);
	assert.ok(kernel.budgetStatus());
});

void test("doctor reports a torn tail and recovery semantics for orphaned effects", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-doctor-"));
	const kernel = new RunKernel(cwd, "session-b");
	const ids = { taskId: "task-b", runId: "run-b", leaseEpoch: 3 };
	kernel.append(
		{ type: "task_started", rootPrompt: "recover", createdAt: 10 },
		ids,
	);
	kernel.append(
		{
			type: "operation_intent_recorded",
			operationId: "safe-op",
			toolName: "read",
			argumentsDigest: "a",
			idempotencyKey: "safe",
			recovery: "pure",
		},
		{ ...ids, operationId: "safe-op" },
	);
	kernel.append(
		{
			type: "operation_intent_recorded",
			operationId: "unknown-op",
			toolName: "external-write",
			argumentsDigest: "b",
			idempotencyKey: "unknown",
			recovery: "at_most_once_unknown",
		},
		{ ...ids, operationId: "unknown-op" },
	);
	kernel.append(
		{
			type: "operation_intent_recorded",
			operationId: "result-op",
			toolCallId: "call-result",
			toolName: "lookup",
			arguments: { key: "x" },
			argumentsDigest: "c",
			idempotencyKey: "result-key",
			recovery: "pure",
		},
		{ ...ids, operationId: "result-op" },
	);
	kernel.append(
		{
			type: "operation_result_recorded",
			operationId: "result-op",
			resultDigest: "result-digest",
			result: "saved result",
			isError: false,
		},
		{ ...ids, operationId: "result-op" },
	);
	appendFileSync(kernel.filePath, '{"schemaVersion":1');

	const report = new RunKernel(cwd, "session-b").doctor();
	assert.equal(report.truncatedFinalRecord, true);
	assert.equal(report.lastValidSequence, 5);
	assert.deepEqual(
		report.incompleteOperations.map(item => [
			item.operationId,
			item.recommendedAction,
		]),
		[
			["safe-op", "retry"],
			["unknown-op", "quarantine"],
			["result-op", "reuse_result"],
		],
	);
	assert.deepEqual(report.incompleteOperations[2]?.arguments, { key: "x" });
	assert.equal(report.incompleteOperations[2]?.result, "saved result");
});

void test("kernel refuses to append events from a stale owner", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-fence-"));
	const kernel = new RunKernel(cwd, "session-c");
	kernel.append(
		{ type: "task_started", rootPrompt: "fence", createdAt: 10 },
		{ taskId: "task-c", runId: "run-c", leaseEpoch: 4 },
	);
	assert.throws(
		() =>
			kernel.append(
				{ type: "run_started", cause: "resume" },
				{ taskId: "task-c", runId: "run-c", leaseEpoch: 3 },
			),
		/older than/i,
	);
	assert.equal(kernel.snapshot().state.lastSequence, 1);
});

void test("lease takeover advances fencing epoch and rejects self-promotion", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-lease-"));
	const kernel = new RunKernel(cwd, "session-d");
	const ids = {
		taskId: "task-d",
		runId: "run-d",
		leaseEpoch: 1,
		timestamp: 10,
	};
	kernel.append(
		{ type: "task_started", rootPrompt: "own it", createdAt: 10 },
		ids,
	);
	const first = kernel.acquireLease("worker-a", {
		taskId: "task-d",
		runId: "run-d",
		now: 20,
		ttlMs: 10,
	});
	assert.equal(first.epoch, 2);
	assert.throws(
		() =>
			kernel.acquireLease("worker-b", {
				taskId: "task-d",
				runId: "run-d",
				now: 25,
				ttlMs: 10,
			}),
		/held by worker-a/,
	);
	const takeover = kernel.acquireLease("worker-b", {
		taskId: "task-d",
		runId: "run-d",
		now: 31,
		ttlMs: 10,
	});
	assert.equal(takeover.epoch, 3);
	assert.equal(kernel.snapshot().state.leaseOwnerId, "worker-b");
	assert.throws(
		() =>
			kernel.append(
				{ type: "budget_consumed", resource: "provider_call", amount: 1 },
				{ taskId: "task-d", runId: "run-d", leaseEpoch: 4, timestamp: 32 },
			),
		/requires lease_acquired/,
	);
});

void test("a stale kernel instance observes an external takeover before append", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-takeover-"));
	const first = new RunKernel(cwd, "shared-session");
	first.append(
		{ type: "task_started", rootPrompt: "shared", createdAt: 10 },
		{ taskId: "task", runId: "run", leaseEpoch: 1, timestamp: 10 },
	);
	const firstLease = first.acquireLease("worker-a", {
		taskId: "task",
		runId: "run",
		now: 20,
		ttlMs: 10,
	});
	const second = new RunKernel(cwd, "shared-session");
	const secondLease = second.acquireLease("worker-b", {
		taskId: "task",
		runId: "run",
		now: 31,
		ttlMs: 10,
	});
	assert.ok(secondLease.epoch > firstLease.epoch);
	assert.throws(
		() =>
			first.append(
				{ type: "budget_consumed", resource: "provider_call", amount: 1 },
				{
					taskId: "task",
					runId: "run",
					leaseEpoch: firstLease.epoch,
					timestamp: 32,
				},
			),
		/older than/,
	);
});
