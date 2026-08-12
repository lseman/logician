import { test } from "bun:test";
import assert from "node:assert/strict";
import { appendFileSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { LLMBackend } from "../agent/backend.ts";
import { BackendError } from "../agent/backend.ts";
import {
	evaluateTrajectory,
	FaultInjectingBackend,
	TrajectoryRecorder,
} from "../agent/trajectory.ts";
import type { AgentConfig } from "../agent/types.ts";

const config: AgentConfig = {
	baseUrl: "http://localhost:8080",
	model: "test-model",
	maxIterations: 7,
	tools: [],
};

void test("trajectory recorder persists correlated metadata and replays a torn journal", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-trajectory-"));
	const recorder = new TrajectoryRecorder(cwd, "session-a");
	recorder.start("run-a", "op-a", config, "prompt");
	recorder.record("run-a", "op-a", {
		type: "tool_execution_start",
		toolCallId: "t1",
		toolName: "read",
		args: {},
	});
	recorder.record("run-a", "op-a", {
		type: "tool_execution_end",
		toolCallId: "t1",
		toolName: "read",
		result: "",
		isError: true,
	});
	recorder.finish("run-a", "op-a", "failed");
	const file = path.join(cwd, ".logician", "trajectories", "session-a.jsonl");
	appendFileSync(file, '{"version":1');

	const entries = new TrajectoryRecorder(cwd, "session-a").load();
	assert.equal(entries.length, 4);
	assert.ok(
		entries.every(
			entry => entry.runId === "run-a" && entry.operationId === "op-a",
		),
	);
	const metadata = entries[0]?.payload.metadata as {
		model?: string;
		config?: { maxIterations?: number };
	};
	assert.equal(metadata.model, "test-model");
	assert.equal(metadata.config?.maxIterations, 7);
	assert.equal(evaluateTrajectory(entries).toolFailures, 1);
	assert.equal(evaluateTrajectory(entries).replayComplete, true);
	assert.match(readFileSync(file, "utf8"), /"kind":"run_finish"/);
	const resumed = new TrajectoryRecorder(cwd, "session-a");
	resumed.finish("run-a", "op-b", "resumed");
	assert.equal(resumed.load().at(-1)?.sequence, 5);
});

void test("trajectory evaluation flags unsupported completed outcomes", () => {
	const base = {
		version: 1 as const,
		sessionId: "s",
		runId: "r",
		operationId: "o",
	};
	const report = evaluateTrajectory([
		{
			...base,
			sequence: 1,
			timestamp: 10,
			kind: "agent_event",
			payload: {
				type: "task_state_update",
				state: {
					phase: "implement",
					blockers: [],
					verification: [{ passed: false }],
				},
			},
		},
		{
			...base,
			sequence: 2,
			timestamp: 20,
			kind: "agent_event",
			payload: { type: "run_outcome", status: "completed" },
		},
		{
			...base,
			sequence: 3,
			timestamp: 30,
			kind: "run_finish",
			payload: { status: "completed" },
		},
	]);
	assert.equal(report.acceptancePassed, false);
	assert.equal(report.prematureStop, true);
	assert.equal(report.durationMs, 20);
});

void test("fault injecting backend deterministically exercises recovery categories", async () => {
	const backend: LLMBackend = {
		model: "base",
		generate: async () => ({
			content: "ok",
			toolCalls: [],
			stopReason: "stop",
		}),
		withModel: () => backend,
	};
	const injected = new FaultInjectingBackend(backend, [
		"rate_limit",
		"context_full",
	]);
	await assert.rejects(
		injected.generate([]),
		error => error instanceof BackendError && error.category === "rate_limit",
	);
	await assert.rejects(
		injected.generate([]),
		error => error instanceof BackendError && error.category === "context_full",
	);
	assert.equal((await injected.generate([])).content, "ok");
});
