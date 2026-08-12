import { test } from "bun:test";
import assert from "node:assert/strict";
import { appendFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { RunStateController } from "../agent/run-state.ts";

void test("durable runtime restores continuation budgets after restart", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-"));
	const limits = { maxRuns: 5, maxNoProgressRuns: 2, maxElapsedMs: 60_000 };
	const first = new RunStateController(cwd, "session-a", limits);
	first.start("finish the task", "state-a");
	assert.equal(
		first.requestContinuation("next_turn", "state-a").action,
		"continue",
	);

	const restored = new RunStateController(cwd, "session-a", limits);
	const decision = restored.requestContinuation("next_turn", "state-a");
	assert.equal(decision.action, "pause");
	assert.match(
		decision.action === "pause" ? decision.reason : "",
		/no semantic progress/,
	);
});

void test("durable runtime replays task state and terminal outcome", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-state-"));
	const controller = new RunStateController(cwd, "session-b");
	controller.start("ship it");
	controller.applyTaskState({
		objective: "ship it",
		phase: "verify",
		hypotheses: [],
		changedFiles: ["src/main.ts"],
		verification: [],
		blockers: [],
		evidence: [],
		toolCalls: 1,
		toolFailures: 0,
	});
	controller.applyOutcome({ status: "completed", source: "structured" });
	controller.recordCompaction();

	const restored = new RunStateController(cwd, "session-b").snapshot();
	assert.equal(restored?.status, "completed");
	assert.equal(restored?.taskState?.phase, "verify");
	assert.equal(restored?.outcome?.source, "structured");
	assert.equal(restored?.compactionGeneration, 1);
});

void test("durable runtime tolerates a truncated final journal record", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-runtime-corrupt-"));
	const controller = new RunStateController(cwd, "session-c");
	controller.start("recover me", "before");
	appendFileSync(
		path.join(cwd, ".logician", "runtime", "session-c.jsonl"),
		'{"version":1,"sequence":2',
	);

	const restored = new RunStateController(cwd, "session-c").snapshot();
	assert.equal(restored?.rootPrompt, "recover me");
	assert.equal(restored?.lastEventSequence, 1);
});
