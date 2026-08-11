import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { ContinuationController } from "../application/continuation-controller.ts";

void test("continuation budgets survive controller restart", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-continuation-"));
	const limits = { maxRuns: 5, maxNoProgressRuns: 2, maxElapsedMs: 60_000 };
	const first = new ContinuationController(cwd, "session-a", limits);
	first.start("finish the task", "state-a");
	assert.equal(first.request("next_turn", "state-a").action, "continue");

	const restored = new ContinuationController(cwd, "session-a", limits);
	const decision = restored.request("next_turn", "state-a");
	assert.equal(decision.action, "pause");
	assert.match(
		decision.action === "pause" ? decision.reason : "",
		/no semantic progress/,
	);
});

void test("semantic progress resets the no-progress budget", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-progress-"));
	const controller = new ContinuationController(cwd, "session-b", {
		maxRuns: 5,
		maxNoProgressRuns: 2,
		maxElapsedMs: 60_000,
	});
	controller.start("finish the task", "state-a");
	assert.equal(controller.request("next_turn", "state-a").action, "continue");
	assert.equal(controller.request("next_turn", "state-b").action, "continue");
	assert.equal(controller.snapshot()?.noProgressRuns, 0);
});
