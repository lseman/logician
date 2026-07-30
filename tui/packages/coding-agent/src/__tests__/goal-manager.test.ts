import assert from "node:assert/strict";
import { test } from "node:test";
import { GoalManager } from "../application/goal-manager.ts";

void test("unmet goal remains active until its turn limit", () => {
	const manager = new GoalManager();
	manager.set("tests pass", 2);
	manager.recordEvaluation(false, "tests still fail");
	assert.equal(manager.getState()?.status, "active");
	assert.equal(manager.getState()?.turnCount, 1);

	manager.recordEvaluation(false, "one test still fails");
	assert.equal(manager.getState()?.status, "cancelled");
	assert.equal(manager.getState()?.turnCount, 2);
});

void test("met goal records the successful evaluator turn", () => {
	const manager = new GoalManager();
	manager.set("build succeeds");
	manager.recordEvaluation(true, "build passed");
	assert.equal(manager.getState()?.status, "achieved");
	assert.equal(manager.getState()?.turnCount, 1);
	assert.equal(manager.getState()?.lastReason, "build passed");
});

void test("goal parser removes a turn limit clause cleanly", () => {
	assert.deepEqual(
		GoalManager.parseCondition("all tests pass or stop after 4 turns"),
		{ condition: "all tests pass", maxTurns: 4 },
	);
});
