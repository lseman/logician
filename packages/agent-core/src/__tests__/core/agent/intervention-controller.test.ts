import { test } from "bun:test";
import assert from "node:assert/strict";

import { HarnessInterventionController } from "../../../core/policy/intervention-controller.ts";

const input = (iteration: number) => ({
	kind: "loop" as const,
	cause: "stagnation",
	detector: "test",
	message: "No verified progress",
	iteration,
	signals: ["same_call", "no_state_delta"],
});

void test("interventions retain an incident id and escalate deterministically", () => {
	const controller = new HarnessInterventionController();
	const first = controller.record(input(1));
	const second = controller.record(input(2));
	const third = controller.record(input(3));

	assert.equal(first.action, "recover");
	assert.equal(second.action, "change_strategy");
	assert.equal(third.action, "pause");
	assert.equal(first.id, second.id);
	assert.equal(second.id, third.id);
	assert.deepEqual([first.attempt, second.attempt, third.attempt], [1, 2, 3]);
});

void test("verified progress closes incidents", () => {
	const controller = new HarnessInterventionController();
	const first = controller.record(input(1));
	controller.recordProgress();
	const recovered = controller.record(input(4));

	assert.equal(recovered.action, "recover");
	assert.equal(recovered.attempt, 1);
	assert.notEqual(recovered.id, first.id);
});

void test("callers can explicitly select lifecycle actions", () => {
	const controller = new HarnessInterventionController();
	const event = controller.record({
		...input(1),
		kind: "continuation",
		cause: "policy",
		action: "continue",
	});

	assert.equal(event.action, "continue");
	assert.equal(event.severity, "info");
});

void test("a pinned action never escalates, even repeated past the auto-escalation threshold", () => {
	// Regression: guard-engine's evaluate() caps heuristic-only signals
	// (progress_signal etc.) at "nudge" severity. The loop runner used to map
	// that to action: undefined, which let record()'s own repeat-count-based
	// escalatedAction() kick in and independently escalate the SAME repeated
	// nudge to "pause" on the 3rd occurrence — silently reintroducing the
	// exact false-positive interrupt the severity cap exists to prevent.
	// Once the caller pins an explicit action (e.g. "continue" for nudge),
	// repeats on the same kind:detector:cause key must never escalate past it.
	const controller = new HarnessInterventionController();
	for (let iteration = 1; iteration <= 5; iteration++) {
		const event = controller.record({
			...input(iteration),
			action: "continue",
		});
		assert.equal(event.action, "continue");
		assert.equal(event.severity, "info");
	}
});

void test("durable trajectories restore escalation state for resumed runs", () => {
	const original = new HarnessInterventionController();
	const trajectory = [original.record(input(1)), original.record(input(2))];
	const resumed = new HarnessInterventionController();
	resumed.replay(trajectory);

	const next = resumed.record(input(5));
	assert.equal(next.id, trajectory[0]?.id);
	assert.equal(next.attempt, 3);
	assert.equal(next.action, "pause");
});
