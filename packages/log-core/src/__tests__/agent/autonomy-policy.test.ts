import { test } from "bun:test";
import assert from "node:assert/strict";
import { decideAutonomousContinuation } from "../../control/policy/autonomy-policy.ts";

const pendingTask = { id: "1", subject: "Run tests", status: "in_progress" };

void test("autonomy pauses when the assistant asks the user", () => {
	assert.equal(
		decideAutonomousContinuation({
			assistantText: "Which environment should I test?",
			tasks: [pendingTask],
		}),
		undefined,
	);
});

void test("autonomy recovers a truncated provider response", () => {
	const decision = decideAutonomousContinuation({
		assistantText: "I was editing",
		stopReason: "length",
		tasks: [],
	});
	assert.equal(decision?.reason, "length_truncation");
});

void test("autonomy continues the active unfinished task", () => {
	const decision = decideAutonomousContinuation({
		assistantText: "I made progress.",
		tasks: [
			{ id: "1", subject: "Inspect", status: "pending" },
			{ id: "2", subject: "Verify", status: "in_progress" },
		],
	});
	assert.equal(decision?.reason, "unfinished_todos");
	assert.match(decision?.message ?? "", /#2 Verify/);
});

void test("autonomy finishes when no work remains", () => {
	assert.equal(
		decideAutonomousContinuation({
			assistantText: "Implemented and verified.",
			tasks: [{ ...pendingTask, status: "completed" }],
		}),
		undefined,
	);
});
