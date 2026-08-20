import { test } from "bun:test";
import assert from "node:assert/strict";
import { ProgressTracker } from "../../../core/policy/progress-tracker.ts";

void test("progress tracker stops repeated evidence-free checks", () => {
	const tracker = new ProgressTracker({ minimumChecks: 1, stalledChecks: 1 });
	assert.equal(tracker.shouldStop([]), false);
	assert.equal(tracker.shouldStop([]), true);
});

void test("new tool evidence resets progress stalling", () => {
	const tracker = new ProgressTracker({ minimumChecks: 1, stalledChecks: 1 });
	assert.equal(tracker.shouldStop([]), false);
	tracker.recordToolResult("edit_file", '{"path":"a.ts"}', "updated");
	assert.equal(tracker.shouldStop([]), false);
});

void test("task transitions count as progress", () => {
	const tracker = new ProgressTracker({ minimumChecks: 1, stalledChecks: 1 });
	assert.equal(tracker.shouldStop([{ id: 1, status: "pending" }]), false);
	assert.equal(tracker.shouldStop([{ id: 1, status: "completed" }]), false);
});
