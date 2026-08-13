import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	formatTaskStateContext,
	shouldProjectTaskState,
	TaskStateController,
	taskObjectiveFromMessages,
} from "../agent/tasks/task-state-controller.ts";
import type { Message, ToolCall } from "../agent/types.ts";

function call(name: string, args: Record<string, unknown>): ToolCall {
	return { id: crypto.randomUUID(), name, arguments: JSON.stringify(args) };
}

function result(content: string): Message {
	return { role: "tool", content };
}

void test("provider projection omits trivial and terminal durable state", () => {
	const controller = new TaskStateController("hi");
	assert.equal(shouldProjectTaskState(controller.snapshot()), false);
	assert.equal(formatTaskStateContext(controller.snapshot()), "");

	controller.recordToolBatch(
		[call("task_status", { status: "done" })],
		[result("Recorded: done — No active tasks — ready for work.")],
		1,
	);
	controller.markHandoff();
	assert.equal(shouldProjectTaskState(controller.snapshot()), false);
	assert.equal(formatTaskStateContext(controller.snapshot()), "");
});

void test("provider projection is bounded and includes only decision-relevant state", () => {
	const controller = new TaskStateController("Fix authentication retries");
	controller.recordToolBatch(
		[call("edit_file", { path: "src/auth.ts" })],
		[result("Updated retry cap")],
		1,
	);
	controller.recordToolBatch(
		[call("bash", { command: "bun test auth.test.ts" })],
		[result("4 pass")],
		2,
	);
	const context = formatTaskStateContext(controller.snapshot());
	assert.match(context, /phase: verify/);
	assert.match(context, /changed_files: src\/auth\.ts/);
	assert.match(context, /verification: pass bun test auth\.test\.ts/);
	assert.doesNotMatch(context, /progress:/);
	assert.doesNotMatch(context, /blockers: none/);
	assert.doesNotMatch(context, /recent_evidence:\n- none/);
});

void test("restored controller resumes a durable active checkpoint", () => {
	const original = new TaskStateController("Fix retries");
	original.recordToolBatch(
		[call("edit_file", { path: "src/retry.ts" })],
		[result("Updated retry policy")],
		1,
	);
	const checkpoint = original.snapshot();
	const restored = new TaskStateController("ignored continuation", checkpoint);
	checkpoint.changedFiles.push("mutated-outside.ts");
	assert.equal(restored.snapshot().phase, "implement");
	assert.deepEqual(restored.snapshot().changedFiles, ["src/retry.ts"]);
	assert.match(restored.toContext(), /objective: Fix retries/);
});

void test("task state advances through investigation, implementation, and verification", () => {
	const controller = new TaskStateController(
		"Fix the parser and run its tests",
	);
	assert.equal(controller.snapshot().phase, "orient");

	controller.recordToolBatch(
		[call("read_file", { path: "src/parser.ts" })],
		[result("source")],
		1,
	);
	assert.equal(controller.snapshot().phase, "investigate");

	controller.recordToolBatch(
		[call("edit_file", { path: "src/parser.ts" })],
		[result("updated")],
		2,
	);
	assert.equal(controller.snapshot().phase, "implement");
	assert.deepEqual(controller.snapshot().changedFiles, ["src/parser.ts"]);

	controller.recordToolBatch(
		[call("bash", { command: "bun test parser.test.ts" })],
		[result("12 pass, 0 fail")],
		3,
	);
	const state = controller.snapshot();
	assert.equal(state.phase, "verify");
	assert.equal(state.verification[0]?.passed, true);
	assert.match(controller.toContext(), /phase: verify/);
});

void test("adaptive routing responds to objective, phase, and repeated failures", () => {
	const controller = new TaskStateController(
		"Review and diagnose the authentication flow",
	);
	assert.equal(controller.selectAdaptiveMode().mode, "analytical");

	controller.recordToolBatch(
		[call("edit_file", { path: "auth.ts" })],
		[result("updated")],
		1,
	);
	assert.equal(controller.selectAdaptiveMode().mode, "instruct-coding");

	controller.recordToolBatch(
		[call("bash", { command: "bun test" })],
		[result("error: failed")],
		2,
	);
	controller.recordToolBatch(
		[call("bash", { command: "bun test" })],
		[result("error: failed again")],
		3,
	);
	assert.equal(controller.selectAdaptiveMode().mode, "thinking-coding");
	assert.equal(controller.snapshot().toolFailures, 2);
});

void test("task state snapshots cannot mutate controller state", () => {
	const controller = new TaskStateController("Implement a feature");
	const snapshot = controller.snapshot();
	snapshot.phase = "blocked";
	snapshot.changedFiles.push("fake.ts");
	assert.equal(controller.snapshot().phase, "orient");
	assert.deepEqual(controller.snapshot().changedFiles, []);
});

void test("continuation turns retain the last meaningful task objective", () => {
	assert.equal(
		taskObjectiveFromMessages([
			{ role: "user", content: "Fix authentication retries" },
			{ role: "assistant", content: "I will inspect it." },
			{ role: "user", content: "continue" },
		]),
		"Fix authentication retries",
	);
});
