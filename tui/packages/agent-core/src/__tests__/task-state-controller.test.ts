import assert from "node:assert/strict";
import { test } from "node:test";
import type { Message, ToolCall } from "../agent/types.ts";
import {
	TaskStateController,
	taskObjectiveFromMessages,
} from "../agent/tasks/task-state-controller.ts";

function call(name: string, args: Record<string, unknown>): ToolCall {
	return { id: crypto.randomUUID(), name, arguments: JSON.stringify(args) };
}

function result(content: string): Message {
	return { role: "tool", content };
}

void test("task state advances through investigation, implementation, and verification", () => {
	const controller = new TaskStateController("Fix the parser and run its tests");
	assert.equal(controller.snapshot().phase, "orient");

	controller.recordToolBatch([call("read_file", { path: "src/parser.ts" })], [result("source")], 1);
	assert.equal(controller.snapshot().phase, "investigate");

	controller.recordToolBatch([call("edit_file", { path: "src/parser.ts" })], [result("updated")], 2);
	assert.equal(controller.snapshot().phase, "implement");
	assert.deepEqual(controller.snapshot().changedFiles, ["src/parser.ts"]);

	controller.recordToolBatch([call("bash", { command: "bun test parser.test.ts" })], [result("12 pass, 0 fail")], 3);
	const state = controller.snapshot();
	assert.equal(state.phase, "verify");
	assert.equal(state.verification[0]?.passed, true);
	assert.match(controller.toContext(), /phase: verify/);
});

void test("adaptive routing responds to objective, phase, and repeated failures", () => {
	const controller = new TaskStateController("Review and diagnose the authentication flow");
	assert.equal(controller.selectAdaptiveMode().mode, "analytical");

	controller.recordToolBatch([call("edit_file", { path: "auth.ts" })], [result("updated")], 1);
	assert.equal(controller.selectAdaptiveMode().mode, "instruct-coding");

	controller.recordToolBatch([call("bash", { command: "bun test" })], [result("error: failed")], 2);
	controller.recordToolBatch([call("bash", { command: "bun test" })], [result("error: failed again")], 3);
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
