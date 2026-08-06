import assert from "node:assert/strict";
import { test } from "node:test";
import { getTasks, onTodosChanged, todo_tool } from "../tasks/todo.ts";

void test("todo store rejects invisible subjects and publishes immutable snapshots", async () => {
	await todo_tool.execute({ action: "clear" }, {});
	const invisible = await todo_tool.execute(
		{ action: "create", subject: "\u200b\u200d" },
		{},
	);
	assert.match(String(invisible), /subject is required/i);

	let observed = false;
	const unsubscribe = onTodosChanged(tasks => {
		observed = true;
		tasks[0].subject = "mutated outside store";
	});
	await todo_tool.execute({ action: "create", subject: "Real task" }, {});
	unsubscribe();
	assert.equal(observed, true);
	assert.equal(getTasks()[0]?.subject, "Real task");

	const external = getTasks();
	external[0].subject = "also mutated";
	assert.equal(getTasks()[0]?.subject, "Real task");
});
