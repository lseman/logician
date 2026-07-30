import assert from "node:assert/strict";
import { test } from "node:test";
import { TodoBar, type TaskItem } from "../src/status/todo-bar.ts";
import { initTheme } from "../src/terminal/theme.ts";

initTheme("dark");

const task = (
	subject: string,
	id: number,
	status: TaskItem["status"] = "pending",
): TaskItem => ({ id, subject, status });

void test("TodoBar drops blank and invisible task labels", () => {
	const bar = new TodoBar();
	bar.setTodos([
		task("   \n\t", 1),
		task("\u200b\u200d", 2),
		task("\x1b[31m\x1b[0m", 3),
		task("  Ship\nfeature  ", 4),
	]);
	const rows = bar.render(80);
	// header + task = 2 lines
	assert.equal(rows.length, 2);
	assert.match(rows[0], /Tasks 0\/1/);
	assert.match(rows[1], /Ship feature/);
});

void test("TodoBar renders nothing when every task label is blank", () => {
	const bar = new TodoBar();
	bar.setTodos([task(" ", 1), task("\u200b", 2)]);
	assert.deepEqual(bar.render(80), []);
});

void test("TodoBar renders clean lines (no borders)", () => {
	const bar = new TodoBar();
	bar.setTodos([task("Fix login", 1)]);
	const rows = bar.render(80);

	// 2 lines: header, task
	assert.equal(rows.length, 2);

	// No box-drawing characters
	assert.ok(!rows[0].includes("┌"));
	assert.ok(!rows[0].includes("├"));
	assert.ok(!rows[0].includes("│"));
	assert.ok(!rows[0].includes("─"));
});

void test("TodoBar groups tasks by status", () => {
	const bar = new TodoBar();
	bar.setTodos([
		task("Fix bug", 1, "pending"),
		task("Writing tests", 2, "in_progress"),
		task("Done task", 3, "completed"),
	]);
	const rows = bar.render(80);

	// Header shows 1/3
	assert.match(rows[0], /Tasks 1\/3/);

	// in_progress should appear before pending (order: in_progress, pending, completed)
	const inProgIdx = rows.findIndex((r) => /Writing tests/.test(r));
	const pendingIdx = rows.findIndex((r) => /Fix bug/.test(r));
	const doneIdx = rows.findIndex((r) => /Done task/.test(r));
	assert.ok(inProgIdx < pendingIdx, "in_progress before pending");
	assert.ok(pendingIdx < doneIdx, "pending before completed");

	// Status markers present
	assert.match(rows[inProgIdx], /▸/); // in_progress marker
	assert.match(rows[pendingIdx], /○/); // pending marker
	assert.match(rows[doneIdx], /✓/); // completed marker
});

void test("TodoBar shows dependencies", () => {
	const bar = new TodoBar();
	bar.setTodos([
		{ id: 1, subject: "Base task", status: "pending" },
		{ id: 2, subject: "Depends on #1", status: "pending", blockedBy: [1] },
	]);
	const rows = bar.render(80);

	// Second task should show dependency
	const depRow = rows.find((r) => /Depends on/.test(r))!;
	assert.ok(depRow, "dependency task should be rendered");
	assert.match(depRow, /→/);
});

void test("TodoBar shows active form for in_progress tasks", () => {
	const bar = new TodoBar();
	bar.setTodos([
		{
			id: 1,
			subject: "Running task",
			status: "in_progress",
			activeForm: "checking logs",
		},
	]);
	const rows = bar.render(80);

	// Should show active form indicator
	const taskRow = rows.find((r) => /Running task/.test(r))!;
	assert.match(taskRow, /—/);
});

void test("TodoBar shows hidden count when too many tasks", () => {
	const bar = new TodoBar();
	bar.setTodos([
		task("A", 1),
		task("B", 2),
		task("C", 3),
		task("D", 4),
		task("E", 5),
		task("F", 6),
		task("G", 7),
	]);
	const rows = bar.render(80);

	// MAX_ROWS is 5, so 2 hidden (header + 5 tasks + hidden hint)
	const hiddenIdx = rows.findIndex((r) => /more/.test(r));
	assert.ok(hiddenIdx >= 0, "should show hidden count");
	assert.match(rows[hiddenIdx], /2 more/);
});
