import { beforeEach, expect, test } from "bun:test";
import { getTasks, todo_tool } from "../../capabilities/tasks/todo.ts";

// ── Helpers ──────────────────────────────────────────────────────────────────

async function execute(args: Record<string, unknown>): Promise<string> {
	const result = await todo_tool.execute(args, {});
	return typeof result === "string" ? result : result.content;
}

// Each test gets a clean slate.
beforeEach(async () => {
	await execute({ action: "init", items: [] });
});

// ── init ─────────────────────────────────────────────────────────────────────

test("init with phase name strings", async () => {
	const result = await execute({
		action: "init",
		items: ["Foundation", "Implementation"],
	});
	expect(result).toContain("Created 2 phase(s)");
	expect(result).toContain("Foundation");
	expect(result).toContain("Implementation");

	const tasks = getTasks();
	expect(tasks.length).toBe(2);
	expect(tasks[0].name).toBe("Foundation");
	expect(tasks[0].tasks.length).toBe(0);
	expect(tasks[1].name).toBe("Implementation");
	expect(tasks[1].tasks.length).toBe(0);
});

test("init with phase+tasks objects", async () => {
	const result = await execute({
		action: "init",
		items: [
			{ phase: "Foundation", items: ["Set up repo", "Install deps"] },
			{ phase: "Testing", items: ["Write unit tests"] },
		],
	});
	expect(result).toContain("Created 2 phase(s)");
	expect(result).toContain("3 tasks");

	const tasks = getTasks();
	expect(tasks.length).toBe(2);
	expect(tasks[0].tasks.length).toBe(2);
	expect(tasks[0].tasks[0].content).toBe("Set up repo");
	expect(tasks[0].tasks[0].status).toBe("pending");
	expect(tasks[1].tasks.length).toBe(1);
});

test("init ignores empty strings", async () => {
	const result = await execute({
		action: "init",
		items: ["", "Valid", ""],
	});
	expect(result).toContain("Created 1 phase(s)");
	expect(result).toContain("Valid");
});

test("init rejects duplicate phases", async () => {
	const result = await execute({
		action: "init",
		items: ["Phase A", "Phase A"],
	});
	expect(result).toContain("Duplicate phase");
	const tasks = getTasks();
	expect(tasks.length).toBe(0);
});

// ── view ─────────────────────────────────────────────────────────────────────

test("view returns 'No tasks.' when empty", async () => {
	const result = await execute({ action: "view" });
	expect(result).toBe("No tasks.");
});

test("view lists phases and tasks with status markers", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["Task A", "Task B"] }],
	});
	const result = await execute({ action: "view" });
	expect(result).toContain("Phase: Work");
	expect(result).toContain("○ Task A");
	expect(result).toContain("○ Task B");
	expect(result).toContain("(0 done, 0 active, 2 total)");
});

test("view shows correct counts for mixed statuses", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A", "B", "C", "D"] }],
	});
	await execute({ action: "done", task: "A" });
	await execute({ action: "start", phase: "Work", task: "B" });
	const result = await execute({ action: "view" });
	expect(result).toContain("(1 done, 1 active, 4 total)");
	expect(result).toContain("✓ A");
	expect(result).toContain("→ B");
	expect(result).toContain("○ C");
});

// ── start ────────────────────────────────────────────────────────────────────

test("start marks task in_progress", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["Task A"] }],
	});
	const result = await execute({
		action: "start",
		phase: "Work",
		task: "Task A",
	});
	expect(result).toContain("Started 'Task A' in Work");

	const tasks = getTasks();
	expect(tasks[0].tasks[0].status).toBe("in_progress");
});

test("start rejects non-existent task", async () => {
	const result = await execute({
		action: "start",
		phase: "Work",
		task: "Ghost",
	});
	expect(result).toContain("not found");
});

test("start rejects already in_progress task", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["Task A"] }],
	});
	await execute({ action: "start", phase: "Work", task: "Task A" });
	const result = await execute({
		action: "start",
		phase: "Work",
		task: "Task A",
	});
	expect(result).toBe("Task already in progress.");
});

test("start rejects completed task", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["Task A"] }],
	});
	await execute({ action: "done", task: "Task A" });
	const result = await execute({
		action: "start",
		phase: "Work",
		task: "Task A",
	});
	expect(result).toBe("Task already completed.");
});

test("start requires both phase and task", async () => {
	const result = await execute({ action: "start", phase: "Work" });
	expect(result).toContain("both 'phase' and 'task' are required");
});

// ── done ─────────────────────────────────────────────────────────────────────

test("done marks task completed", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["Task A"] }],
	});
	const result = await execute({ action: "done", task: "Task A" });
	expect(result).toContain("Completed 'Task A'");

	const tasks = getTasks();
	expect(tasks[0].tasks[0].status).toBe("completed");
});

test("done rejects non-existent task", async () => {
	const result = await execute({ action: "done", task: "Ghost" });
	expect(result).toContain("not found");
});

test("done requires task parameter", async () => {
	const result = await execute({ action: "done" });
	expect(result).toContain("'task' is required");
});

// ── rm ───────────────────────────────────────────────────────────────────────

test("rm removes a task from a phase", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A", "B", "C"] }],
	});
	const result = await execute({ action: "rm", phase: "Work", task: "B" });
	expect(result).toContain("Removed 'B' from Work");

	const tasks = getTasks();
	expect(tasks[0].tasks.length).toBe(2);
	expect(tasks[0].tasks.map(t => t.content)).toEqual(["A", "C"]);
});

test("rm rejects non-existent phase", async () => {
	const result = await execute({ action: "rm", phase: "Ghost", task: "A" });
	expect(result).toContain("not found");
});

test("rm rejects non-existent task in phase", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A"] }],
	});
	const result = await execute({ action: "rm", phase: "Work", task: "B" });
	expect(result).toContain("not in phase");
});

test("rm requires both phase and task", async () => {
	const result = await execute({ action: "rm", phase: "Work" });
	expect(result).toContain("both 'phase' and 'task' are required");
});

// ── drop ─────────────────────────────────────────────────────────────────────

test("drop removes an entire phase", async () => {
	await execute({
		action: "init",
		items: [
			{ phase: "Work", items: ["A", "B"] },
			{ phase: "Rest", items: ["C"] },
		],
	});
	const result = await execute({ action: "drop", phase: "Work" });
	expect(result).toContain("Dropped phase 'Work' (2 task(s))");

	const tasks = getTasks();
	expect(tasks.length).toBe(1);
	expect(tasks[0].name).toBe("Rest");
});

test("drop rejects non-existent phase", async () => {
	const result = await execute({ action: "drop", phase: "Ghost" });
	expect(result).toContain("not found");
});

test("drop requires phase parameter", async () => {
	const result = await execute({ action: "drop" });
	expect(result).toContain("'phase' is required");
});

// ── block / unblock ──────────────────────────────────────────────────────────

test("block sets a blocker on a task", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A", "B"] }],
	});
	const result = await execute({
		action: "block",
		phase: "Work",
		task: "B",
		blocker: "Waiting for A",
	});
	expect(result).toContain("Blocked 'B' by 'Waiting for A'");

	const tasks = getTasks();
	expect(tasks[0].tasks[1].blocker).toBe("Waiting for A");
	expect(tasks[0].tasks[1].status).toBe("pending");
});

test("block rejects non-existent task", async () => {
	const result = await execute({
		action: "block",
		phase: "Work",
		task: "Ghost",
		blocker: "X",
	});
	expect(result).toContain("not found");
});

test("block requires all three params", async () => {
	const result = await execute({ action: "block", phase: "Work", task: "A" });
	expect(result).toContain("'phase', 'task', and 'blocker' are required");
});

test("unblock clears the blocker", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A"] }],
	});
	await execute({
		action: "block",
		phase: "Work",
		task: "A",
		blocker: "X",
	});
	const result = await execute({ action: "unblock", task: "A" });
	expect(result).toContain("Unblocked 'A'");

	const tasks = getTasks();
	expect(tasks[0].tasks[0].blocker).toBeUndefined();
});

test("unblock rejects non-existent task", async () => {
	const result = await execute({ action: "unblock", task: "Ghost" });
	expect(result).toContain("not found");
});

test("unblock requires task parameter", async () => {
	const result = await execute({ action: "unblock" });
	expect(result).toContain("'task' is required");
});

// ── append ───────────────────────────────────────────────────────────────────

test("append adds tasks to existing phase", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A"] }],
	});
	const result = await execute({
		action: "append",
		phase: "Work",
		tasks: ["B", "C"],
	});
	expect(result).toContain("Added 2 task(s) to Work");

	const tasks = getTasks();
	expect(tasks[0].tasks.length).toBe(3);
	expect(tasks[0].tasks.map(t => t.content)).toEqual(["A", "B", "C"]);
});

test("append creates phase if it doesn't exist", async () => {
	const result = await execute({
		action: "append",
		phase: "NewPhase",
		tasks: ["A", "B"],
	});
	expect(result).toContain("Created phase 'NewPhase'");

	const tasks = getTasks();
	expect(tasks.length).toBe(1);
	expect(tasks[0].name).toBe("NewPhase");
	expect(tasks[0].tasks.length).toBe(2);
});

test("append skips duplicate tasks", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: ["A"] }],
	});
	const result = await execute({
		action: "append",
		phase: "Work",
		tasks: ["A", "B"],
	});
	expect(result).toContain("Added 1 task(s) to Work");

	const tasks = getTasks();
	expect(tasks[0].tasks.length).toBe(2);
});

test("append requires phase and tasks array", async () => {
	const result = await execute({ action: "append", phase: "Work" });
	expect(result).toContain("'phase' and 'tasks' (array) are required");
});

test("append ignores empty strings in task list", async () => {
	await execute({
		action: "init",
		items: [{ phase: "Work", items: [] }],
	});
	const result = await execute({
		action: "append",
		phase: "Work",
		tasks: ["", "A", ""],
	});
	expect(result).toContain("Added 1 task(s)");
});

// ── unknown op ───────────────────────────────────────────────────────────────

test("unknown op returns error", async () => {
	const result = await execute({ action: "bogus" });
	expect(result).toContain("unknown op 'bogus'");
});

test("missing op returns error", async () => {
	const result = await execute({});
	expect(result).toContain("'op' is required");
});

// ── clear (edge case via init) ───────────────────────────────────────────────

test("init with empty items leaves no phases", async () => {
	const result = await execute({ action: "init", items: [] });
	expect(result).toContain("Created 0 phase(s)");
	expect(getTasks().length).toBe(0);
});
