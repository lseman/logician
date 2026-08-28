import { beforeEach, expect, test } from "bun:test";
import {
	getTasks,
	onTodosChanged,
	todo_tool,
} from "../../capabilities/tasks/todo.ts";

async function execute(args: Record<string, unknown>): Promise<string> {
	return String(await todo_tool.execute(args, {}));
}

beforeEach(async () => {
	await execute({ action: "clear" });
});

test("failed creates are atomic and do not consume task ids", async () => {
	expect(
		await execute({
			action: "create",
			subject: "invalid",
			blockedBy: [999],
		}),
	).toContain("not found");

	expect(await execute({ action: "create", subject: "first valid task" })).toBe(
		"Created task #1: first valid task",
	);
});

test("invalid metadata cannot partially mutate an existing task", async () => {
	await execute({
		action: "create",
		subject: "original",
		metadata: { source: "test" },
	});

	expect(
		await execute({
			action: "update",
			id: 1,
			subject: "partially changed",
			metadata: "not-an-object",
		}),
	).toBe("Error: metadata must be an object.");
	expect(getTasks()).toEqual([
		{
			id: 1,
			subject: "original",
			status: "pending",
			metadata: { source: "test" },
		},
	]);
});

test("dependents unblock only after every blocker is completed", async () => {
	await execute({ action: "create", subject: "blocker" });
	await execute({ action: "create", subject: "dependent", blockedBy: [1] });
	await execute({ action: "update", id: 2, status: "in_progress" });

	await execute({ action: "update", id: 1, subject: "renamed blocker" });
	expect(getTasks().find(task => task.id === 2)?.status).toBe("in_progress");

	const completion = await execute({
		action: "update",
		id: 1,
		status: "completed",
	});
	expect(completion).toContain("Unblocked #2");
	expect(getTasks().find(task => task.id === 2)?.status).toBe("pending");
});

test("dependency cycles are rejected without mutating either task", async () => {
	await execute({ action: "create", subject: "first" });
	await execute({ action: "create", subject: "second", blockedBy: [1] });

	expect(
		await execute({ action: "update", id: 1, addBlockedBy: [2] }),
	).toContain("dependency cycle");
	expect(getTasks().find(task => task.id === 1)?.blockedBy).toBeUndefined();
	expect(getTasks().find(task => task.id === 2)?.blockedBy).toEqual([1]);
});

test("observers receive one isolated snapshot per successful mutation", async () => {
	const snapshots: string[][] = [];
	const unsubscribe = onTodosChanged(tasks => {
		snapshots.push(tasks.map(task => task.subject));
		if (tasks[0]) tasks[0].subject = "observer mutation";
	});

	try {
		await execute({ action: "list" });
		await execute({ action: "update", id: 999, subject: "missing" });
		await execute({ action: "create", subject: "committed" });
		expect(snapshots).toEqual([["committed"]]);
		expect(getTasks()[0]?.subject).toBe("committed");
	} finally {
		unsubscribe();
	}
});
