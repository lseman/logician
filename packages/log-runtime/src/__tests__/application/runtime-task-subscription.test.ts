import { expect, test } from "bun:test";
import { todo_tool } from "../../capabilities/tasks/todo.ts";
import { AgentRuntime } from "../../runtime/bridge/agent-bridge.ts";

async function mutateTodo(args: Record<string, unknown>): Promise<void> {
	await todo_tool.execute(args, {});
}

test("stopping a runtime detaches it from global task updates", async () => {
	await mutateTodo({ action: "clear" });
	const stopped = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});
	const active = new AgentRuntime({
		baseUrl: "http://127.0.0.1:1",
		model: "test",
		runtimeHooksEnabled: false,
	});

	await stopped.stop();
	const stoppedSequence = stopped.events.snapshot().at(-1)?.sequence ?? 0;
	const activeSequence = active.events.snapshot().at(-1)?.sequence ?? 0;
	await mutateTodo({ action: "create", subject: "Verify detachment" });

	expect(stopped.events.snapshot().at(-1)?.sequence ?? 0).toBe(stoppedSequence);
	expect(active.events.snapshot().at(-1)?.sequence ?? 0).toBeGreaterThan(
		activeSequence,
	);
	expect(active.events.snapshot().at(-1)?.event.type).toBe("todos");
	await active.stop();
	await mutateTodo({ action: "clear" });
});
