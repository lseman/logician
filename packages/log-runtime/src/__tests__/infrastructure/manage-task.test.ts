import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createManageTaskTool } from "../../capabilities/tools/manage-task.ts";
import { BackgroundTaskRegistry } from "../../capabilities/tools/support/utils/background-task-registry.ts";
import { spawn } from "node:child_process";
import { getShellConfig } from "../../capabilities/tools/support/utils/shell.ts";

void test("manage_task tool lists, checks status, and kills background tasks", async () => {
	const logDir = mkdtempSync(join(tmpdir(), "logician-manage-task-"));
	const manager = new BackgroundTaskRegistry(logDir);
	const tool = createManageTaskTool(manager);
	const { shell, args } = getShellConfig();

	// Initially empty list
	const emptyRes = await tool.execute({ action: "list" }, {});
	const emptyContent = typeof emptyRes === "string" ? emptyRes : emptyRes.content;
	assert.match(emptyContent, /No background tasks/);

	// Register a task
	const child = spawn(shell, [...args, "echo 'running bg task'; sleep 5"], {
		stdio: ["pipe", "pipe", "pipe"],
	});
	const taskEntry = manager.registerTask({
		command: "echo 'running bg task'; sleep 5",
		cwd: logDir,
		child,
	});

	// List shows task
	const listRes = await tool.execute({ action: "list" }, {});
	const listContent = typeof listRes === "string" ? listRes : listRes.content;
	assert.match(listContent, new RegExp(taskEntry.id));
	assert.match(listContent, /RUNNING/);

	// Check status
	const statusRes = await tool.execute({ action: "status", taskId: taskEntry.id }, {});
	const statusContent = typeof statusRes === "string" ? statusRes : statusRes.content;
	assert.match(statusContent, /Status: RUNNING/);
	assert.match(statusContent, new RegExp(taskEntry.id));

	// Send input
	const inputRes = await tool.execute({ action: "send_input", taskId: taskEntry.id, input: "test" }, {});
	const inputContent = typeof inputRes === "string" ? inputRes : inputRes.content;
	assert.match(inputContent, /Sent \d+ bytes/);

	// Kill task
	const killRes = await tool.execute({ action: "kill", taskId: taskEntry.id }, {});
	const killContent = typeof killRes === "string" ? killRes : killRes.content;
	assert.match(killContent, /terminated/);

	manager.cleanupAll();
});
