import { test } from "bun:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BackgroundTaskRegistry } from "../../capabilities/tools/support/utils/background-task-registry.ts";
import { getShellConfig } from "../../capabilities/tools/support/utils/shell.ts";

void test("BackgroundTaskRegistry registers, tracks, lists and gets status of a background task", async () => {
	const logDir = mkdtempSync(join(tmpdir(), "logician-task-mgr-"));
	const manager = new BackgroundTaskRegistry(logDir);
	const { shell, args } = getShellConfig();

	const child = spawn(shell, [...args, "echo 'hello task'; sleep 0.2; echo 'done task'"], {
		stdio: ["pipe", "pipe", "pipe"],
	});

	const entry = manager.registerTask({
		command: "echo 'hello task'; sleep 0.2; echo 'done task'",
		cwd: logDir,
		child,
	});

	assert.ok(entry.id.startsWith("task-"));
	assert.equal(entry.status, "running");

	const list = manager.listTasks();
	assert.equal(list.length, 1);
	assert.equal(list[0].id, entry.id);

	// Wait for process to complete
	await new Promise(r => setTimeout(r, 400));

	const status = manager.getTaskStatus(entry.id);
	assert.ok(status !== null);
	assert.equal(status?.status, "completed");
	assert.equal(status?.exitCode, 0);
	assert.match(status?.recentOutput ?? "", /hello task/);
	assert.match(status?.recentOutput ?? "", /done task/);

	manager.cleanupAll();
});

void test("BackgroundTaskRegistry allows sending stdin and killing a task", async () => {
	const logDir = mkdtempSync(join(tmpdir(), "logician-task-kill-"));
	const manager = new BackgroundTaskRegistry(logDir);
	const { shell, args } = getShellConfig();

	const child = spawn(shell, [...args, "sleep 10"], {
		stdio: ["pipe", "pipe", "pipe"],
	});

	const entry = manager.registerTask({
		command: "sleep 10",
		cwd: logDir,
		child,
	});

	assert.equal(entry.status, "running");

	const sendRes = manager.sendInput(entry.id, "some text");
	assert.equal(sendRes.success, true);

	const killRes = manager.killTask(entry.id);
	assert.equal(killRes.success, true);

	assert.equal(manager.getTask(entry.id)?.status, "killed");

	manager.cleanupAll();
});
