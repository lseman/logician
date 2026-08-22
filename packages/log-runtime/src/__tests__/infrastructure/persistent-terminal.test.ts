import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TerminalPool } from "../../capabilities/tools/support/utils/terminal-pool.ts";

void test("TerminalPool preserves environment variables across calls", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-term-env-"));
	const manager = new TerminalPool();

	const res1 = await manager.execute("test-term", "export LOGICIAN_TEST_VAR='alpha_beta_123'", { cwd });
	assert.equal(res1.status, "completed");
	assert.equal(res1.exitCode, 0);

	const res2 = await manager.execute("test-term", "echo \"VAL=$LOGICIAN_TEST_VAR\"", { cwd });
	assert.equal(res2.status, "completed");
	assert.equal(res2.exitCode, 0);
	assert.match(res2.content, /VAL=alpha_beta_123/);

	manager.closeAll();
});

void test("TerminalPool preserves directory navigation across calls", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-term-dir-"));
	const manager = new TerminalPool();

	const res1 = await manager.execute("dir-term", "mkdir -p my_sub_dir && cd my_sub_dir", { cwd });
	assert.equal(res1.status, "completed");

	const res2 = await manager.execute("dir-term", "pwd", { cwd });
	assert.equal(res2.status, "completed");
	assert.match(res2.content, /my_sub_dir/);

	manager.closeAll();
});

void test("TerminalPool accurately captures exit codes", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "logician-term-exit-"));
	const manager = new TerminalPool();

	const res1 = await manager.execute("exit-term", "exit 42 || false", { cwd });
	// in subshell { exit 42; } __LOGICIAN_EXIT=$?
	assert.equal(res1.exitCode, 42);
	assert.equal(res1.status, "failed");

	manager.closeAll();
});
