import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { RunKernel } from "@logician/agent-core";
import { runKernelCommand } from "./run-kernel-cli.ts";

function capture() {
	let stdout = "";
	let stderr = "";
	return {
		io: {
			stdout: (text: string) => {
				stdout += text;
			},
			stderr: (text: string) => {
				stderr += text;
			},
		},
		output: () => ({ stdout, stderr }),
	};
}

void test("run replay prints the deterministic kernel projection", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-run-replay-"));
	const kernel = new RunKernel(cwd, "session-a");
	kernel.append(
		{ type: "task_started", rootPrompt: "ship", createdAt: 10 },
		{ taskId: "task-a", runId: "run-a", leaseEpoch: 2 },
	);
	const captured = capture();
	assert.equal(
		runKernelCommand(["replay", "session-a", "--json"], cwd, captured.io),
		0,
	);
	const parsed = JSON.parse(captured.output().stdout) as {
		state: { taskId: string; leaseEpoch: number };
	};
	assert.equal(parsed.state.taskId, "task-a");
	assert.equal(parsed.state.leaseEpoch, 2);
});

void test("run doctor surfaces incomplete side effects and missing ledgers", () => {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-run-doctor-"));
	const kernel = new RunKernel(cwd, "session-b");
	const ids = { taskId: "task-b", runId: "run-b", leaseEpoch: 1 };
	kernel.append(
		{ type: "task_started", rootPrompt: "recover", createdAt: 10 },
		ids,
	);
	kernel.append(
		{
			type: "operation_intent_recorded",
			operationId: "effect",
			toolName: "write",
			argumentsDigest: "digest",
			idempotencyKey: "key",
			recovery: "at_most_once_unknown",
		},
		{ ...ids, operationId: "effect" },
	);
	const captured = capture();
	assert.equal(runKernelCommand(["doctor", "session-b"], cwd, captured.io), 1);
	assert.match(captured.output().stdout, /action=quarantine/);

	const missing = capture();
	assert.equal(runKernelCommand(["replay", "missing"], cwd, missing.io), 1);
	assert.match(missing.output().stderr, /not found/);
});
