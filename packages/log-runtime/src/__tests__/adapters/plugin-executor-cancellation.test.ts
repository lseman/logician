import { test } from "bun:test";
import assert from "node:assert/strict";
import { CancellationError } from "@logician/log-core/runtime";
import { runShellCommand } from "../../adapters/claude-code/plugin-executor.ts";

void test("plugin shell deadlines use the shared cancellation module", async () => {
	await assert.rejects(
		runShellCommand(
			`exec ${process.execPath} -e "setTimeout(() => {}, 10000)"`,
			{
				cwd: process.cwd(),
				env: process.env,
				input: "",
				timeoutMs: 10,
			},
		),
		(error: unknown) =>
			error instanceof CancellationError && error.kind === "timeout",
	);
});
