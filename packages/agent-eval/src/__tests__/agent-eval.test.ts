import { describe, expect, test } from "bun:test";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { grade } from "../graders.ts";
import { runProcess } from "../process.ts";
import { buildReport } from "../report.ts";
import { replayTrial } from "../runner.ts";
import { validateCorpus } from "../schema.ts";

describe("agent eval", () => {
	test("validates versioned corpora and rejects duplicate ids", () => {
		const task = {
			schemaVersion: 1,
			id: "one",
			title: "One",
			kind: "bugfix",
			prompt: "Fix it",
			fixture: { repository: "repo", revision: "abc" },
			agent: { command: "true" },
			graders: [{ id: "test", type: "command", command: "true" }],
		};
		expect(
			validateCorpus({ schemaVersion: 1, name: "sample", tasks: [task] }).tasks,
		).toHaveLength(1);
		expect(() =>
			validateCorpus({ schemaVersion: 1, name: "sample", tasks: [task, task] }),
		).toThrow("duplicate task id");
	});

	test("grades filesystem evidence without allowing path traversal", async () => {
		const workspace = mkdtempSync(path.join(tmpdir(), "logician-eval-"));
		writeFileSync(path.join(workspace, "result.txt"), "verified outcome\n");
		expect(
			(
				await grade(
					{
						id: "content",
						type: "file_contains",
						path: "result.txt",
						pattern: "verified",
					},
					workspace,
				)
			).passed,
		).toBe(true);
		expect(
			(
				await grade(
					{ id: "escape", type: "file_absent", path: "../secret" },
					workspace,
				)
			).passed,
		).toBe(false);
	});

	test("reports independently graded outcomes", () => {
		const report = buildReport([
			{
				schemaVersion: 1,
				taskId: "task",
				trialId: "trial",
				startedAt: new Date(0).toISOString(),
				workspace: "/tmp",
				agentDeclaredComplete: true,
				environmentGradedPass: false,
				graders: [],
				metrics: {
					durationMs: 10,
					exitCode: 0,
					timedOut: false,
					changedFiles: 0,
				},
			},
		]);
		expect(report.summary.passRate).toBe(0);
		expect(report.summary.failed).toBe(1);
	});

	test("kills a timed-out process group", async () => {
		const result = await runProcess("bash", ["-c", "sleep 30 & wait"], {
			cwd: process.cwd(),
			timeoutMs: 20,
		});
		expect(result.timedOut).toBe(true);
		expect(result.durationMs).toBeLessThan(2500);
	});

	test("replays a recorded trajectory against current graders", async () => {
		const workspace = mkdtempSync(path.join(tmpdir(), "logician-replay-"));
		writeFileSync(path.join(workspace, "result.txt"), "verified\n");
		execFileSync("git", ["init", "-q"], { cwd: workspace });
		execFileSync("git", ["add", "."], { cwd: workspace });
		execFileSync(
			"git",
			[
				"-c",
				"user.name=Eval",
				"-c",
				"user.email=eval@localhost",
				"commit",
				"-qm",
				"baseline",
			],
			{ cwd: workspace },
		);
		const task = {
			schemaVersion: 1 as const,
			id: "replay",
			title: "Replay",
			kind: "bugfix" as const,
			prompt: "verify",
			fixture: { repository: workspace, revision: "HEAD" },
			agent: { command: "true" },
			graders: [
				{
					id: "result",
					type: "file_contains" as const,
					path: "result.txt",
					pattern: "verified",
				},
			],
		};
		const trajectory = [
			JSON.stringify({ type: "agent_start", ts: 100 }),
			JSON.stringify({ type: "tool_execution_start", ts: 140 }),
			JSON.stringify({ type: "permission_request", ts: 141 }),
			JSON.stringify({ type: "agent_end", status: "completed", ts: 180 }),
		].join("\n");
		const trial = await replayTrial(task, workspace, trajectory);
		expect(trial.environmentGradedPass).toBe(true);
		expect(trial.agentDeclaredComplete).toBe(true);
		expect(trial.metrics.toolCalls).toBe(1);
		expect(trial.metrics.permissionRequests).toBe(1);
		expect(trial.metrics.timeToFirstToolMs).toBe(40);
	});
});
