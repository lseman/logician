import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { grade } from "../graders.ts";
import { runProcess } from "../process.ts";
import { buildReport } from "../report.ts";
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
});
