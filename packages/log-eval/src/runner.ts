import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import {
	cpSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { grade } from "./graders.ts";
import { runProcess } from "./process.ts";
import {
	EVAL_SCHEMA_VERSION,
	type EvalTask,
	type EvalTrial,
	type HarnessConfigSnapshot,
} from "./types.ts";

/**
 * Snapshot the settings the agent subprocess will actually read from disk.
 * Best-effort: the eval runner doesn't control this file, so a missing or
 * unparsable config yields an empty snapshot rather than failing the trial.
 */
function readHarnessConfigSnapshot(): HarnessConfigSnapshot {
	try {
		const settingsPath = path.join(homedir(), ".logician", "settings.json");
		const raw = JSON.parse(readFileSync(settingsPath, "utf8")) as Record<
			string,
			unknown
		>;
		const mcpServers = raw.mcpServers;
		return {
			model: typeof raw.model === "string" ? raw.model : undefined,
			permissionMode:
				typeof raw.permissionMode === "string" ? raw.permissionMode : undefined,
			toolExecution:
				typeof raw.toolExecution === "string" ? raw.toolExecution : undefined,
			maxIterations:
				typeof raw.maxIterations === "number" ? raw.maxIterations : undefined,
			compaction:
				raw.compaction && typeof raw.compaction === "object"
					? (raw.compaction as Record<string, unknown>)
					: undefined,
			mcpServerNames:
				mcpServers && typeof mcpServers === "object"
					? Object.keys(mcpServers)
					: undefined,
		};
	} catch {
		return {};
	}
}

export function fixtureDigest(root: string): string {
	const hash = createHash("sha256");
	const visit = (directory: string) => {
		for (const name of readdirSync(directory).sort()) {
			if (name === ".git" || name === ".logician" || name === "node_modules")
				continue;
			const file = path.join(directory, name);
			const relative = path.relative(root, file).split(path.sep).join("/");
			const stat = statSync(file);
			if (stat.isDirectory()) visit(file);
			else {
				hash.update(relative);
				hash.update("\0");
				hash.update(readFileSync(file));
				hash.update("\0");
			}
		}
	};
	visit(root);
	return hash.digest("hex");
}

export function prepareTrialWorkspace(
	task: EvalTask,
	workRoot: string,
	trialId: string,
): string {
	const source = path.resolve(task.fixture.repository);
	const workspace = path.join(workRoot, `${task.id}-${trialId}`);
	rmSync(workspace, { recursive: true, force: true });
	mkdirSync(workRoot, { recursive: true });
	cpSync(source, workspace, { recursive: true });
	execFileSync("git", ["init", "-q"], { cwd: workspace });
	execFileSync("git", ["add", "."], { cwd: workspace });
	execFileSync(
		"git",
		[
			"-c",
			"user.name=Logician Eval",
			"-c",
			"user.email=eval@localhost",
			"commit",
			"-qm",
			"fixture baseline",
		],
		{ cwd: workspace },
	);
	return workspace;
}

async function verifyFixture(task: EvalTask, workspace: string): Promise<void> {
	if (task.fixture.revision.startsWith("sha256:")) {
		const actual = fixtureDigest(workspace);
		if (`sha256:${actual}` !== task.fixture.revision)
			throw new Error(`fixture digest mismatch for ${task.id}`);
		return;
	}
	if (task.fixture.revision === "HEAD") return;
	const expected = await runProcess(
		"git",
		["rev-parse", "--verify", `${task.fixture.revision}^{commit}`],
		{ cwd: workspace, timeoutMs: 30_000 },
	);
	const actual = await runProcess("git", ["rev-parse", "HEAD"], {
		cwd: workspace,
		timeoutMs: 30_000,
	});
	if (
		expected.exitCode !== 0 ||
		actual.exitCode !== 0 ||
		expected.stdout.trim() !== actual.stdout.trim()
	)
		throw new Error(
			`workspace HEAD does not match pinned revision ${task.fixture.revision}`,
		);
}

function changedFileCount(workspace: string): Promise<number> {
	return runProcess("git", ["status", "--porcelain"], {
		cwd: workspace,
		timeoutMs: 30_000,
	}).then(
		result =>
			result.stdout
				.split("\n")
				.filter(line => line && !line.slice(3).startsWith(".logician/")).length,
	);
}

/**
 * Read the agent's own completion claim from the terminal `metadata` record's
 * `meta.status` field (see apps/tui/src/app/headless-exec.ts). A prior
 * version of this function checked `entry.status` on the bare `{type: "done"}`
 * record that follows `metadata` in the real stream — a field that record
 * never carries — so this always evaluated to `true` (not failed), silently
 * reporting every real run as agent-declared-complete regardless of outcome.
 */
function readAgentDeclaration(
	trajectoryPath: string | undefined,
	agentOutput = "",
): boolean | null {
	try {
		const content = trajectoryPath
			? readFileSync(trajectoryPath, "utf8")
			: agentOutput;
		if (!content.trim()) return null;
		const lines = content.trim().split("\n");
		for (const line of lines.reverse()) {
			const entry = JSON.parse(line) as {
				type?: string;
				meta?: { status?: string };
			};
			if (entry.type === "metadata") return entry.meta?.status === "completed";
		}
	} catch {
		return null;
	}
	return null;
}

/**
 * Read metrics from the terminal `metadata` record the exec stream always
 * emits last (see apps/tui/src/app/headless-exec.ts). Counters are tallied by
 * the harness itself as events happen, rather than re-derived here by
 * pattern-matching event type strings against the JSONL stream — a prior
 * version of this function matched several event names
 * (`tool_execution_start`, bare `compaction`/`agent_retry_start` records)
 * that the exec stream never actually emits, so those counters were silently
 * always zero.
 */
function outputMetrics(
	output: string,
): Pick<
	EvalTrial["metrics"],
	| "toolCalls"
	| "contextTokens"
	| "model"
	| "permissionRequests"
	| "compactions"
	| "retries"
> {
	for (const line of output.split("\n").reverse()) {
		try {
			const entry = JSON.parse(line) as {
				type?: string;
				meta?: {
					context_tokens?: number;
					model?: string;
					tool_calls?: number;
					permission_requests?: number;
					compactions?: number;
					retries?: number;
				};
			};
			if (entry.type !== "metadata") continue;
			return {
				toolCalls: entry.meta?.tool_calls,
				contextTokens: entry.meta?.context_tokens,
				model: entry.meta?.model,
				permissionRequests: entry.meta?.permission_requests,
				compactions: entry.meta?.compactions,
				retries: entry.meta?.retries,
			};
		} catch {
			// Non-JSON diagnostics are retained in the artifact but not projected.
		}
	}
	return {};
}

/** Re-grade a recorded trajectory against the current deterministic graders. */
export async function replayTrial(
	task: EvalTask,
	workspace: string,
	trajectory: string,
): Promise<EvalTrial> {
	await verifyFixture(task, workspace);
	const started = performance.now();
	const graders = [];
	for (const spec of task.graders) graders.push(await grade(spec, workspace));
	return {
		schemaVersion: EVAL_SCHEMA_VERSION,
		taskId: task.id,
		trialId: randomUUID(),
		startedAt: new Date().toISOString(),
		workspace: path.resolve(workspace),
		agentDeclaredComplete: readAgentDeclaration(undefined, trajectory),
		environmentGradedPass: graders.every(result => result.passed),
		graders,
		metrics: {
			durationMs: Math.round(performance.now() - started),
			exitCode: null,
			timedOut: false,
			changedFiles: await changedFileCount(workspace),
			...outputMetrics(trajectory),
		},
		agentOutput: trajectory.slice(-200_000),
	};
}

export async function runTrial(
	task: EvalTask,
	workspace: string,
	options: { trajectoryPath?: string } = {},
): Promise<EvalTrial> {
	await verifyFixture(task, workspace);
	const startedAt = new Date().toISOString();
	const started = performance.now();
	const harnessConfig = readHarnessConfigSnapshot();
	const logicianRoot = path.resolve(import.meta.dirname, "../../../");
	const expand = (value: string) =>
		value
			.replaceAll("{logicianRoot}", logicianRoot)
			.replaceAll(
				"{logicianEntry}",
				path.join(logicianRoot, "apps/tui/src/index.ts"),
			);
	const agent = await runProcess(
		expand(task.agent.command),
		[...(task.agent.args ?? []).map(expand), task.prompt],
		{
			cwd: workspace,
			timeoutMs: task.limits?.wallTimeMs ?? task.agent.timeoutMs ?? 30 * 60_000,
			env: { ...process.env, LOGICIAN_EVAL_TASK_ID: task.id },
		},
	);
	const graders = [];
	for (const spec of task.graders) graders.push(await grade(spec, workspace));
	const metrics = outputMetrics(agent.stdout);
	const trial: EvalTrial = {
		schemaVersion: EVAL_SCHEMA_VERSION,
		taskId: task.id,
		trialId: randomUUID(),
		startedAt,
		workspace: path.resolve(workspace),
		agentDeclaredComplete: readAgentDeclaration(
			options.trajectoryPath,
			agent.stdout,
		),
		environmentGradedPass:
			agent.exitCode === 0 &&
			!agent.timedOut &&
			graders.every(result => result.passed),
		graders,
		metrics: {
			durationMs: Math.round(performance.now() - started),
			exitCode: agent.exitCode,
			timedOut: agent.timedOut,
			changedFiles: await changedFileCount(workspace),
			...metrics,
		},
		// The trajectory's own reported model is ground truth for what actually
		// ran; the settings-file snapshot only fills in what the stream didn't
		// report (e.g. when metadata.model is absent).
		harnessConfig: { ...harnessConfig, model: metrics.model ?? harnessConfig.model },
		trajectoryPath: options.trajectoryPath,
		agentOutput: `${agent.stdout}${agent.stderr}`.slice(-200_000),
	};
	return trial;
}

export function writeTrial(file: string, trial: EvalTrial): void {
	mkdirSync(path.dirname(file), { recursive: true });
	writeFileSync(file, `${JSON.stringify(trial, null, 2)}\n`, "utf8");
}
