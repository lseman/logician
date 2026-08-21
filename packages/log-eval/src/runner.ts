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
import path from "node:path";
import { grade } from "./graders.ts";
import { runProcess } from "./process.ts";
import { EVAL_SCHEMA_VERSION, type EvalTask, type EvalTrial } from "./types.ts";

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
				event?: { type?: string; status?: string };
				payload?: { type?: string; status?: string };
				type?: string;
				status?: string;
			};
			const event = entry.event ?? entry.payload ?? entry;
			if (event?.type === "run_finished" || event?.type === "agent_end")
				return event.status === "completed";
			if (event?.type === "done") return entry.status !== "failed";
		}
	} catch {
		return null;
	}
	return null;
}

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
	| "timeToFirstToolMs"
> {
	let toolCalls = 0;
	let permissionRequests = 0;
	let compactions = 0;
	let retries = 0;
	let contextTokens: number | undefined;
	let model: string | undefined;
	let firstTimestamp: number | undefined;
	let firstToolTimestamp: number | undefined;
	for (const line of output.split("\n")) {
		try {
			const entry = JSON.parse(line) as {
				type?: string;
				ts?: number;
				timestamp?: number;
				event?: { type?: string; ts?: number };
				meta?: { context_tokens?: number; model?: string };
			};
			const type = entry.event?.type ?? entry.type;
			const timestamp = entry.event?.ts ?? entry.ts ?? entry.timestamp;
			if (timestamp !== undefined && firstTimestamp === undefined)
				firstTimestamp = timestamp;
			if (type === "tool_use" || type === "tool_execution_start") {
				toolCalls++;
				if (firstToolTimestamp === undefined) firstToolTimestamp = timestamp;
			}
			if (type === "permission_request") permissionRequests++;
			if (type === "compaction") compactions++;
			if (type === "agent_retry_start") retries++;
			if (type === "metadata") {
				contextTokens = entry.meta?.context_tokens;
				model = entry.meta?.model;
			}
		} catch {
			// Non-JSON diagnostics are retained in the artifact but not projected.
		}
	}
	return {
		toolCalls,
		contextTokens,
		model,
		permissionRequests,
		compactions,
		retries,
		timeToFirstToolMs:
			firstTimestamp !== undefined && firstToolTimestamp !== undefined
				? Math.max(0, firstToolTimestamp - firstTimestamp)
				: undefined,
	};
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
			...outputMetrics(agent.stdout),
		},
		trajectoryPath: options.trajectoryPath,
		agentOutput: `${agent.stdout}${agent.stderr}`.slice(-200_000),
	};
	return trial;
}

export function writeTrial(file: string, trial: EvalTrial): void {
	mkdirSync(path.dirname(file), { recursive: true });
	writeFileSync(file, `${JSON.stringify(trial, null, 2)}\n`, "utf8");
}
