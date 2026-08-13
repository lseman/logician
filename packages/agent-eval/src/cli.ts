#!/usr/bin/env bun
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { buildReport, reportMarkdown } from "./report.ts";
import { prepareTrialWorkspace, runTrial } from "./runner.ts";
import { validateCorpus } from "./schema.ts";

function load(file: string) {
	return validateCorpus(JSON.parse(readFileSync(file, "utf8")));
}

async function main(): Promise<void> {
	const [command, corpusFile, ...rest] = process.argv.slice(2);
	if (!command || !corpusFile)
		throw new Error(
			"usage: logician-eval <validate|run> <corpus.json> [options]",
		);
	const corpus = load(path.resolve(corpusFile));
	if (command === "validate") {
		console.log(
			`valid corpus: ${corpus.name} (${corpus.tasks.length} task(s))`,
		);
		return;
	}
	if (command !== "run") throw new Error(`unknown command: ${command}`);
	const workspaceFlag = rest.indexOf("--workspace");
	const outputFlag = rest.indexOf("--output");
	const taskFlag = rest.indexOf("--task");
	const trialsFlag = rest.indexOf("--trials");
	const workRootFlag = rest.indexOf("--work-root");
	const workspace = path.resolve(
		workspaceFlag >= 0 ? rest[workspaceFlag + 1] : process.cwd(),
	);
	const selected =
		taskFlag >= 0
			? corpus.tasks.filter(task => task.id === rest[taskFlag + 1])
			: corpus.tasks;
	if (selected.length === 0) throw new Error("no matching tasks");
	const trialCount = trialsFlag >= 0 ? Number(rest[trialsFlag + 1]) : 1;
	if (!Number.isInteger(trialCount) || trialCount < 1 || trialCount > 20)
		throw new Error("--trials must be an integer from 1 to 20");
	const workRoot = path.resolve(
		workRootFlag >= 0
			? rest[workRootFlag + 1]
			: "outputs/agent-eval-workspaces",
	);
	const trials = [];
	for (const task of selected) {
		for (let index = 1; index <= trialCount; index++) {
			const trialWorkspace =
				workRootFlag >= 0
					? prepareTrialWorkspace(task, workRoot, String(index))
					: workspace;
			trials.push(await runTrial(task, trialWorkspace));
		}
	}
	const report = buildReport(trials);
	const output = path.resolve(
		outputFlag >= 0 ? rest[outputFlag + 1] : "outputs/agent-eval-report.json",
	);
	mkdirSync(path.dirname(output), { recursive: true });
	const artifactRoot = output.replace(/\.json$/, ".artifacts");
	mkdirSync(artifactRoot, { recursive: true });
	for (const trial of trials) {
		const artifact = path.join(
			artifactRoot,
			`${trial.taskId}-${trial.trialId}.jsonl`,
		);
		writeFileSync(artifact, trial.agentOutput ?? "", "utf8");
		trial.trajectoryPath = artifact;
		trial.agentOutput = undefined;
	}
	writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
	writeFileSync(
		output.replace(/\.json$/, ".md"),
		reportMarkdown(report),
		"utf8",
	);
	console.log(
		`wrote ${output}; pass rate ${(report.summary.passRate * 100).toFixed(1)}%`,
	);
	if (report.summary.failed > 0) process.exitCode = 1;
}

main().catch(error => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
});
