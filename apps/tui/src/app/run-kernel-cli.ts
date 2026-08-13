import { existsSync } from "node:fs";
import { migrateLegacyRunData, RunKernel } from "@logician/agent-core";

export interface RunKernelCommandIO {
	stdout: (text: string) => void;
	stderr: (text: string) => void;
}

function usage(): string {
	return [
		"Usage:",
		"  logician run replay <session-id> [--json]",
		"  logician run doctor <session-id> [--json]",
		"  logician run migrate <session-id> [--json]",
	].join("\n");
}

/** Execute read-only Run Kernel operator commands without starting the TUI. */
export function runKernelCommand(
	args: string[],
	cwd: string,
	io: RunKernelCommandIO,
): number {
	const action = args[0];
	const sessionId = args[1];
	const json = args.includes("--json");
	if (
		(action !== "replay" && action !== "doctor" && action !== "migrate") ||
		!sessionId
	) {
		io.stderr(`${usage()}\n`);
		return 2;
	}
	const kernel = new RunKernel(cwd, sessionId);
	if (action === "migrate") {
		const migrated = migrateLegacyRunData(kernel, cwd, sessionId);
		if (json)
			io.stdout(
				`${JSON.stringify({ sessionId, migrated, file: kernel.filePath })}\n`,
			);
		else
			io.stdout(
				migrated
					? `Migrated legacy execution data for ${sessionId} to ${kernel.filePath}\n`
					: `No legacy execution data migrated for ${sessionId}.\n`,
			);
		return migrated ? 0 : 1;
	}
	if (!existsSync(kernel.filePath)) {
		io.stderr(
			`Run Kernel ledger not found for session ${sessionId}: ${kernel.filePath}\n`,
		);
		return 1;
	}
	if (action === "doctor") {
		const report = kernel.doctor();
		if (json) io.stdout(`${JSON.stringify(report, null, 2)}\n`);
		else {
			const lines = [
				`Run Kernel doctor: ${sessionId}`,
				`  file: ${report.file}`,
				`  events: ${report.events}`,
				`  last valid sequence: ${report.lastValidSequence}`,
				`  torn final record: ${report.truncatedFinalRecord ? "yes" : "no"}`,
				`  parse errors: ${report.parseErrors.length}`,
				`  invariant violations: ${report.violations.length}`,
				`  incomplete operations: ${report.incompleteOperations.length}`,
				`  orphaned subagents: ${report.orphanedSubagents.length}`,
			];
			for (const operation of report.incompleteOperations) {
				lines.push(
					`    ${operation.operationId}: ${operation.status}; recovery=${operation.recovery}; action=${operation.recommendedAction}`,
				);
			}
			for (const violation of report.violations) {
				lines.push(`    violation ${violation.code}: ${violation.message}`);
			}
			for (const child of report.orphanedSubagents)
				lines.push(
					`    orphaned subagent ${child.agentId}: ${child.agent}; last=${child.lastEventType ?? "unknown"}`,
				);
			io.stdout(`${lines.join("\n")}\n`);
		}
		return report.parseErrors.length ||
			report.violations.length ||
			report.truncatedFinalRecord ||
			report.incompleteOperations.length ||
			report.orphanedSubagents.length
			? 1
			: 0;
	}

	const replay = kernel.snapshot();
	const result = {
		sessionId,
		file: kernel.filePath,
		events: kernel.loadEvents().length,
		state: replay.state,
		violations: replay.violations,
	};
	if (json) io.stdout(`${JSON.stringify(result, null, 2)}\n`);
	else {
		io.stdout(
			`${[
				`Run Kernel replay: ${sessionId}`,
				`  events: ${result.events}`,
				`  task: ${result.state.taskId ?? "none"}`,
				`  run: ${result.state.runId ?? "none"}`,
				`  status: ${result.state.status}`,
				`  sequence: ${result.state.lastSequence}`,
				`  lease epoch: ${result.state.leaseEpoch}`,
				`  provider/tool calls: ${result.state.budgets.provider_call}/${result.state.budgets.tool_call}`,
				`  operations: ${Object.keys(result.state.operations).length}`,
				`  permission decisions: ${result.state.permissionDecisions.length}`,
				`  subagents active/total: ${Object.values(result.state.subagents).filter(child => child.status === "running").length}/${Object.keys(result.state.subagents).length}`,
				`  queued guidance: ${result.state.queues.steering.length + result.state.queues.followUp.length + result.state.queues.nextTurn.length}`,
				`  violations: ${result.violations.length}`,
			].join("\n")}\n`,
		);
	}
	return replay.violations.length ? 1 : 0;
}
