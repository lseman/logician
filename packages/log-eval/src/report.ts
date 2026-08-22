import type { EvalReport, EvalTrial } from "./types.ts";
import { EVAL_SCHEMA_VERSION } from "./types.ts";

export function buildReport(trials: EvalTrial[]): EvalReport {
	const durations = trials
		.map(trial => trial.metrics.durationMs)
		.sort((a, b) => a - b);
	const passed = trials.filter(trial => trial.environmentGradedPass).length;
	const middle = Math.floor(durations.length / 2);
	const medianDurationMs =
		durations.length === 0
			? 0
			: durations.length % 2
				? durations[middle]
				: Math.round((durations[middle - 1] + durations[middle]) / 2);
	return {
		schemaVersion: EVAL_SCHEMA_VERSION,
		generatedAt: new Date().toISOString(),
		trials,
		summary: {
			passed,
			failed: trials.length - passed,
			passRate: trials.length === 0 ? 0 : passed / trials.length,
			medianDurationMs,
		},
	};
}

export function reportMarkdown(report: EvalReport): string {
	const lines = [
		"# Logician agent evaluation",
		"",
		`Generated: ${report.generatedAt}`,
		"",
		`Environment-graded pass rate: **${report.summary.passed}/${report.trials.length} (${(report.summary.passRate * 100).toFixed(1)}%)**`,
		`Median duration: **${report.summary.medianDurationMs} ms**`,
		"",
		"| Task | Trial | Environment | Agent declaration | Duration | Model |",
		"|---|---|---:|---:|---:|---|",
	];
	for (const trial of report.trials) {
		lines.push(
			`| ${trial.taskId} | ${trial.trialId} | ${trial.environmentGradedPass ? "pass" : "fail"} | ${trial.agentDeclaredComplete === null ? "unknown" : trial.agentDeclaredComplete ? "complete" : "incomplete"} | ${trial.metrics.durationMs} ms | ${trial.harnessConfig?.model ?? "unknown"} |`,
		);
	}
	return `${lines.join("\n")}\n`;
}
