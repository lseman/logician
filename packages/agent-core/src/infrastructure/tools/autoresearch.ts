// ── autoresearch tools ───────────────────────────────────────────────────────
// init_experiment / run_experiment / log_experiment — thin Tool wrappers
// around @logician/autoresearch's AutoresearchSession, which owns all the
// actual logic (state, .auto/ file I/O, git commit/revert, confidence
// scoring). One AutoresearchSession is shared across all three tools per
// session — see apps/tui/src/app/research-manager.ts for how the TUI
// constructs and threads it through.
//
// Not gated readOnly (same as bash/write_file): run_experiment executes
// arbitrary shell commands and log_experiment runs git commit/checkout, so
// both go through the same permission prompts as any other mutating tool.

import {
	type AutoresearchSession,
	INIT_EXPERIMENT_PARAMETERS,
	LOG_EXPERIMENT_PARAMETERS,
	RUN_EXPERIMENT_PARAMETERS,
} from "@logician/autoresearch";
import type { Tool, ToolResult } from "../../core/types/index.ts";

export function createAutoresearchTools(session: AutoresearchSession): Tool[] {
	const init_experiment: Tool = {
		name: "init_experiment",
		label: "Init Experiment",
		description:
			"Initialize the experiment session. Call once before the first run_experiment to set the name, primary metric, unit, and direction. Writes the config header to .auto/log.jsonl.",
		parameters: INIT_EXPERIMENT_PARAMETERS,
		execute: async (args): Promise<ToolResult> => {
			return session.initExperiment(args);
		},
	};

	const run_experiment: Tool = {
		name: "run_experiment",
		label: "Run Experiment",
		description:
			"Run a shell command as an experiment. Times wall-clock duration, captures output, detects pass/fail. Output is truncated. If METRIC lines are found, they are parsed automatically.",
		parameters: RUN_EXPERIMENT_PARAMETERS,
		execute: async (args): Promise<ToolResult> => {
			return session.runExperiment(args);
		},
	};

	const log_experiment: Tool = {
		name: "log_experiment",
		label: "Log Experiment",
		description:
			"Record an experiment result. Tracks metrics, auto-commits on 'keep', auto-reverts on 'discard'/'crash'/'checks_failed'.",
		parameters: LOG_EXPERIMENT_PARAMETERS,
		execute: async (args): Promise<ToolResult> => {
			return session.logExperiment(args);
		},
	};

	return [init_experiment, run_experiment, log_experiment];
}
