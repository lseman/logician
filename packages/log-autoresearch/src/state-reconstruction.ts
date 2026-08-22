/** Rebuilds ExperimentState from the persisted .auto/log.jsonl — used on
 * AutoresearchSession.reload() (e.g. after a restart or context reset). */

import * as fs from "node:fs";
import { readMaxExperiments, resolveWorkDir } from "./config.ts";
import {
	computeConfidence,
	createExperimentState,
	type ExperimentState,
	findBaselineMetric,
} from "./experiment-state.ts";
import { reconstructJsonlState } from "./jsonl.ts";
import { sessionFilePath } from "./paths.ts";

export function autoresearchJsonlPath(dir: string): string {
	return sessionFilePath(dir, "log");
}

export function autoresearchMdPath(dir: string): string {
	return sessionFilePath(dir, "prompt");
}

export function autoresearchChecksPath(dir: string): string {
	return sessionFilePath(dir, "checks");
}

export function autoresearchScriptPath(dir: string): string {
	return sessionFilePath(dir, "measure");
}

export function reconstructState(cwd: string): ExperimentState {
	const state = createExperimentState();

	const workDir = resolveWorkDir(cwd);
	const jsonlPath = autoresearchJsonlPath(workDir);
	const hasPersistedLog = fs.existsSync(jsonlPath);

	try {
		if (hasPersistedLog) {
			const reconstructed = reconstructJsonlState(
				fs.readFileSync(jsonlPath, "utf-8"),
			);
			state.name = reconstructed.name;
			state.metricName = reconstructed.metricName;
			state.metricUnit = reconstructed.metricUnit;
			state.bestDirection = reconstructed.bestDirection;
			state.currentSegment = reconstructed.currentSegment;
			state.results = reconstructed.results.map(r => ({
				...r,
				metrics: { ...r.metrics },
			}));
			state.secondaryMetrics = reconstructed.secondaryMetrics.map(m => ({
				...m,
			}));

			if (state.results.length > 0) {
				state.bestMetric = findBaselineMetric(
					state.results,
					state.currentSegment,
				);
				state.confidence = computeConfidence(
					state.results,
					state.currentSegment,
					state.bestDirection,
				);
			}
		}
	} catch {
		// Fall through
	}

	state.maxExperiments = readMaxExperiments(cwd);
	return state;
}
