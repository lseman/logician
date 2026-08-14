/**
 * Deterministic compaction summary for EoH sessions.
 *
 * Replaces the default LLM-generated summary with a synthesized view of
 * persisted state — problem definition, population stats, and recent runs.
 * Everything that matters between iterations already lives on disk, so we
 * skip the LLM call entirely and keep the summary lossless on what counts.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { reconstructEohState } from "./jsonl.ts";
import { sessionFilePath } from "./paths.ts";
import type { EohRunEntry, ReconstructedEohState } from "./types.ts";

const RECENT_RUN_LIMIT = 50;

type RunStatus = EohRunEntry["status"];
type StatusCounts = Record<RunStatus, number>;

export interface EohSummaryPaths {
	workDir: string;
	jsonlPath: string;
	problemPath: string;
	promptPath: string;
}

export function eohSummaryPathsFor(workDir: string): EohSummaryPaths {
	return {
		workDir,
		jsonlPath: sessionFilePath(workDir, "log"),
		problemPath: sessionFilePath(workDir, "problem"),
		promptPath: sessionFilePath(workDir, "prompt"),
	};
}

/**
 * Build the full compaction summary text from persisted EoH state.
 * Returns a markdown string that is itself the entire compaction summary.
 */
export function buildEohCompactionSummary(paths: EohSummaryPaths): string {
	const state = loadState(paths.jsonlPath);
	const sections = [
		headerSection(),
		sessionSection(state),
		problemSection(paths.workDir, paths.problemPath),
		rulesSection(paths.workDir, paths.promptPath),
		recentRunsSection(state, paths.workDir, paths.jsonlPath),
		nextStepSection(),
	];
	return sections.filter(Boolean).join("\n\n");
}

function loadState(jsonlPath: string): ReconstructedEohState {
	return reconstructEohState(readFileOrEmpty(jsonlPath));
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

function headerSection(): string {
	return [
		"# EoH Compaction Summary",
		"",
		"The conversation history was discarded; the persisted EoH state below is the source of truth.",
		"Continue the evolution loop using only what is included here plus the live tools.",
	].join("\n");
}

function sessionSection(state: ReconstructedEohState): string {
	const runs = currentSegmentRuns(state);
	const lines = [
		"## Session",
		"",
		`Goal: ${state.name ?? "—"}`,
		`Population: ${state.populationSize}`,
		`Max generations: ${state.maxGenerations === 0 ? "unlimited" : state.maxGenerations}`,
		runCountLine(runs),
		...baselineAndBestLines(runs),
	];
	return lines.join("\n");
}

function currentSegmentRuns(state: ReconstructedEohState): EohRunEntry[] {
	return state.results.filter(run => run.segment === state.currentSegment);
}

function runCountLine(runs: EohRunEntry[]): string {
	if (runs.length === 0) return "Runs so far: 0";
	const counts = countByStatus(runs);
	const parts = [
		`${counts.keep} keep`,
		counts.discard ? `${counts.discard} discard` : "",
		counts.crash ? `${counts.crash} crash` : "",
	].filter(Boolean);
	return `Runs so far: ${runs.length} (${parts.join(" · ")})`;
}

function countByStatus(runs: EohRunEntry[]): StatusCounts {
	const counts: StatusCounts = {
		keep: 0,
		discard: 0,
		crash: 0,
	};
	for (const run of runs) counts[run.status]++;
	return counts;
}

function baselineAndBestLines(runs: EohRunEntry[]): string[] {
	const baseline = runs[0];
	if (!baseline) return [];
	const lines = [
		`Baseline (#${baseline.run}): ${formatMetric(baseline.fitness)}`,
	];
	const best = bestRun(runs);
	if (best && best.run !== baseline.run) {
		lines.push(
			`Best     (#${best.run}): ${formatMetric(best.fitness)}${formatDelta(best.fitness, baseline.fitness)}`,
		);
	}
	return lines;
}

function bestRun(runs: EohRunEntry[]): EohRunEntry | null {
	const kept = runs.filter(
		run => run.status === "keep" && Number.isFinite(run.fitness),
	);
	if (kept.length === 0) return null;
	return kept.reduce((best, run) => (run.fitness > best.fitness ? run : best));
}

function formatMetric(value: number): string {
	if (!Number.isFinite(value)) return "—";
	if (Number.isInteger(value)) return String(value);
	return value.toFixed(4);
}

function formatDelta(value: number, baseline: number): string {
	if (baseline === 0 || value === baseline) return "";
	const pct = ((value - baseline) / Math.abs(baseline)) * 100;
	const sign = pct > 0 ? "+" : "";
	return ` (${sign}${pct.toFixed(1)}%)`;
}

function readablePath(workDir: string, filePath: string): string {
	const relative = path.relative(workDir, filePath);
	if (!relative || relative.startsWith("..") || path.isAbsolute(relative))
		return filePath;
	return relative;
}

function problemSection(workDir: string, problemPath: string): string {
	const content = readTrimmedFile(problemPath);
	if (!content) return "";
	return `## Problem Definition (${readablePath(workDir, problemPath)})\n\n${content}`;
}

function rulesSection(workDir: string, promptPath: string): string {
	const content = readTrimmedFile(promptPath);
	if (!content) return "";
	return `## Evolution Rules (${readablePath(workDir, promptPath)})\n\n${content}`;
}

function recentRunsSection(
	state: ReconstructedEohState,
	workDir: string,
	jsonlPath: string,
): string {
	const runs = state.results.slice(-RECENT_RUN_LIMIT);
	if (runs.length === 0) {
		return "## Recent Runs\n\nNo runs yet — start with the first heuristic.";
	}
	const lines = runs.map(run => formatRunLine(run));
	return [
		`## Recent Runs (last ${runs.length})`,
		"",
		"Format: `#run status fitness (delta) | desc | operator: ...`",
		"",
		...lines,
		"",
		`If you need more details, read additional lines from ${readablePath(workDir, jsonlPath)}.`,
	].join("\n");
}

function nextStepSection(): string {
	return [
		"## Next Step",
		"",
		"Run the next generation of evolution. The engine will apply all 5 operators (E1, E2, M1, M2, M3) to generate new candidates, evaluate them, and select the next population.",
	].join("\n");
}

// ---------------------------------------------------------------------------
// Recent runs
// ---------------------------------------------------------------------------

function formatRunLine(run: EohRunEntry): string {
	const head = `#${run.run} ${padStatus(run.status)} ${formatMetric(run.fitness)}`;
	const parts = [head, formatDescription(run), `operator: ${run.createdBy}`];
	return parts.filter(Boolean).join(" | ");
}

function padStatus(status: EohRunEntry["status"]): string {
	return status.padEnd(5);
}

function formatDescription(run: EohRunEntry): string {
	return run.description ? `desc: ${run.description}` : "";
}

// ---------------------------------------------------------------------------
// File IO
// ---------------------------------------------------------------------------

function readTrimmedFile(filePath: string): string {
	return readFileOrEmpty(filePath).trim();
}

function readFileOrEmpty(filePath: string): string {
	if (!fs.existsSync(filePath)) return "";
	try {
		return fs.readFileSync(filePath, "utf-8");
	} catch {
		return "";
	}
}
