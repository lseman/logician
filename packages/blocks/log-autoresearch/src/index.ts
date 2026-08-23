/**
 * autoresearch — session logic
 *
 * Generic autonomous experiment loop infrastructure: try an idea, benchmark
 * it, keep improvements, revert regressions, repeat. Ported from
 * pi-autoresearch.
 *
 * This module owns the pure session state/logic — the `AutoresearchSession`
 * class below exposes plain methods (initExperiment/runExperiment/
 * logExperiment/handleCommand) with no dependency on any particular tool-
 * registration or extension-API shape. Callers wire these into real tools
 * and slash commands (see packages/coding-agent/src/tools/autoresearch.ts
 * and apps/tui/src/app/research-manager.ts).
 */

import { execFile as execFileCb, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { promisify } from "node:util";
import {
	autoresearchSummaryPathsFor,
	buildAutoresearchCompactionSummary,
} from "./compaction.ts";
import {
	readMaxExperiments,
	resolveWorkDir,
	validateWorkDir,
} from "./config.ts";
import {
	broadcastDashboardUpdate,
	exportDashboard,
	type NotifyFn,
	stopDashboardServer,
} from "./dashboard-server.ts";
import {
	type ASI,
	cloneExperimentState,
	computeConfidence,
	createExperimentState,
	currentResults,
	type ExperimentResult,
	type ExperimentState,
	findBaselineMetric,
	findBestMetric,
	formatNum,
} from "./experiment-state.ts";
import {
	appendHookLogEntryIfConfigured,
	type HookPayload,
	runHook,
	type SessionSnapshot,
	steerMessageFor,
} from "./hooks.ts";
import { isAutoresearchRunEntry, parseJsonlEntry } from "./jsonl.ts";
import { parseMetricLines } from "./metrics.ts";
import { AUTO_DIR, ensureParentDir, sessionFileCandidates } from "./paths.ts";
import {
	appendOutputTail,
	formatSize,
	killTree,
	runScript,
	truncateTail,
} from "./process.ts";
import {
	autoresearchChecksPath,
	autoresearchJsonlPath,
	autoresearchMdPath,
	autoresearchScriptPath,
	reconstructState,
} from "./state-reconstruction.ts";

const execFile = promisify(execFileCb);

export type { NotifyFn, NotifyLevel } from "./dashboard-server.ts";

/** Result returned by every AutoresearchSession method — mirrors the real
 * agent-core ToolResult shape (content/details/isError) so callers can pass
 * it straight through as a tool's return value. */
export interface AutoresearchResult {
	content: string;
	details?: Record<string, unknown>;
	isError?: boolean;
}

/** Render-ready snapshot for a persistent status widget — see
 * AutoresearchSession.getWidgetSummary(). */
export interface AutoresearchWidgetSummary {
	active: boolean;
	name: string | null;
	metricName: string;
	metricUnit: string;
	/** First run's metric in the current segment — the point of comparison,
	 * not necessarily the best value seen (that's bestMetric below). Despite
	 * the name, the internal ExperimentState.bestMetric field this is sourced
	 * from is actually the baseline; kept distinct here to avoid repeating
	 * that confusion in every consumer. */
	baselineMetric: number | null;
	/** Best (min/max per bestDirection) metric among all "keep" runs in the
	 * current segment. This is what a dashboard/widget should highlight. */
	bestMetric: number | null;
	bestDirection: "lower" | "higher";
	runCount: number;
	confidence: number | null;
	running: { command: string; elapsedMs: number } | null;
	maxExperiments: number | null;
}

/** One row of the full-screen dashboard's results table. */
export interface AutoresearchDashboardRow {
	run: number;
	commit: string;
	status: "keep" | "discard" | "crash" | "checks_failed";
	metric: number;
	metricFormatted: string;
	description: string;
	timestamp: number;
	isBest: boolean;
}

/** Full-screen dashboard snapshot — see AutoresearchSession.getDashboardData(). */
export interface AutoresearchDashboardData {
	summary: AutoresearchWidgetSummary | null;
	rows: AutoresearchDashboardRow[];
}

export { formatNum } from "./experiment-state.ts";

// ---------------------------------------------------------------------------
// Experiment output limits (sent to LLM — keep small to save context)
// ---------------------------------------------------------------------------
const EXPERIMENT_MAX_LINES = 10;
const EXPERIMENT_MAX_BYTES = 4 * 1024; // 4KB

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

interface ExperimentStateForLog extends Record<string, unknown> {
	experiment: ExperimentResult;
	state: ExperimentState;
	wallClockSeconds: number | null;
}

interface AutoresearchRuntime {
	autoresearchMode: boolean;
	experimentsThisSession: number;
	autoResumeTurns: number;
	lastRunChecks: { pass: boolean; output: string; duration: number } | null;
	lastRunDuration: number | null;
	runningExperiment: { startedAt: number; command: string } | null;
	state: ExperimentState;
}

function createSessionRuntime(): AutoresearchRuntime {
	return {
		autoresearchMode: false,
		experimentsThisSession: 0,
		autoResumeTurns: 0,
		lastRunChecks: null,
		lastRunDuration: null,
		runningExperiment: null,
		state: createExperimentState(),
	};
}

// ---------------------------------------------------------------------------
// Session entry point
// ---------------------------------------------------------------------------

const BENCHMARK_GUARDRAIL =
	"Be careful not to overfit to the benchmarks and do not cheat on the benchmarks.";

/** JSON-schema `parameters` blocks for the three tools, exported so callers
 * building real Tool objects don't have to duplicate them. */
export const INIT_EXPERIMENT_PARAMETERS = {
	type: "object",
	properties: {
		name: {
			type: "string",
			description: "Human-readable name for this experiment session",
		},
		metric_name: {
			type: "string",
			description: "Display name for the primary metric",
		},
		metric_unit: {
			type: "string",
			description:
				'Unit for the primary metric (e.g. "µs", "ms", "s", "kb", "mb")',
		},
		direction: {
			type: "string",
			description: 'Whether "lower" or "higher" is better. Default: "lower".',
		},
	},
	required: ["name", "metric_name"],
} as const;

export const RUN_EXPERIMENT_PARAMETERS = {
	type: "object",
	properties: {
		command: { type: "string", description: "Shell command to run" },
		timeout_seconds: {
			type: "number",
			description: "Kill after this many seconds (default: 600)",
		},
		checks_timeout_seconds: {
			type: "number",
			description:
				"Kill .auto/checks.sh after this many seconds (default: 300)",
		},
	},
	required: ["command"],
} as const;

export const LOG_EXPERIMENT_PARAMETERS = {
	type: "object",
	properties: {
		commit: {
			type: "string",
			description: "Git commit hash (short, 7 chars)",
		},
		metric: {
			type: "number",
			description: "The primary optimization metric value",
		},
		status: {
			type: "string",
			enum: ["keep", "discard", "crash", "checks_failed"],
			description: "Result status",
		},
		description: {
			type: "string",
			description: "Short description of what this experiment tried",
		},
		metrics: {
			type: "object",
			description: "Additional metrics to track as { name: value } pairs",
			additionalProperties: { type: "number" },
		},
		force: {
			type: "boolean",
			description: "Force add new secondary metric",
		},
		asi: {
			type: "object",
			description: "Actionable Side Information — structured diagnostics",
			additionalProperties: true,
		},
	},
	required: ["commit", "metric", "status", "description"],
} as const;

/**
 * Owns one autoresearch session's runtime state (running experiment,
 * confidence score, .auto/ file paths) for a given working directory.
 * No dependency on any tool-registration or extension-API shape — callers
 * (real Tool.execute functions, the /autoresearch command handler, bridge
 * event handlers) call these methods directly and adapt the result shape
 * to whatever their integration point needs.
 */
export class AutoresearchSession {
	private runtime: AutoresearchRuntime;
	private readonly cwd: string;
	private readonly notify: NotifyFn;

	constructor(cwd: string, notify: NotifyFn = () => {}) {
		this.cwd = cwd;
		this.notify = notify;
		this.runtime = createSessionRuntime();
	}

	/** Call once when a session/turn starts, to reload persisted state from
	 * .auto/log.jsonl (e.g. after a restart or context reset). */
	reload(): void {
		this.runtime.state = reconstructState(this.cwd);
	}

	isActive(): boolean {
		return this.runtime.autoresearchMode;
	}

	getRuntime(): Readonly<AutoresearchRuntime> {
		return this.runtime;
	}

	/**
	 * Compact, render-ready snapshot for a persistent status widget. Returns
	 * null when there's nothing worth showing (mode off and no results yet),
	 * so the widget can render zero lines rather than an empty-state message.
	 */
	getWidgetSummary(): AutoresearchWidgetSummary | null {
		const state = this.runtime.state;
		if (!this.runtime.autoresearchMode && state.results.length === 0) {
			return null;
		}
		const runCount = currentResults(state.results, state.currentSegment).length;
		return {
			active: this.runtime.autoresearchMode,
			name: state.name,
			metricName: state.metricName,
			metricUnit: state.metricUnit,
			baselineMetric: state.bestMetric,
			bestMetric: findBestMetric(
				state.results,
				state.currentSegment,
				state.bestDirection,
			),
			bestDirection: state.bestDirection,
			runCount,
			confidence: state.confidence,
			running: this.runtime.runningExperiment
				? {
						command: this.runtime.runningExperiment.command,
						elapsedMs: Date.now() - this.runtime.runningExperiment.startedAt,
					}
				: null,
			maxExperiments: state.maxExperiments,
		};
	}

	/** Full results table for the fullscreen dashboard overlay — every run
	 * in the current segment, most recent last, with the current best-metric
	 * run flagged for highlighting. */
	getDashboardData(): AutoresearchDashboardData {
		const state = this.runtime.state;
		const segmentResults = currentResults(state.results, state.currentSegment);
		const bestMetric = findBestMetric(
			state.results,
			state.currentSegment,
			state.bestDirection,
		);
		const rows: AutoresearchDashboardRow[] = segmentResults.map(
			(result, i) => ({
				run: i + 1,
				commit: result.commit,
				status: result.status,
				metric: result.metric,
				metricFormatted: formatNum(result.metric, state.metricUnit),
				description: result.description,
				timestamp: result.timestamp,
				isBest:
					result.status === "keep" &&
					bestMetric !== null &&
					result.metric === bestMetric,
			}),
		);
		return { summary: this.getWidgetSummary(), rows };
	}

	onAgentStart(): void {
		this.runtime.experimentsThisSession = 0;
	}

	onAgentEnd(): void {
		this.runtime.runningExperiment = null;
	}

	shutdown(): void {
		stopDashboardServer();
	}

	compactionSummary(): string {
		return buildAutoresearchCompactionSummary(
			autoresearchSummaryPathsFor(this.getWorkDir()),
		);
	}

	private getWorkDir(): string {
		return resolveWorkDir(this.cwd);
	}

	private setAutoresearchMode(enabled: boolean): void {
		this.runtime.autoresearchMode = enabled;
	}

	private hasAutoresearchRules(): boolean {
		return fs.existsSync(autoresearchMdPath(this.getWorkDir()));
	}

	private readJsonlLines(workDir: string): string[] {
		const jsonlPath = autoresearchJsonlPath(workDir);
		if (!fs.existsSync(jsonlPath)) return [];
		return fs.readFileSync(jsonlPath, "utf-8").split("\n").filter(Boolean);
	}

	private readLastRun(workDir: string): Record<string, unknown> | null {
		const lines = this.readJsonlLines(workDir);
		for (let i = lines.length - 1; i >= 0; i--) {
			const entry = parseJsonlEntry(lines[i]);
			if (isAutoresearchRunEntry(entry)) return entry;
		}
		return null;
	}

	private buildSessionSnapshot(state: ExperimentState): SessionSnapshot {
		return {
			metric_name: state.metricName,
			metric_unit: state.metricUnit,
			direction: state.bestDirection,
			baseline_metric: state.bestMetric,
			best_metric: findBestMetric(
				state.results,
				state.currentSegment,
				state.bestDirection,
			),
			run_count: state.results.length,
			goal: state.name ?? "",
		};
	}

	private async fireHook(payload: HookPayload): Promise<string | null> {
		const result = await runHook(payload);
		appendHookLogEntryIfConfigured(
			autoresearchJsonlPath(payload.cwd),
			payload.event,
			result,
		);
		return steerMessageFor(payload.event, result);
	}

	// -----------------------------------------------------------------------
	// init_experiment tool
	// -----------------------------------------------------------------------

	/** Initialize the experiment session: name, primary metric, unit,
	 * direction. Writes the config header to .auto/log.jsonl. */
	async initExperiment(
		params: Record<string, unknown>,
	): Promise<AutoresearchResult> {
		const workDirError = validateWorkDir(this.cwd);
		if (workDirError) {
			return { content: `❌ ${workDirError}`, details: {} };
		}

		const state = this.runtime.state;
		const isReinit = state.results.length > 0;

		state.name = params.name as string;
		state.metricName = params.metric_name as string;
		state.metricUnit = (params.metric_unit as string) ?? "";
		if (params.direction === "lower" || params.direction === "higher") {
			state.bestDirection = params.direction as "lower" | "higher";
		}

		if (isReinit) {
			state.currentSegment++;
		}
		state.bestMetric = null;
		state.secondaryMetrics = [];
		state.confidence = null;
		state.maxExperiments = readMaxExperiments(this.cwd);

		const workDir = this.getWorkDir();
		try {
			const jsonlPath = autoresearchJsonlPath(workDir);
			ensureParentDir(jsonlPath);
			const config = JSON.stringify({
				type: "config",
				name: state.name,
				metricName: state.metricName,
				metricUnit: state.metricUnit,
				bestDirection: state.bestDirection,
			});
			if (fs.existsSync(jsonlPath)) {
				fs.appendFileSync(jsonlPath, `${config}\n`);
			} else {
				fs.writeFileSync(jsonlPath, `${config}\n`);
			}
			broadcastDashboardUpdate(workDir);
		} catch (e) {
			return {
				content: `⚠️ Failed to write .auto/log.jsonl: ${e instanceof Error ? e.message : String(e)}`,
				details: {},
			};
		}

		this.setAutoresearchMode(true);

		const steer = await this.fireHook({
			event: "before",
			cwd: workDir,
			next_run: state.results.length + 1,
			last_run: this.readLastRun(workDir),
			session: this.buildSessionSnapshot(state),
		});

		const reinitNote = isReinit
			? " (re-initialized — new baseline needed)"
			: "";
		const limitNote =
			state.maxExperiments !== null
				? `\nMax iterations: ${state.maxExperiments}`
				: "";
		const workDirNote =
			workDir !== this.cwd ? `\nWorking directory: ${workDir}` : "";

		return {
			content: `✅ Experiment initialized: "${state.name}"${reinitNote}\nMetric: ${state.metricName} (${state.metricUnit || "unitless"}, ${state.bestDirection} is better)${limitNote}${workDirNote}\nConfig written to .auto/log.jsonl.${steer ? `\n\n${steer}` : ""}`,
			details: { state: cloneExperimentState(state) },
		};
	}

	// -----------------------------------------------------------------------
	// run_experiment tool
	// -----------------------------------------------------------------------

	/** Run a shell command as an experiment: times wall-clock duration,
	 * captures output, detects pass/fail, parses METRIC lines. */
	async runExperiment(
		params: Record<string, unknown>,
	): Promise<AutoresearchResult> {
		const workDirError = validateWorkDir(this.cwd);
		if (workDirError) {
			return { content: `❌ ${workDirError}`, details: {} };
		}
		const workDir = this.getWorkDir();
		const state = this.runtime.state;

		if (state.maxExperiments !== null) {
			const segCount = currentResults(
				state.results,
				state.currentSegment,
			).length;
			if (segCount >= state.maxExperiments) {
				return {
					content: `🛑 Maximum experiments reached (${state.maxExperiments}).`,
					details: {},
				};
			}
		}

		const timeout = ((params.timeout_seconds as number) ?? 600) * 1000;
		const command = params.command as string;

		// Guard: if benchmark script exists, only allow running it
		const autoresearchShPath = autoresearchScriptPath(workDir);
		if (fs.existsSync(autoresearchShPath)) {
			const benchmarkScriptRel =
				path.relative(workDir, autoresearchShPath) ||
				path.basename(autoresearchShPath);
			if (
				!command.includes(benchmarkScriptRel) &&
				!command.includes(".auto/measure.sh")
			) {
				return {
					content: `❌ ${benchmarkScriptRel} exists — you must run it instead of a custom command.`,
					details: {
						command,
						exitCode: null,
						durationSeconds: 0,
						passed: false,
						crashed: true,
					},
				};
			}
		}

		this.runtime.runningExperiment = { startedAt: Date.now(), command };
		const t0 = Date.now();

		return new Promise(resolve => {
			const child = spawn("bash", ["-c", command], {
				cwd: workDir,
				detached: true,
				stdio: ["ignore", "pipe", "pipe"],
			});

			const chunks: Buffer[] = [];

			if (child.stdout)
				child.stdout.on("data", (d: Buffer) => {
					appendOutputTail(chunks, d);
				});
			if (child.stderr)
				child.stderr.on("data", (d: Buffer) => {
					appendOutputTail(chunks, d);
				});

			let timedOut = false;
			let forceKillTimer: NodeJS.Timeout | undefined;
			const timeoutHandle =
				timeout > 0
					? setTimeout(() => {
							timedOut = true;
							if (child.pid) {
								const pid = child.pid;
								killTree(pid);
								forceKillTimer = setTimeout(
									() => killTree(pid, "SIGKILL"),
									1_000,
								);
							}
						}, timeout)
					: undefined;

			child.on("close", async code => {
				if (timeoutHandle) clearTimeout(timeoutHandle);
				if (forceKillTimer) clearTimeout(forceKillTimer);
				const durationSeconds = (Date.now() - t0) / 1000;
				this.runtime.lastRunDuration = durationSeconds;
				this.runtime.runningExperiment = null;

				const output = Buffer.concat(chunks).toString("utf-8");
				const benchmarkPassed = code === 0 && !timedOut;

				// Run checks if benchmark passed
				let checksPass: boolean | null = null;
				let checksOutput = "";
				let checksDuration = 0;

				if (benchmarkPassed && fs.existsSync(autoresearchChecksPath(workDir))) {
					const checksTimeout =
						((params.checks_timeout_seconds as number) ?? 300) * 1000;
					const ct0 = Date.now();
					try {
						const checksResult = await runScript(
							autoresearchChecksPath(workDir),
							workDir,
							checksTimeout,
						);
						checksDuration = (Date.now() - ct0) / 1000;
						checksPass = checksResult.code === 0 && !checksResult.killed;
						checksOutput = (
							checksResult.stdout +
							"\n" +
							checksResult.stderr
						).trim();
					} catch (e) {
						checksDuration = (Date.now() - ct0) / 1000;
						checksPass = false;
						checksOutput = e instanceof Error ? e.message : String(e);
					}
				}

				this.runtime.lastRunChecks =
					checksPass !== null
						? {
								pass: checksPass,
								output: checksOutput,
								duration: checksDuration,
							}
						: null;

				const passed = benchmarkPassed && (checksPass === null || checksPass);
				const llmTruncation = truncateTail(
					output,
					EXPERIMENT_MAX_LINES,
					EXPERIMENT_MAX_BYTES,
				);

				// Parse METRIC lines
				const parsedMetricMap = parseMetricLines(output);
				const parsedMetrics =
					parsedMetricMap.size > 0 ? Object.fromEntries(parsedMetricMap) : null;
				const parsedPrimary = parsedMetricMap.get(state.metricName) ?? null;

				let text = "";
				if (timedOut) {
					text += `⏰ TIMEOUT after ${durationSeconds.toFixed(1)}s\n`;
				} else if (!benchmarkPassed) {
					text += `💥 FAILED (exit code ${code}) in ${durationSeconds.toFixed(1)}s\n`;
				} else if (checksPass === false) {
					text += `✅ Benchmark PASSED in ${durationSeconds.toFixed(1)}s\n💥 CHECKS FAILED in ${checksDuration.toFixed(1)}s\nLog as 'checks_failed'.\n`;
				} else if (checksPass === true) {
					text += `✅ PASSED in ${durationSeconds.toFixed(1)}s\n✅ Checks passed in ${checksDuration.toFixed(1)}s\n`;
				} else {
					text += `✅ PASSED in ${durationSeconds.toFixed(1)}s\n`;
				}

				if (state.bestMetric !== null) {
					text += `📊 Current best ${state.metricName}: ${formatNum(state.bestMetric, state.metricUnit)}\n`;
				}

				if (parsedMetrics) {
					const secondary = Object.entries(parsedMetrics).filter(
						([k]) => k !== state.metricName,
					);
					text += `\n📐 Parsed metrics:`;
					if (parsedPrimary !== null) {
						text += ` ★ ${state.metricName}=${formatNum(parsedPrimary, state.metricUnit)}`;
					}
					for (const [name, value] of secondary) {
						const sm = state.secondaryMetrics.find(m => m.name === name);
						const unit = sm?.unit ?? "";
						text += ` ${name}=${formatNum(value, unit)}`;
					}
					const secArgs = secondary.map(([k, v]) => `"${k}": ${v}`).join(", ");
					text += `\nUse these values: metric: ${parsedPrimary ?? "?"}, metrics: {${secArgs}}\n`;
				}

				text += `\n${llmTruncation.content}`;
				if (llmTruncation.truncated) {
					text += `\n[Showing last ${llmTruncation.outputLines} of ${llmTruncation.totalLines} lines (${formatSize(EXPERIMENT_MAX_BYTES)} limit).]`;
				}

				resolve({
					content: text,
					details: {
						command,
						exitCode: code,
						durationSeconds,
						passed,
						crashed: !passed,
						timedOut,
						tailOutput: llmTruncation.content,
						checksPass,
						checksOutput,
						checksDuration,
						parsedMetrics,
						parsedPrimary,
						metricName: state.metricName,
						metricUnit: state.metricUnit,
					},
				});
			});
		});
	}

	// -----------------------------------------------------------------------
	// log_experiment tool
	// -----------------------------------------------------------------------

	/** Record an experiment result: tracks metrics, auto-commits on 'keep',
	 * auto-reverts on 'discard'/'crash'/'checks_failed'. */
	async logExperiment(
		params: Record<string, unknown>,
	): Promise<AutoresearchResult> {
		const workDirError = validateWorkDir(this.cwd);
		if (workDirError) {
			return { content: `❌ ${workDirError}`, details: {} };
		}
		const workDir = this.getWorkDir();
		const state = this.runtime.state;

		// Gate: prevent "keep" when last run's checks failed
		if (
			(params.status as string) === "keep" &&
			this.runtime.lastRunChecks &&
			!this.runtime.lastRunChecks.pass
		) {
			return {
				content: `❌ Cannot keep — checks failed.\n\n${this.runtime.lastRunChecks.output.slice(-500)}\n\nLog as 'checks_failed' instead.`,
				details: {},
			};
		}

		const secondaryMetrics = (params.metrics as Record<string, number>) ?? {};
		const mergedASI =
			params.asi && Object.keys(params.asi).length > 0
				? (params.asi as ASI)
				: undefined;

		const experiment: ExperimentResult = {
			commit: (params.commit as string).slice(0, 7),
			metric: params.metric as number,
			metrics: secondaryMetrics,
			status: params.status as ExperimentResult["status"],
			description: params.description as string,
			timestamp: Date.now(),
			segment: state.currentSegment,
			confidence: null,
			asi: mergedASI,
		};

		state.results.push(experiment);
		this.runtime.experimentsThisSession++;

		// Register secondary metrics
		for (const name of Object.keys(secondaryMetrics)) {
			if (!state.secondaryMetrics.find(m => m.name === name)) {
				let unit = "";
				if (name.endsWith("µs")) unit = "µs";
				else if (name.endsWith("_ms")) unit = "ms";
				else if (name.endsWith("_s") || name.endsWith("_sec")) unit = "s";
				else if (name.endsWith("_kb")) unit = "kb";
				else if (name.endsWith("_mb")) unit = "mb";
				state.secondaryMetrics.push({ name, unit });
			}
		}

		state.bestMetric = findBaselineMetric(state.results, state.currentSegment);
		state.confidence = computeConfidence(
			state.results,
			state.currentSegment,
			state.bestDirection,
		);
		experiment.confidence = state.confidence;

		const segmentCount = currentResults(
			state.results,
			state.currentSegment,
		).length;
		let text = `Logged #${state.results.length}: ${experiment.status} — ${experiment.description}`;

		if (state.bestMetric !== null) {
			text += `\nBaseline ${state.metricName}: ${formatNum(state.bestMetric, state.metricUnit)}`;
		}

		if (Object.keys(secondaryMetrics).length > 0) {
			const parts = Object.entries(secondaryMetrics).map(([name, value]) => {
				const def = state.secondaryMetrics.find(m => m.name === name);
				return `${name}: ${formatNum(value, def?.unit ?? "")}`;
			});
			text += `\nSecondary: ${parts.join("  ")}`;
		}

		if (mergedASI) {
			const asiParts = Object.entries(mergedASI).map(([k, v]) => {
				const s = typeof v === "string" ? v : JSON.stringify(v);
				return `${k}: ${s.length > 80 ? `${s.slice(0, 77)}…` : s}`;
			});
			text += `\n📋 ASI: ${asiParts.join(" | ")}`;
		}

		if (state.confidence !== null) {
			const confStr = state.confidence.toFixed(1);
			if (state.confidence >= 2.0) {
				text += `\n📊 Confidence: ${confStr}× — improvement is likely real`;
			} else if (state.confidence >= 1.0) {
				text += `\n📊 Confidence: ${confStr}× — improvement is above noise but marginal`;
			} else {
				text += `\n⚠️ Confidence: ${confStr}× — improvement is within noise`;
			}
		}

		text += `\n(${segmentCount} experiments)`;

		// Auto-commit or revert
		if ((params.status as string) === "keep") {
			try {
				const resultData: Record<string, unknown> = {
					status: params.status,
					[state.metricName || "metric"]: params.metric,
					...secondaryMetrics,
				};
				const trailerJson = JSON.stringify(resultData);
				const commitMsg = `${params.description}\n\nResult: ${trailerJson}`;

				await execFile("git", ["add", "-A"], {
					cwd: workDir,
					timeout: 10000,
				});
				await execFile("git", ["commit", "-m", commitMsg], {
					cwd: workDir,
					timeout: 10000,
				});
				text += `\n📝 Git: committed`;
			} catch (e) {
				text += `\n⚠️ Git commit error: ${e instanceof Error ? e.message : String(e)}`;
			}
		} else {
			try {
				const revertScript = `
					git checkout -- . ':(exclude,glob)**/${AUTO_DIR}' ':(exclude,glob)**/${AUTO_DIR}/**'
					git clean -fd -e '${AUTO_DIR}' -e '**/${AUTO_DIR}/**' 2>/dev/null || true
				`;
				await execFile("bash", ["-c", revertScript], {
					cwd: workDir,
					timeout: 10000,
				});
				text += `\n📝 Git: reverted changes (${params.status}) — autoresearch files preserved`;
			} catch (e) {
				text += `\n⚠️ Git revert failed: ${e instanceof Error ? e.message : String(e)}`;
			}
		}

		// Write to JSONL
		const jsonlEntry: Record<string, unknown> = {
			run: state.results.length,
			...experiment,
		};
		if (!mergedASI) jsonlEntry.asi = undefined;
		try {
			const jsonlPath = autoresearchJsonlPath(workDir);
			ensureParentDir(jsonlPath);
			fs.appendFileSync(jsonlPath, `${JSON.stringify(jsonlEntry)}\n`);
			broadcastDashboardUpdate(workDir);
		} catch (_e) {
			text += `\n⚠️ Failed to write .auto/log.jsonl`;
		}

		// Fire hooks
		await this.fireHook({
			event: "after",
			cwd: workDir,
			run_entry: jsonlEntry,
			session: this.buildSessionSnapshot(state),
		});

		const limitReached =
			state.maxExperiments !== null && segmentCount >= state.maxExperiments;
		if (limitReached) {
			text += `\n\n🛑 Maximum experiments reached (${state.maxExperiments}). STOP.`;
			this.setAutoresearchMode(false);
		}

		return {
			content: text,
			details: {
				experiment: { ...experiment, metrics: { ...experiment.metrics } },
				state: cloneExperimentState(state),
				wallClockSeconds: this.runtime.lastRunDuration,
			} as ExperimentStateForLog,
		};
	}

	// -----------------------------------------------------------------------
	// /autoresearch command
	// -----------------------------------------------------------------------

	private autoresearchHelp(): string {
		return [
			"Usage: /autoresearch [off|clear|export|<text>]",
			"",
			"<text> enters autoresearch mode and starts the loop.",
			"off leaves autoresearch mode.",
			"clear deletes the session log and turns autoresearch mode off.",
			"export opens a local live dashboard in your browser.",
			"",
			"Examples:",
			"  /autoresearch optimize unit test runtime, monitor correctness",
			"  /autoresearch export",
		].join("\n");
	}

	/**
	 * Handle the `/autoresearch [off|clear|export|<text>]` command.
	 *
	 * `injectAsTurn` is true only for the activation path (a bare
	 * `/autoresearch <goal>` that isn't off/clear/export/already-active) —
	 * that message is a kickoff prompt meant to become the agent's next
	 * turn, not just a status line shown to the user. Every other reply is
	 * informational (help text, confirmations, errors) and should just be
	 * displayed.
	 */
	async handleCommand(
		args: string | undefined,
	): Promise<{ message: string; injectAsTurn: boolean }> {
		const trimmedArgs = (args ?? "").trim();
		const command = trimmedArgs.toLowerCase();

		if (!trimmedArgs) {
			return { message: this.autoresearchHelp(), injectAsTurn: false };
		}

		if (command === "off") {
			this.setAutoresearchMode(false);
			this.runtime.autoResumeTurns = 0;
			this.runtime.experimentsThisSession = 0;
			stopDashboardServer();
			return { message: "Autoresearch mode OFF", injectAsTurn: false };
		}

		if (command === "export") {
			await exportDashboard(this.notify, this.getWorkDir());
			return { message: "", injectAsTurn: false };
		}

		if (command === "clear") {
			const workDir = this.getWorkDir();
			this.setAutoresearchMode(false);
			this.runtime.autoResumeTurns = 0;
			this.runtime.experimentsThisSession = 0;
			this.runtime.state = createExperimentState();
			stopDashboardServer();

			const jsonlPaths = sessionFileCandidates(workDir, "log");
			for (const jsonlPath of [jsonlPaths.current, jsonlPaths.legacy]) {
				if (fs.existsSync(jsonlPath)) {
					try {
						fs.unlinkSync(jsonlPath);
					} catch {
						/* ignore */
					}
				}
			}

			return {
				message: "Autoresearch cleared and mode OFF",
				injectAsTurn: false,
			};
		}

		if (this.runtime.autoresearchMode) {
			return {
				message:
					"Autoresearch already active — use '/autoresearch off' to stop first",
				injectAsTurn: false,
			};
		}

		const workDir = this.getWorkDir();
		this.setAutoresearchMode(true);

		const rulesLoaded = this.hasAutoresearchRules();
		const kickoff = rulesLoaded
			? `Autoresearch mode active. ${trimmedArgs} ${BENCHMARK_GUARDRAIL}`
			: `Autoresearch mode ON — no rules found, use /autoresearch to configure`;

		const activationSteer = await this.fireHook({
			event: "before",
			cwd: workDir,
			next_run: this.runtime.state.results.length + 1,
			last_run: this.readLastRun(workDir),
			session: this.buildSessionSnapshot(this.runtime.state),
		});

		const message =
			activationSteer && rulesLoaded
				? `${activationSteer}\n\n${kickoff}`
				: kickoff;
		this.notify(
			rulesLoaded
				? "Autoresearch mode ON — rules loaded from .auto/prompt.md"
				: "Autoresearch mode ON — no .auto/prompt.md found",
			"info",
		);

		return { message, injectAsTurn: true };
	}
}
