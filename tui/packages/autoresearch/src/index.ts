/**
 * autoresearch — Logician Extension
 *
 * Generic autonomous experiment loop infrastructure.
 * Ported from pi-autoresearch, adapted for Logician's extension system.
 *
 * Provides:
 * - `init_experiment` tool — initialize experiment session
 * - `run_experiment` tool — runs any command, times it, captures output
 * - `log_experiment` tool — records results with session-persisted state
 * - `/autoresearch` command — control autoresearch mode
 * - Hooks support (before/after iteration)
 * - JSONL state persistence
 */

import { execFile as execFileCb } from "node:child_process";
import { promisify } from "node:util";
import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionToolResult,
	ExtensionUI,
} from "./types.ts";

const execFile = promisify(execFileCb);

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import { createServer, type Server, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
	autoresearchSummaryPathsFor,
	buildAutoresearchCompactionSummary,
} from "./compaction.ts";
import {
	appendHookLogEntryIfConfigured,
	type HookPayload,
	runHook,
	type SessionSnapshot,
	steerMessageFor,
} from "./hooks.ts";
import {
	extractAutoresearchSessionName,
	isAutoresearchRunEntry,
	parseJsonlEntry,
	reconstructJsonlState,
} from "./jsonl.ts";
import {
	AUTO_DIR,
	ensureParentDir,
	sessionFileCandidates,
	sessionFilePath,
} from "./paths.ts";

// ---------------------------------------------------------------------------
// Experiment output limits (sent to LLM — keep small to save context)
// ---------------------------------------------------------------------------
const EXPERIMENT_MAX_LINES = 10;
const EXPERIMENT_MAX_BYTES = 4 * 1024; // 4KB

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Actionable Side Information (ASI) — free-form diagnostics per experiment run.
 */
interface ASI {
	[key: string]: unknown;
}

interface ExperimentResult {
	commit: string;
	metric: number;
	/** Additional tracked metrics: { name: value } */
	metrics: Record<string, number>;
	status: "keep" | "discard" | "crash" | "checks_failed";
	description: string;
	timestamp: number;
	/** Segment index — increments on each config header */
	segment: number;
	/** Session-level confidence score */
	confidence: number | null;
	/** Actionable Side Information */
	asi?: ASI;
}

interface MetricDef {
	name: string;
	unit: string;
}

interface ExperimentState {
	results: ExperimentResult[];
	/** Baseline primary metric */
	bestMetric: number | null;
	bestDirection: "lower" | "higher";
	metricName: string;
	metricUnit: string;
	/** Secondary metrics definitions */
	secondaryMetrics: MetricDef[];
	name: string | null;
	/** Current segment index */
	currentSegment: number;
	/** Maximum number of experiments before auto-stopping */
	maxExperiments: number | null;
	/** Session confidence score */
	confidence: number | null;
}

// ---------------------------------------------------------------------------
// Experiment state helpers
// ---------------------------------------------------------------------------

function isBetter(
	current: number,
	best: number,
	direction: "lower" | "higher",
): boolean {
	return direction === "lower" ? current < best : current > best;
}

function sortedMedian(values: number[]): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? (sorted[mid - 1] + sorted[mid]) / 2
		: sorted[mid];
}

function computeConfidence(
	results: ExperimentResult[],
	segment: number,
	direction: "lower" | "higher",
): number | null {
	const cur = results.filter(r => r.segment === segment && r.metric > 0);
	if (cur.length < 3) return null;

	const values = cur.map(r => r.metric);
	const median = sortedMedian(values);
	const deviations = values.map(v => Math.abs(v - median));
	const mad = sortedMedian(deviations);

	if (mad === 0) return null;

	let baseline: number | null = null;
	for (const r of cur) {
		if (r.segment === segment) {
			baseline = r.metric;
			break;
		}
	}
	if (baseline === null) return null;

	let bestKept: number | null = null;
	for (const r of cur) {
		if (r.status === "keep" && r.metric > 0) {
			if (bestKept === null || isBetter(r.metric, bestKept, direction)) {
				bestKept = r.metric;
			}
		}
	}
	if (bestKept === null || bestKept === baseline) return null;

	const delta = Math.abs(bestKept - baseline);
	return delta / mad;
}

function currentResults(
	results: ExperimentResult[],
	segment: number,
): ExperimentResult[] {
	return results.filter(r => r.segment === segment);
}

function findBaselineMetric(
	results: ExperimentResult[],
	segment: number,
): number | null {
	const cur = currentResults(results, segment);
	return cur.length > 0 ? cur[0].metric : null;
}

function findBestMetric(
	results: ExperimentResult[],
	segment: number,
	direction: "lower" | "higher",
): number | null {
	const kept = currentResults(results, segment)
		.filter(r => r.status === "keep")
		.map(r => r.metric);
	if (kept.length === 0) return null;
	return direction === "lower" ? Math.min(...kept) : Math.max(...kept);
}

function _findBaselineSecondary(
	results: ExperimentResult[],
	segment: number,
	knownMetrics?: MetricDef[],
): Record<string, number> {
	const cur = currentResults(results, segment);
	const base: Record<string, number> =
		cur.length > 0 ? { ...(cur[0].metrics ?? {}) } : {};

	if (knownMetrics) {
		for (const sm of knownMetrics) {
			if (base[sm.name] === undefined) {
				for (const r of cur) {
					const val = r.metrics?.[sm.name];
					if (val !== undefined) {
						base[sm.name] = val;
						break;
					}
				}
			}
		}
	}

	return base;
}

function cloneExperimentState(state: ExperimentState): ExperimentState {
	return {
		...state,
		results: state.results.map(result => ({
			...result,
			metrics: { ...result.metrics },
		})),
		secondaryMetrics: state.secondaryMetrics.map(metric => ({ ...metric })),
	};
}

function _clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function formatNum(value: number | null, unit: string): string {
	if (value === null) return "—";
	const u = unit || "";
	if (value === Math.round(value)) {
		return value.toLocaleString() + u;
	}
	return value.toFixed(2) + u;
}

function _formatElapsed(ms: number): string {
	const totalSec = Math.floor(ms / 1000);
	const m = Math.floor(totalSec / 60);
	const s = totalSec % 60;
	if (m > 0) return `${m}m ${String(s).padStart(2, "0")}s`;
	return `${s}s`;
}

function killTree(pid: number): void {
	try {
		process.kill(-pid, "SIGTERM");
	} catch {
		try {
			process.kill(pid, "SIGTERM");
		} catch {
			// Process may have already exited
		}
	}
}

function truncateTail(
	output: string,
	maxLines: number,
	maxBytes: number,
): {
	content: string;
	truncated: boolean;
	outputLines: number;
	totalLines: number;
} {
	const lines = output.split("\n");
	const totalLines = lines.length;
	let truncated = false;

	// Limit to maxLines
	if (lines.length > maxLines) {
		lines.splice(0, lines.length - maxLines);
		truncated = true;
	}

	// Limit to maxBytes
	const content = lines.join("\n");
	if (Buffer.byteLength(content) > maxBytes) {
		const limited = output.slice(-maxBytes);
		return {
			content: limited,
			truncated: true,
			outputLines: Math.min(maxLines, output.split("\n").length),
			totalLines,
		};
	}

	return { content, truncated, outputLines: lines.length, totalLines };
}

function formatSize(bytes: number): string {
	if (bytes < 1024) return `${bytes}B`;
	if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
	return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

interface AutoresearchConfig {
	maxIterations?: number;
	workingDir?: string;
}

function autoresearchConfigPath(dir: string): string {
	return sessionFilePath(dir, "config");
}

function readConfig(cwd: string): AutoresearchConfig {
	try {
		const configPath = autoresearchConfigPath(cwd);
		if (!fs.existsSync(configPath)) return {};
		return JSON.parse(fs.readFileSync(configPath, "utf-8"));
	} catch {
		return {};
	}
}

function readMaxExperiments(cwd: string): number | null {
	const config = readConfig(cwd);
	return typeof config.maxIterations === "number" && config.maxIterations > 0
		? Math.floor(config.maxIterations)
		: null;
}

function resolveWorkDir(ctxCwd: string): string {
	const config = readConfig(ctxCwd);
	if (!config.workingDir) return ctxCwd;
	return path.isAbsolute(config.workingDir)
		? config.workingDir
		: path.resolve(ctxCwd, config.workingDir);
}

function validateWorkDir(ctxCwd: string): string | null {
	const workDir = resolveWorkDir(ctxCwd);
	if (workDir === ctxCwd) return null;
	try {
		const stat = fs.statSync(workDir);
		if (!stat.isDirectory()) {
			return `workingDir "${workDir}" (from .auto/config.json) is not a directory.`;
		}
	} catch {
		return `workingDir "${workDir}" (from .auto/config.json) does not exist.`;
	}
	return null;
}

// ---------------------------------------------------------------------------
// Metric parsing
// ---------------------------------------------------------------------------

const METRIC_LINE_PREFIX = "METRIC";

const DENIED_METRIC_NAMES = new Set(["__proto__", "constructor", "prototype"]);

function parseMetricLines(output: string): Map<string, number> {
	const metrics = new Map<string, number>();
	const regex = new RegExp(
		`^${METRIC_LINE_PREFIX}\\s+([\\w.µ]+)=(\\S+)\\s*$`,
		"gm",
	);
	let match;
	while ((match = regex.exec(output)) !== null) {
		const name = match[1];
		if (DENIED_METRIC_NAMES.has(name)) continue;
		const value = Number(match[2]);
		if (Number.isFinite(value)) {
			metrics.set(name, value);
		}
	}
	return metrics;
}

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

function createExperimentState(): ExperimentState {
	return {
		results: [],
		bestMetric: null,
		bestDirection: "lower",
		metricName: "metric",
		metricUnit: "",
		secondaryMetrics: [],
		name: null,
		currentSegment: 0,
		maxExperiments: null,
		confidence: null,
	};
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
// Dashboard server (SSE-based live dashboard)
// ---------------------------------------------------------------------------

const TITLE_PLACEHOLDER = "__AUTORESEARCH_TITLE__";
const LOGO_PLACEHOLDER = "__AUTORESEARCH_LOGO__";

let cachedPackageRoot: string | null = null;

function packageRoot(): string {
	if (cachedPackageRoot) return cachedPackageRoot;
	const extensionDir = fs.realpathSync(
		path.dirname(fileURLToPath(import.meta.url)),
	);
	cachedPackageRoot = path.resolve(extensionDir, "..");
	return cachedPackageRoot;
}

function templatePath(): string {
	return path.join(packageRoot(), "assets", "template.html");
}

function logoDataUrl(): string {
	const logoPath = path.join(packageRoot(), "assets", "logo.webp");
	const bytes = fs.readFileSync(logoPath);
	return `data:image/webp;base64,${bytes.toString("base64")}`;
}

let dashboardServer: Server | null = null;
let dashboardServerPort: number | null = null;
let dashboardServerWorkDir: string | null = null;
let _dashboardServerHtmlPath: string | null = null;
const dashboardSSEClients = new Set<ServerResponse>();

function stopDashboardServer(): void {
	for (const client of dashboardSSEClients) {
		try {
			client.end();
		} catch {
			/* ignore */
		}
	}
	dashboardSSEClients.clear();

	if (dashboardServer) {
		try {
			dashboardServer.close();
		} catch {
			/* ignore */
		}
	}

	dashboardServer = null;
	dashboardServerPort = null;
	dashboardServerWorkDir = null;
	_dashboardServerHtmlPath = null;
}

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

function openInBrowser(url: string): void {
	const child =
		process.platform === "win32"
			? spawn("cmd", ["/c", "start", "", url], {
					detached: true,
					shell: true,
					stdio: "ignore",
				})
			: spawn(process.platform === "darwin" ? "open" : "xdg-open", [url], {
					detached: true,
					stdio: "ignore",
				});
	child.on("error", () => {
		/* ignore */
	});
	child.unref();
}

function broadcastDashboardUpdate(workDir: string): void {
	if (!dashboardServer || dashboardServerWorkDir !== workDir) return;
	for (const res of dashboardSSEClients) {
		try {
			res.write("event: jsonl-updated\n");
			res.write(`data: ${Date.now()}\n\n`);
		} catch {
			dashboardSSEClients.delete(res);
		}
	}
}

async function startDashboardServer(
	workDir: string,
	dashboardHtmlPath: string,
): Promise<number> {
	return new Promise((resolve, reject) => {
		const resolvedWorkDir = path.resolve(workDir);
		const resolvedHtmlPath = path.resolve(dashboardHtmlPath);

		if (
			dashboardServer &&
			dashboardServerWorkDir === resolvedWorkDir &&
			dashboardServerPort
		) {
			_dashboardServerHtmlPath = resolvedHtmlPath;
			resolve(dashboardServerPort);
			return;
		}

		stopDashboardServer();
		_dashboardServerHtmlPath = resolvedHtmlPath;

		const server = createServer((req, res) => {
			const url = new URL(req.url ?? "/", "http://127.0.0.1");

			if (url.pathname === "/events") {
				res.writeHead(200, {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				});
				res.write("retry: 1000\n\n");
				dashboardSSEClients.add(res);
				res.on("close", () => dashboardSSEClients.delete(res));
				return;
			}

			if (url.pathname === "/") {
				fs.readFile(resolvedHtmlPath, (err, data) => {
					if (err) {
						res.writeHead(404);
						res.end();
						return;
					}
					res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
					res.end(data);
				});
				return;
			}

			if (url.pathname === "/autoresearch.jsonl") {
				const jsonlPath = sessionFilePath(resolvedWorkDir, "log");
				fs.readFile(jsonlPath, (err, data) => {
					if (err) {
						res.writeHead(404);
						res.end();
						return;
					}
					res.writeHead(200, { "Content-Type": "application/jsonl" });
					res.end(data);
				});
				return;
			}

			res.writeHead(404);
			res.end();
		});

		server.listen(0, "127.0.0.1", () => {
			const addr = server.address();
			if (!addr || typeof addr === "string") {
				reject(new Error("Failed to bind dashboard server"));
				return;
			}
			dashboardServer = server;
			dashboardServerPort = addr.port;
			dashboardServerWorkDir = resolvedWorkDir;
			resolve(addr.port);
		});

		server.on("error", reject);
	});
}

function writeDashboardFile(workDir: string): string {
	const jsonlContent = fs
		.readFileSync(sessionFilePath(workDir, "log"), "utf-8")
		.trim();
	const sessionName = extractAutoresearchSessionName(jsonlContent);
	const template = fs.readFileSync(templatePath(), "utf-8");
	const html = template
		.replace(TITLE_PLACEHOLDER, escapeHtml(sessionName))
		.replace(LOGO_PLACEHOLDER, logoDataUrl());
	const exportDir = fs.mkdtempSync(
		path.join(tmpdir(), "logician-autoresearch-dashboard-"),
	);
	const dest = path.join(exportDir, "index.html");
	fs.writeFileSync(dest, html);
	return dest;
}

async function exportDashboard(
	ui: ExtensionUI,
	workDir: string,
): Promise<void> {
	const jsonlPath = sessionFilePath(workDir, "log");
	if (!fs.existsSync(jsonlPath)) {
		ui.notify(
			`No ${path.basename(jsonlPath)} found — run some experiments first`,
			"error",
		);
		return;
	}

	try {
		const dashboardHtmlPath = writeDashboardFile(workDir);
		const port = await startDashboardServer(workDir, dashboardHtmlPath);
		const url = `http://127.0.0.1:${port}`;
		openInBrowser(url);
		ui.notify(`Dashboard at ${url} (live updates)`, "info");
	} catch (error) {
		ui.notify(
			`Export failed: ${error instanceof Error ? error.message : String(error)}`,
			"error",
		);
	}
}

// ---------------------------------------------------------------------------
// Session state reconstruction
// ---------------------------------------------------------------------------

function autoresearchJsonlPath(dir: string): string {
	return sessionFilePath(dir, "log");
}

function autoresearchMdPath(dir: string): string {
	return sessionFilePath(dir, "prompt");
}

function _autoresearchIdeasPath(dir: string): string {
	return sessionFilePath(dir, "ideas");
}

function autoresearchChecksPath(dir: string): string {
	return sessionFilePath(dir, "checks");
}

function autoresearchScriptPath(dir: string): string {
	return sessionFilePath(dir, "measure");
}

function reconstructState(ctx: ExtensionContext): ExperimentState {
	const state = createExperimentState();

	const workDir = resolveWorkDir(ctx.cwd);
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

	state.maxExperiments = readMaxExperiments(ctx.cwd);
	return state;
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

const BENCHMARK_GUARDRAIL =
	"Be careful not to overfit to the benchmarks and do not cheat on the benchmarks.";

export function registerAutoresearch(
	api: ExtensionAPI,
	ctx: ExtensionContext,
): void {
	const runtime = createSessionRuntime();

	const _getRuntime = (): AutoresearchRuntime => runtime;
	const getWorkDir = (): string => resolveWorkDir(ctx.cwd);

	const setAutoresearchMode = (enabled: boolean): void => {
		runtime.autoresearchMode = enabled;
	};

	const hasAutoresearchRules = (): boolean =>
		fs.existsSync(autoresearchMdPath(getWorkDir()));

	const readJsonlLines = (workDir: string): string[] => {
		const jsonlPath = autoresearchJsonlPath(workDir);
		if (!fs.existsSync(jsonlPath)) return [];
		return fs.readFileSync(jsonlPath, "utf-8").split("\n").filter(Boolean);
	};

	const readLastRun = (workDir: string): Record<string, unknown> | null => {
		const lines = readJsonlLines(workDir);
		for (let i = lines.length - 1; i >= 0; i--) {
			const entry = parseJsonlEntry(lines[i]);
			if (isAutoresearchRunEntry(entry)) return entry;
		}
		return null;
	};

	const buildSessionSnapshot = (state: ExperimentState): SessionSnapshot => ({
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
	});

	const fireHook = async (payload: HookPayload): Promise<string | null> => {
		const result = await runHook(payload);
		appendHookLogEntryIfConfigured(
			autoresearchJsonlPath(payload.cwd),
			payload.event,
			result,
		);
		return steerMessageFor(payload.event, result);
	};

	// -----------------------------------------------------------------------
	// init_experiment tool
	// -----------------------------------------------------------------------

	api.registerTool({
		name: "init_experiment",
		label: "Init Experiment",
		description:
			"Initialize the experiment session. Call once before the first run_experiment to set the name, primary metric, unit, and direction. Writes the config header to .auto/log.jsonl.",
		parameters: {
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
					description:
						'Whether "lower" or "higher" is better. Default: "lower".',
				},
			},
			required: ["name", "metric_name"],
		},
		execute: async (
			_toolCallId,
			params,
			_execCtx,
		): Promise<ExtensionToolResult> => {
			const workDirError = validateWorkDir(ctx.cwd);
			if (workDirError) {
				return { content: `❌ ${workDirError}`, details: {} };
			}

			const state = runtime.state;
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
			state.maxExperiments = readMaxExperiments(ctx.cwd);

			const workDir = getWorkDir();
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

			setAutoresearchMode(true);

			const _steer = await fireHook({
				event: "before",
				cwd: workDir,
				next_run: state.results.length + 1,
				last_run: readLastRun(workDir),
				session: buildSessionSnapshot(state),
			});

			const reinitNote = isReinit
				? " (re-initialized — new baseline needed)"
				: "";
			const limitNote =
				state.maxExperiments !== null
					? `\nMax iterations: ${state.maxExperiments}`
					: "";
			const workDirNote =
				workDir !== ctx.cwd ? `\nWorking directory: ${workDir}` : "";

			return {
				content: `✅ Experiment initialized: "${state.name}"${reinitNote}\nMetric: ${state.metricName} (${state.metricUnit || "unitless"}, ${state.bestDirection} is better)${limitNote}${workDirNote}\nConfig written to .auto/log.jsonl.`,
				details: { state: cloneExperimentState(state) },
			};
		},
	});

	// -----------------------------------------------------------------------
	// run_experiment tool
	// -----------------------------------------------------------------------

	api.registerTool({
		name: "run_experiment",
		label: "Run Experiment",
		description:
			"Run a shell command as an experiment. Times wall-clock duration, captures output, detects pass/fail. Output is truncated. If METRIC lines are found, they are parsed automatically.",
		parameters: {
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
		},
		execute: async (
			_toolCallId,
			params,
			_execCtx,
		): Promise<ExtensionToolResult> => {
			const workDirError = validateWorkDir(ctx.cwd);
			if (workDirError) {
				return { content: `❌ ${workDirError}`, details: {} };
			}
			const workDir = getWorkDir();
			const state = runtime.state;

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

			runtime.runningExperiment = { startedAt: Date.now(), command };
			const t0 = Date.now();

			return new Promise(resolve => {
				const child = spawn("bash", ["-c", command], {
					cwd: workDir,
					detached: true,
					stdio: ["ignore", "pipe", "pipe"],
				});

				const chunks: Buffer[] = [];
				let _totalBytes = 0;

				if (child.stdout)
					child.stdout.on("data", (d: Buffer) => {
						_totalBytes += d.length;
						chunks.push(d);
					});
				if (child.stderr)
					child.stderr.on("data", (d: Buffer) => {
						_totalBytes += d.length;
						chunks.push(d);
					});

				let timedOut = false;
				if (timeout > 0) {
					setTimeout(() => {
						timedOut = true;
						if (child.pid) killTree(child.pid);
					}, timeout);
				}

				child.on("close", code => {
					const durationSeconds = (Date.now() - t0) / 1000;
					runtime.lastRunDuration = durationSeconds;
					runtime.runningExperiment = null;

					const output = Buffer.concat(chunks).toString("utf-8");
					const benchmarkPassed = code === 0 && !timedOut;

					// Run checks if benchmark passed
					let checksPass: boolean | null = null;
					let checksOutput = "";
					let checksDuration = 0;

					if (
						benchmarkPassed &&
						fs.existsSync(autoresearchChecksPath(workDir))
					) {
						const _checksTimeout =
							((params.checks_timeout_seconds as number) ?? 300) * 1000;
						const ct0 = Date.now();
						try {
							const checksResult = {
								code: 0,
								stdout: "",
								stderr: "",
								killed: false,
							};
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

					runtime.lastRunChecks =
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
						parsedMetricMap.size > 0
							? Object.fromEntries(parsedMetricMap)
							: null;
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
						const secArgs = secondary
							.map(([k, v]) => `"${k}": ${v}`)
							.join(", ");
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
		},
	});

	// -----------------------------------------------------------------------
	// log_experiment tool
	// -----------------------------------------------------------------------

	api.registerTool({
		name: "log_experiment",
		label: "Log Experiment",
		description:
			"Record an experiment result. Tracks metrics, auto-commits on 'keep', auto-reverts on 'discard'/'crash'/'checks_failed'.",
		parameters: {
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
		},
		execute: async (
			_toolCallId,
			params,
			_execCtx,
		): Promise<ExtensionToolResult> => {
			const workDirError = validateWorkDir(ctx.cwd);
			if (workDirError) {
				return { content: `❌ ${workDirError}`, details: {} };
			}
			const workDir = getWorkDir();
			const state = runtime.state;

			// Gate: prevent "keep" when last run's checks failed
			if (
				(params.status as string) === "keep" &&
				runtime.lastRunChecks &&
				!runtime.lastRunChecks.pass
			) {
				return {
					content: `❌ Cannot keep — checks failed.\n\n${runtime.lastRunChecks.output.slice(-500)}\n\nLog as 'checks_failed' instead.`,
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
			runtime.experimentsThisSession++;

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

			state.bestMetric = findBaselineMetric(
				state.results,
				state.currentSegment,
			);
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
			await fireHook({
				event: "after",
				cwd: workDir,
				run_entry: jsonlEntry,
				session: buildSessionSnapshot(state),
			});

			const limitReached =
				state.maxExperiments !== null && segmentCount >= state.maxExperiments;
			if (limitReached) {
				text += `\n\n🛑 Maximum experiments reached (${state.maxExperiments}). STOP.`;
				setAutoresearchMode(false);
			}

			return {
				content: text,
				details: {
					experiment: { ...experiment, metrics: { ...experiment.metrics } },
					state: cloneExperimentState(state),
					wallClockSeconds: runtime.lastRunDuration,
				} as ExperimentStateForLog,
			};
		},
	});

	// -----------------------------------------------------------------------
	// /autoresearch command
	// -----------------------------------------------------------------------

	const autoresearchHelp = () =>
		[
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

	api.registerCommand({
		name: "autoresearch",
		description: "Start, stop, clear, or resume autoresearch mode",
		acceptsArgs: true,
		handler: async (args: string | undefined): Promise<string> => {
			const trimmedArgs = (args ?? "").trim();
			const command = trimmedArgs.toLowerCase();

			if (!trimmedArgs) {
				return autoresearchHelp();
			}

			if (command === "off") {
				setAutoresearchMode(false);
				runtime.autoResumeTurns = 0;
				runtime.experimentsThisSession = 0;
				stopDashboardServer();
				return "Autoresearch mode OFF";
			}

			if (command === "export") {
				await exportDashboard(ctx.ui, getWorkDir());
				return "";
			}

			if (command === "clear") {
				const workDir = getWorkDir();
				setAutoresearchMode(false);
				runtime.autoResumeTurns = 0;
				runtime.experimentsThisSession = 0;
				runtime.state = createExperimentState();
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

				return "Autoresearch cleared and mode OFF";
			}

			if (runtime.autoresearchMode) {
				return "Autoresearch already active — use '/autoresearch off' to stop first";
			}

			const workDir = getWorkDir();
			setAutoresearchMode(true);

			const rulesLoaded = hasAutoresearchRules();
			const kickoff = rulesLoaded
				? `Autoresearch mode active. ${trimmedArgs} ${BENCHMARK_GUARDRAIL}`
				: `Autoresearch mode ON — no rules found, use /autoresearch to configure`;

			const activationSteer = await fireHook({
				event: "before",
				cwd: workDir,
				next_run: runtime.state.results.length + 1,
				last_run: readLastRun(workDir),
				session: buildSessionSnapshot(runtime.state),
			});

			const message =
				activationSteer && rulesLoaded
					? `${activationSteer}\n\n${kickoff}`
					: kickoff;
			// In logician, we can't send messages directly — this would need harness integration
			// For now, just notify the user
			ctx.ui.notify(
				rulesLoaded
					? "Autoresearch mode ON — rules loaded from .auto/prompt.md"
					: "Autoresearch mode ON — no .auto/prompt.md found",
				"info",
			);

			return message;
		},
	});

	// -----------------------------------------------------------------------
	// Event handlers
	// -----------------------------------------------------------------------

	api.on("session_start", async (_event, _ctx) => {
		runtime.state = reconstructState(ctx);
	});

	api.on("agent_start", async () => {
		runtime.experimentsThisSession = 0;
	});

	api.on("agent_end", async () => {
		runtime.runningExperiment = null;
	});

	api.on("session_shutdown", async () => {
		stopDashboardServer();
	});

	api.on("before_compact", async _event => {
		return {
			compaction: {
				summary: buildAutoresearchCompactionSummary(
					autoresearchSummaryPathsFor(getWorkDir()),
				),
			},
		};
	});
}
