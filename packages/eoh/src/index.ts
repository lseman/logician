/**
 * eoh — Evolution of Heuristics (EoH) session logic
 *
 * Generic autonomous heuristic evolution infrastructure: seed a heuristic,
 * evolve it via LLM-driven operators, evaluate fitness, keep improvements.
 * Ported from agent-blocks/src/eoh.
 *
 * This module owns the pure session state/logic — the `EohSession` class
 * exposes plain methods (initEvolution/runGeneration/getStatus/best) with no
 * dependency on any particular tool-registration or extension-API shape.
 * Callers wire these into real tools and slash commands.
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import { createServer, type Server, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { buildEohCompactionSummary, eohSummaryPathsFor } from "./compaction.ts";
import { EohEngine } from "./engine.ts";
import {
	appendHookLogEntryIfConfigured,
	type HookPayload,
	runHook,
	type SessionSnapshot,
	steerMessageFor,
} from "./hooks.ts";
import { extractEohSessionName, reconstructEohState } from "./jsonl.ts";
import { ensureParentDir, sessionFilePath } from "./paths.ts";
import { populationStats } from "./population.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Actionable Side Information (ASI) — free-form diagnostics per generation run.
 */
interface ASI {
	[key: string]: unknown;
}

interface EohRunResult {
	thought: string;
	code: string;
	fitness: number;
	generation: number;
	createdBy: string;
	parentIds: string[];
	status: "keep" | "discard" | "crash";
	description: string;
	timestamp: number;
	segment: number;
	asi?: ASI;
}

interface EohState {
	results: EohRunResult[];
	bestDirection: "lower" | "higher";
	populationSize: number;
	maxGenerations: number;
	currentSegment: number;
	name: string | null;
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

interface EohRuntime {
	eohMode: boolean;
	generationsThisSession: number;
	autoResumeTurns: number;
	runningGeneration: { startedAt: number; description: string } | null;
	state: EohState;
}

function createEohState(): EohState {
	return {
		results: [],
		bestDirection: "lower",
		populationSize: 10,
		maxGenerations: 0,
		currentSegment: 0,
		name: null,
	};
}

function createSessionRuntime(): EohRuntime {
	return {
		eohMode: false,
		generationsThisSession: 0,
		autoResumeTurns: 0,
		runningGeneration: null,
		state: createEohState(),
	};
}
// ---------------------------------------------------------------------------
// Dashboard server (SSE-based live dashboard)
// ---------------------------------------------------------------------------

const TITLE_PLACEHOLDER = "__EOH_TITLE__";
const LOGO_PLACEHOLDER = "__EOH_LOGO__";

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
	if (!fs.existsSync(logoPath)) return "";
	const bytes = fs.readFileSync(logoPath);
	return `data:image/webp;base64,${bytes.toString("base64")}`;
}

let dashboardServer: Server | null = null;
let dashboardServerPort: number | null = null;
let dashboardServerWorkDir: string | null = null;
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
			resolve(dashboardServerPort);
			return;
		}

		stopDashboardServer();

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

			if (url.pathname === "/eoh.jsonl") {
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
	const sessionName = extractEohSessionName(jsonlContent);
	const template = fs.readFileSync(templatePath(), "utf-8");
	const html = template
		.replace(TITLE_PLACEHOLDER, escapeHtml(sessionName))
		.replace(LOGO_PLACEHOLDER, logoDataUrl());
	const exportDir = fs.mkdtempSync(
		path.join(tmpdir(), "logician-eoh-dashboard-"),
	);
	const dest = path.join(exportDir, "index.html");
	fs.writeFileSync(dest, html);
	return dest;
}

async function exportDashboard(
	notify: NotifyFn,
	workDir: string,
): Promise<void> {
	const jsonlPath = sessionFilePath(workDir, "log");
	if (!fs.existsSync(jsonlPath)) {
		notify(
			`No ${path.basename(jsonlPath)} found — run some generations first`,
			"error",
		);
		return;
	}

	try {
		const dashboardHtmlPath = writeDashboardFile(workDir);
		const port = await startDashboardServer(workDir, dashboardHtmlPath);
		const url = `http://127.0.0.1:${port}`;
		openInBrowser(url);
		notify(`Dashboard at ${url} (live updates)`, "info");
	} catch (error) {
		notify(
			`Export failed: ${error instanceof Error ? error.message : String(error)}`,
			"error",
		);
	}
}
// ---------------------------------------------------------------------------
// Session state reconstruction
// ---------------------------------------------------------------------------

function eohJsonlPath(dir: string): string {
	return sessionFilePath(dir, "log");
}

function eohPromptPath(dir: string): string {
	return sessionFilePath(dir, "prompt");
}

function reconstructState(cwd: string): EohState {
	const state = createEohState();

	const jsonlPath = eohJsonlPath(cwd);
	const hasPersistedLog = fs.existsSync(jsonlPath);

	try {
		if (hasPersistedLog) {
			const reconstructed = reconstructEohState(
				fs.readFileSync(jsonlPath, "utf-8"),
			);
			state.name = reconstructed.name;
			state.populationSize = reconstructed.populationSize;
			state.maxGenerations = reconstructed.maxGenerations;
			state.currentSegment = reconstructed.currentSegment;
			state.results = reconstructed.results.map(r => ({
				...r,
			}));
		}
	} catch {
		// Fall through
	}

	return state;
}

function formatNum(value: number, decimals = 4): string {
	if (!Number.isFinite(value)) return "—";
	if (Number.isInteger(value)) return String(value);
	return value.toFixed(decimals);
}
// ---------------------------------------------------------------------------
// Notify types
// ---------------------------------------------------------------------------

export type NotifyLevel = "info" | "warning" | "error";
export type NotifyFn = (message: string, level?: NotifyLevel) => void;

/** Result returned by every EohSession method — mirrors the real
 * agent-core ToolResult shape (content/details/isError) so callers can pass
 * it straight through as a tool's return value. */
export interface EohResult {
	content: string;
	details?: Record<string, unknown>;
	isError?: boolean;
}

/** Render-ready snapshot for a persistent status widget. */
export interface EohWidgetSummary {
	active: boolean;
	name: string | null;
	populationSize: number;
	generation: number;
	bestFitness: number | null;
	meanFitness: number | null;
	worstFitness: number | null;
	totalLLMCalls: number;
	running: { description: string; elapsedMs: number } | null;
	maxGenerations: number | null;
}

/** One row of the full-screen dashboard's results table. */
export interface EohDashboardRow {
	run: number;
	thought: string;
	fitness: number;
	fitnessFormatted: string;
	generation: number;
	createdBy: string;
	parentIds: string[];
	status: "keep" | "discard" | "crash";
	description: string;
	timestamp: number;
	isBest: boolean;
}

/** Full-screen dashboard snapshot. */
export interface EohDashboardData {
	summary: EohWidgetSummary | null;
	rows: EohDashboardRow[];
}

// ---------------------------------------------------------------------------
// EohSession — owns one EoH session's runtime state
// ---------------------------------------------------------------------------

export class EohSession {
	private runtime: EohRuntime;
	private engine: EohEngine | null = null;
	private readonly cwd: string;
	private readonly notify: NotifyFn;

	constructor(cwd: string, notify: NotifyFn = () => {}) {
		this.cwd = cwd;
		this.notify = notify;
		this.runtime = createSessionRuntime();
	}

	/** Call once when a session/turn starts, to reload persisted state from
	 * .eoh/log.jsonl (e.g. after a restart or context reset). */
	reload(): void {
		this.runtime.state = reconstructState(this.cwd);
	}

	isActive(): boolean {
		return this.runtime.eohMode;
	}

	getRuntime(): Readonly<EohRuntime> {
		return this.runtime;
	}

	/**
	 * Compact, render-ready snapshot for a persistent status widget.
	 */
	getWidgetSummary(): EohWidgetSummary | null {
		const state = this.runtime.state;
		if (!this.runtime.eohMode && state.results.length === 0) {
			return null;
		}
		const engine = this.engine;
		const engineState = engine?.getState();
		const stats = engineState
			? populationStats(engineState.population)
			: { best: 0, worst: 0, mean: 0, size: 0 };

		return {
			active: this.runtime.eohMode,
			name: state.name,
			populationSize: state.populationSize,
			generation: engineState?.generation ?? 0,
			bestFitness: stats.best > 0 ? stats.best : null,
			meanFitness: stats.mean > 0 ? stats.mean : null,
			worstFitness: stats.worst > 0 ? stats.worst : null,
			totalLLMCalls: engineState?.totalLLMCalls ?? 0,
			running: this.runtime.runningGeneration
				? {
						description: this.runtime.runningGeneration.description,
						elapsedMs: Date.now() - this.runtime.runningGeneration.startedAt,
					}
				: null,
			maxGenerations: state.maxGenerations,
		};
	}

	/** Full results table for the fullscreen dashboard overlay. */
	getDashboardData(): EohDashboardData {
		const state = this.runtime.state;
		const engine = this.engine;
		const engineState = engine?.getState();
		const bestFitness = engineState
			? populationStats(engineState.population).best
			: null;

		const rows: EohDashboardRow[] = state.results.map((result, i) => ({
			run: i + 1,
			thought: result.thought.slice(0, 100),
			fitness: result.fitness,
			fitnessFormatted: formatNum(result.fitness),
			generation: result.generation,
			createdBy: result.createdBy,
			parentIds: result.parentIds,
			status: result.status,
			description: result.description,
			timestamp: result.timestamp,
			isBest:
				result.status === "keep" &&
				bestFitness !== null &&
				result.fitness === bestFitness,
		}));
		return { summary: this.getWidgetSummary(), rows };
	}

	onAgentStart(): void {
		this.runtime.generationsThisSession = 0;
	}

	onAgentEnd(): void {
		this.runtime.runningGeneration = null;
	}

	shutdown(): void {
		stopDashboardServer();
	}

	compactionSummary(): string {
		return buildEohCompactionSummary(eohSummaryPathsFor(this.getWorkDir()));
	}

	private getWorkDir(): string {
		return this.cwd;
	}

	private setEohMode(enabled: boolean): void {
		this.runtime.eohMode = enabled;
	}

	private getEngine(): EohEngine {
		if (!this.engine) {
			this.engine = new EohEngine();
		}
		return this.engine;
	}

	private hasEohRules(): boolean {
		return fs.existsSync(eohPromptPath(this.getWorkDir()));
	}

	private readJsonlLines(workDir: string): string[] {
		const jsonlPath = eohJsonlPath(workDir);
		if (!fs.existsSync(jsonlPath)) return [];
		return fs.readFileSync(jsonlPath, "utf-8").split("\n").filter(Boolean);
	}

	private readLastRun(workDir: string): { generation: number } | null {
		const lines = this.readJsonlLines(workDir);
		for (let i = lines.length - 1; i >= 0; i--) {
			const entry = JSON.parse(lines[i]) as Record<string, unknown>;
			if (typeof entry.run === "number") {
				return {
					generation:
						typeof entry.generation === "number" ? entry.generation : 0,
				};
			}
		}
		return null;
	}

	private buildSessionSnapshot(): SessionSnapshot {
		const state = this.runtime.state;
		const engine = this.engine;
		const engineState = engine?.getState();
		const stats = engineState
			? populationStats(engineState.population)
			: { best: 0, worst: 0, mean: 0, size: 0 };

		return {
			goal: state.name ?? "",
			population_size: state.populationSize,
			generation: engineState?.generation ?? 0,
			best_fitness: stats.best > 0 ? stats.best : null,
			mean_fitness: stats.mean > 0 ? stats.mean : null,
			run_count: state.results.length,
		};
	}

	private async fireHook(payload: HookPayload): Promise<string | null> {
		const result = await runHook(payload);
		appendHookLogEntryIfConfigured(
			eohJsonlPath(payload.cwd),
			payload.event,
			result,
		);
		return steerMessageFor(payload.event, result);
	}
	// -----------------------------------------------------------------------
	// init_evolution tool
	// -----------------------------------------------------------------------

	/** Initialize the evolution session: name, problem definition, config.
	 * Writes the config header to .eoh/log.jsonl. */
	async initEvolution(params: Record<string, unknown>): Promise<EohResult> {
		const state = this.runtime.state;
		const isReinit = state.results.length > 0;

		state.name = params.name as string;
		if (
			typeof params.populationSize === "number" &&
			params.populationSize > 0
		) {
			state.populationSize = params.populationSize as number;
		}
		if (
			typeof params.maxGenerations === "number" &&
			params.maxGenerations >= 0
		) {
			state.maxGenerations = params.maxGenerations as number;
		}
		if (params.direction === "lower" || params.direction === "higher") {
			state.bestDirection = params.direction as "lower" | "higher";
		}

		if (isReinit) {
			state.currentSegment++;
		}

		const workDir = this.getWorkDir();
		try {
			const jsonlPath = eohJsonlPath(workDir);
			ensureParentDir(jsonlPath);
			const config = JSON.stringify({
				type: "eoh_config",
				name: state.name,
				populationSize: state.populationSize,
				maxGenerations: state.maxGenerations,
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
				content: `⚠️ Failed to write .eoh/log.jsonl: ${e instanceof Error ? e.message : String(e)}`,
				details: {},
			};
		}

		this.setEohMode(true);

		const steer = await this.fireHook({
			event: "before",
			cwd: workDir,
			next_generation: state.results.length + 1,
			last_generation: this.readLastRun(workDir)?.generation ?? 0,
			session: this.buildSessionSnapshot(),
		});

		const reinitNote = isReinit
			? " (re-initialized — new generation needed)"
			: "";
		const limitNote =
			state.maxGenerations !== null && state.maxGenerations > 0
				? `\nMax generations: ${state.maxGenerations}`
				: "";
		const workDirNote =
			workDir !== this.cwd ? `\nWorking directory: ${workDir}` : "";

		return {
			content: `✅ Evolution initialized: "${state.name}"${reinitNote}\nPopulation: ${state.populationSize}${limitNote}${workDirNote}\nConfig written to .eoh/log.jsonl.${steer ? `\n\n${steer}` : ""}`,
			details: {
				state: { ...state, results: state.results.map(r => ({ ...r })) },
			},
		};
	}

	// -----------------------------------------------------------------------
	// run_generation tool
	// -----------------------------------------------------------------------

	/** Run one generation of evolution: apply all 5 operators, evaluate, select next population. */
	async runGeneration(
		params: Record<string, unknown> = {},
	): Promise<EohResult> {
		const workDir = this.getWorkDir();
		const state = this.runtime.state;

		if (state.maxGenerations !== null && state.maxGenerations > 0) {
			const segCount = state.results.filter(
				r => r.segment === state.currentSegment,
			).length;
			if (segCount >= state.maxGenerations) {
				return {
					content: `🛑 Maximum generations reached (${state.maxGenerations}).`,
					details: {},
				};
			}
		}

		const engine = this.getEngine();
		const engineState = engine.getState();

		if (!engineState.problem) {
			return {
				content:
					"❌ No problem set. Call initEvolution with a problem definition first.",
				details: {},
			};
		}

		if (engineState.population.length === 0) {
			return {
				content:
					"❌ Population empty. Call initEvolution to initialize the population first.",
				details: {},
			};
		}

		this.runtime.runningGeneration = {
			startedAt: Date.now(),
			description: `Generation ${engineState.generation + 1}`,
		};

		try {
			const candidates = await engine.runGeneration();
			const stats = populationStats(engineState.population);

			// Log the generation result
			const generationResult: EohRunResult = {
				thought: `Generation ${engineState.generation} complete`,
				code: "",
				fitness: stats.best,
				generation: engineState.generation,
				createdBy: "generation",
				parentIds: [],
				status: stats.best > 0 ? "keep" : "crash",
				description: `${candidates.length} candidates evaluated`,
				timestamp: Date.now(),
				segment: state.currentSegment,
			};

			state.results.push(generationResult);
			this.runtime.generationsThisSession++;

			// Write to JSONL
			const jsonlEntry = {
				run: state.results.length,
				...generationResult,
			};
			try {
				const jsonlPath = eohJsonlPath(workDir);
				ensureParentDir(jsonlPath);
				fs.appendFileSync(jsonlPath, `${JSON.stringify(jsonlEntry)}\n`);
				broadcastDashboardUpdate(workDir);
			} catch (_e) {
				// ignore
			}

			// Fire hooks
			await this.fireHook({
				event: "after",
				cwd: workDir,
				generation: engineState.generation,
				best_fitness: stats.best,
				population_size: stats.size,
				session: this.buildSessionSnapshot(),
			});

			const limitReached =
				state.maxGenerations !== null &&
				state.maxGenerations > 0 &&
				state.results.filter(r => r.segment === state.currentSegment).length >=
					state.maxGenerations;

			let text = `🧬 Generation ${engineState.generation} complete\n`;
			text += `Population: ${stats.size}\n`;
			text += `Best fitness: ${formatNum(stats.best)}\n`;
			text += `Mean fitness: ${formatNum(stats.mean)}\n`;
			text += `Worst fitness: ${formatNum(stats.worst)}\n`;
			text += `LLM calls: ${engineState.totalLLMCalls}\n`;
			text += `Candidates: ${candidates.length}\n`;

			if (limitReached) {
				text += `\n🛑 Maximum generations reached (${state.maxGenerations}). STOP.`;
				this.setEohMode(false);
			}

			return {
				content: text,
				details: {
					generation: engineState.generation,
					stats,
					candidates: candidates.slice(0, 5).map(c => ({
						thought: c.thought.slice(0, 200),
						code: c.code.slice(0, 500),
						fitness: c.fitness,
					})),
				},
			};
		} catch (error) {
			return {
				content: `❌ Generation failed: ${error instanceof Error ? error.message : String(error)}`,
				details: { error: String(error) },
				isError: true,
			};
		} finally {
			this.runtime.runningGeneration = null;
		}
	}
	// -----------------------------------------------------------------------
	// status / best / stop
	// -----------------------------------------------------------------------

	/** Get current evolution status. */
	getStatus(): EohResult {
		const engine = this.getEngine();
		const engineState = engine.getState();
		const stats = populationStats(engineState.population);
		const state = this.runtime.state;

		return {
			content: [
				"EoH Status",
				`  Running: ${engineState.running}`,
				`  Generation: ${engineState.generation}`,
				`  LLM calls: ${engineState.totalLLMCalls}`,
				`  Population: ${stats.size}`,
				`  Best fitness: ${formatNum(stats.best)}`,
				`  Mean fitness: ${formatNum(stats.mean)}`,
				`  Worst fitness: ${formatNum(stats.worst)}`,
				`  Sessions: ${state.results.length} runs`,
				`  Mode: ${this.runtime.eohMode ? "ON" : "OFF"}`,
			].join("\n"),
			details: {
				state: engineState,
				stats,
			},
		};
	}

	/** Get the best heuristic found so far. */
	getBest(): EohResult {
		const engine = this.getEngine();
		const best = engine.getBestHeuristic();
		if (!best) {
			return { content: "No heuristics yet — run init_evolution first" };
		}
		return {
			content: `Best heuristic (fitness=${formatNum(best.fitness)}, gen=${best.generation}, by=${best.createdBy}):

Thought:
${best.thought}

Code:
\`\`\`python
${best.code}
\`\`\``,
			details: { heuristic: best },
		};
	}

	/** Stop the current evolution. */
	stopEvolution(): EohResult {
		const engine = this.getEngine();
		engine.stop();
		this.setEohMode(false);
		this.runtime.generationsThisSession = 0;
		stopDashboardServer();
		return { content: "EoH stop signal sent" };
	}

	/** Clear all EoH state. */
	clear(): EohResult {
		const workDir = this.getWorkDir();
		this.setEohMode(false);
		this.runtime.generationsThisSession = 0;
		this.runtime.state = createEohState();
		stopDashboardServer();

		const jsonlPath = eohJsonlPath(workDir);
		if (fs.existsSync(jsonlPath)) {
			try {
				fs.unlinkSync(jsonlPath);
			} catch {
				/* ignore */
			}
		}

		return { content: "EoH cleared and mode OFF" };
	}

	// -----------------------------------------------------------------------
	// /eoh command
	// -----------------------------------------------------------------------

	private eohHelp(): string {
		return [
			"Usage: /eoh [off|clear|export|<text>]",
			"",
			"<text> enters EoH mode and starts the evolution loop.",
			"off leaves EoH mode.",
			"clear deletes the session log and turns EoH mode off.",
			"export opens a local live dashboard in your browser.",
			"",
			"Examples:",
			"  /eoh evolve bin packing heuristic, maximize fitness",
			"  /eoh export",
		].join("\n");
	}

	/**
	 * Handle the `/eoh [off|clear|export|<text>]` command.
	 */
	async handleCommand(
		args: string | undefined,
	): Promise<{ message: string; injectAsTurn: boolean }> {
		const trimmedArgs = (args ?? "").trim();
		const command = trimmedArgs.toLowerCase();

		if (!trimmedArgs) {
			return { message: this.eohHelp(), injectAsTurn: false };
		}

		if (command === "off") {
			this.setEohMode(false);
			this.runtime.autoResumeTurns = 0;
			this.runtime.generationsThisSession = 0;
			stopDashboardServer();
			return { message: "EoH mode OFF", injectAsTurn: false };
		}

		if (command === "export") {
			await exportDashboard(this.notify, this.getWorkDir());
			return { message: "", injectAsTurn: false };
		}

		if (command === "clear") {
			return { message: this.clear().content, injectAsTurn: false };
		}

		if (this.runtime.eohMode) {
			return {
				message: "EoH already active — use '/eoh off' to stop first",
				injectAsTurn: false,
			};
		}

		const workDir = this.getWorkDir();
		this.setEohMode(true);

		const rulesLoaded = this.hasEohRules();
		const kickoff = rulesLoaded
			? `EoH mode active. ${trimmedArgs}`
			: `EoH mode ON — no rules found, use /eoh to configure`;

		const activationSteer = await this.fireHook({
			event: "before",
			cwd: workDir,
			next_generation: this.runtime.state.results.length + 1,
			last_generation: this.readLastRun(workDir)?.generation ?? 0,
			session: this.buildSessionSnapshot(),
		});

		const message =
			activationSteer && rulesLoaded
				? `${activationSteer}\n\n${kickoff}`
				: kickoff;
		this.notify(
			rulesLoaded
				? "EoH mode ON — rules loaded from .eoh/prompt.md"
				: "EoH mode ON — no .eoh/prompt.md found",
			"info",
		);

		return { message, injectAsTurn: true };
	}
}
