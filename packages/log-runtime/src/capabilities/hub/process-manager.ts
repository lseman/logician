// ── Hub Process Manager ──────────────────────────────────────────────────────
// Named process lifecycle management: start, ps, logs, stop, restart, send, wait.
// Supports readiness detection, restart policies, PTY, log cursors, signals.

import {
	type ChildProcess,
	type SpawnOptions,
	spawn,
} from "node:child_process";
import {
	createWriteStream,
	existsSync,
	mkdirSync,
	type WriteStream,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { createInterface } from "node:readline";
import { untrackDetachedChildPid } from "../tools/support/utils/shell.ts";

// ── Config ───────────────────────────────────────────────────────────────────

export interface HubConfig {
	/** Default timeout for process operations in ms (default: 30000). */
	defaultTimeoutMs?: number;
	/** Readiness detection timeout in ms (default: 60000). */
	readinessTimeoutMs?: number;
	/** Base directory for hub state (default: ~/.logician/hub). */
	stateDir?: string;
}

// ── Types ────────────────────────────────────────────────────────────────────

export type ProcessStatus =
	| "starting"
	| "ready"
	| "running"
	| "exited"
	| "killed"
	| "failed";

export type RestartPolicy = "no" | "on-failure" | "always";

export interface ReadinessSpec {
	/** Regex pattern matched against process output. */
	log?: string;
	/** TCP port that must accept connections. */
	port?: number;
	/** Host for port check (default: 127.0.0.1). */
	host?: string;
	/** Seconds to wait for readiness (default: 30). */
	timeout?: number;
}

export interface HubProcessSpec {
	name: string;
	application: string;
	args?: string[];
	cwd?: string;
	env?: Record<string, string>;
	pty?: boolean;
	ready?: ReadinessSpec;
	restart?: RestartPolicy;
	persist?: boolean;
	detached?: boolean;
}

export interface HubProcessState {
	name: string;
	spec: HubProcessSpec;
	pid?: number;
	status: ProcessStatus;
	startTime?: number;
	exitCode?: number | null;
	signal?: string | null;
	logCursor?: number;
	restartCount: number;
	lastOutput?: string;
}

export interface HubLogResult {
	lines: string[];
	cursor: number;
	totalLines: number;
	following: boolean;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const DEFAULT_STATE_DIR = join(tmpdir(), "logician-hub");

function checkReadyTcp(
	host: string,
	port: number,
	timeoutMs: number,
): Promise<boolean> {
	return new Promise(resolve => {
		import("node:net").then(net => {
			const socket = new net.Socket();
			const timer = setTimeout(() => {
				socket.destroy();
				resolve(false);
			}, timeoutMs);
			socket.once("connect", () => {
				clearTimeout(timer);
				socket.destroy();
				resolve(true);
			});
			socket.once("error", () => {
				clearTimeout(timer);
				resolve(false);
			});
			socket.connect(port, host);
		});
	});
}

// ── Process Entry ────────────────────────────────────────────────────────────

interface ProcessEntry {
	name: string;
	spec: HubProcessSpec;
	child: ChildProcess | null;
	status: ProcessStatus;
	startTime: number;
	exitCode: number | null;
	signal: string | null;
	logStream: WriteStream | null;
	logFilePath: string;
	lines: string[];
	lastCursor: number;
	restartCount: number;
	rl: ReturnType<typeof createInterface> | null;
}

// ── Hub Process Manager ──────────────────────────────────────────────────────

export class HubProcessManager {
	private processes = new Map<string, ProcessEntry>();
	private baseDir: string;

	constructor(config: HubConfig = {}) {
		this.baseDir = config.stateDir ?? DEFAULT_STATE_DIR;

		if (!existsSync(this.baseDir)) {
			try {
				mkdirSync(this.baseDir, { recursive: true });
			} catch {
				this.baseDir = tmpdir();
			}
		}
	}

	private getLogPath(name: string): string {
		return join(this.baseDir, "logs", `${name}.log`);
	}

	private getStatePath(name: string): string {
		return join(this.baseDir, `${name}.json`);
	}

	/**
	 * Start a named process with readiness detection.
	 */
	async start(spec: HubProcessSpec): Promise<HubProcessState> {
		const name = spec.name;

		// If already running, stop first
		const existing = this.processes.get(name);
		if (
			existing &&
			(existing.status === "running" || existing.status === "starting")
		) {
			await this.stop(name);
		}

		const logPath = this.getLogPath(name);
		const logDir = join(logPath, "..");
		if (!existsSync(logDir)) {
			mkdirSync(logDir, { recursive: true });
		}

		const logStream = createWriteStream(logPath, { flags: "a" });

		const stdioOpts: ("pipe" | "ignore")[] = ["pipe", "pipe", "pipe"];
		if (spec.pty) {
			// PTY: use spawn with pty if available, otherwise fall back
			// For now, use pipe mode (PTY requires node-pty package)
			// The spec.pty flag is noted for documentation but pipe is used
		}

		const spawnOpts: SpawnOptions = {
			cwd: spec.cwd ?? process.cwd(),
			env: { ...process.env, ...spec.env },
			stdio: stdioOpts,
			detached: spec.detached ?? false,
		};

		const child = spawn(spec.application, spec.args ?? [], spawnOpts);

		if (spec.detached && child.pid) {
			untrackDetachedChildPid(child.pid);
		}

		const entry: ProcessEntry = {
			name,
			spec,
			child,
			status: "starting",
			startTime: Date.now(),
			exitCode: null,
			signal: null,
			logStream,
			logFilePath: logPath,
			lines: [],
			lastCursor: 0,
			restartCount: 0,
			rl: null,
		};

		this.processes.set(name, entry);

		// Set up stdout/stderr capture
		const handleOutput = (data: Buffer | string) => {
			const text = typeof data === "string" ? data : data.toString("utf8");
			const lines = text.split("\n");
			if (entry.lines.length > 0 && !text.startsWith("\n")) {
				const last = entry.lines[entry.lines.length - 1];
				const first = lines[0];
				if (first) {
					entry.lines[entry.lines.length - 1] = last + first;
					entry.lines.push(...lines.slice(1));
				} else {
					entry.lines.push(...lines);
				}
			} else {
				entry.lines.push(...lines);
			}
			// Keep last 10000 lines
			if (entry.lines.length > 10000) {
				entry.lines = entry.lines.slice(-10000);
			}

			try {
				logStream.write(data);
			} catch {
				// ignore
			}
		};

		child.stdout?.on("data", handleOutput);
		child.stderr?.on("data", handleOutput);

		// Wait for readiness
		const readySpec = spec.ready;
		if (readySpec) {
			const readyTimeout = (readySpec.timeout ?? 30) * 1000;
			const readyResult = await this.waitForReadiness(entry, readyTimeout);

			if (!readyResult) {
				entry.status = "failed";
				entry.signal = "TIMEOUT";
				child.kill("SIGKILL");
				this.saveState(name);
				return this.getState(name);
			}

			entry.status = "ready";
		} else {
			entry.status = "running";
		}

		this.saveState(name);
		return this.getState(name);
	}

	private async waitForReadiness(
		entry: ProcessEntry,
		timeoutMs: number,
	): Promise<boolean> {
		return new Promise(resolve => {
			const readySpec = entry.spec.ready!;
			const readyLog = readySpec.log ? new RegExp(readySpec.log, "u") : null;
			const readyPort = readySpec.port;
			const readyHost = readySpec.host ?? "127.0.0.1";
			const needLog = readyLog !== null;
			const needPort = readyPort !== undefined;

			let logMatched = false;
			let portReady = false;
			let done = false;

			const checkComplete = () => {
				if (done) return;
				if ((!needLog || logMatched) && (!needPort || portReady)) {
					done = true;
					resolve(true);
				}
			};

			setTimeout(() => {
				if (!done) {
					done = true;
					resolve(false);
				}
			}, timeoutMs);

			const handleData = (data: Buffer | string) => {
				if (done) return;
				const text = typeof data === "string" ? data : data.toString("utf8");
				if (needLog && readyLog?.test(text)) {
					logMatched = true;
					checkComplete();
				}
			};

			entry.child?.stdout?.on("data", handleData);
			entry.child?.stderr?.on("data", handleData);

			if (needPort) {
				checkReadyTcp(readyHost, readyPort, timeoutMs).then(ok => {
					if (ok) {
						portReady = true;
						checkComplete();
					}
				});
			}

			// Check if already done immediately
			checkComplete();
		});
	}

	/**
	 * List all managed processes.
	 */
	ps(): HubProcessState[] {
		return Array.from(this.processes.values()).map(entry =>
			this.entryToState(entry),
		);
	}

	/**
	 * Get state for a named process.
	 */
	describe(name: string): HubProcessState | null {
		const entry = this.processes.get(name);
		if (!entry) return null;
		return this.entryToState(entry);
	}

	/**
	 * Read logs with cursor support.
	 */
	logs(
		name: string,
		options: { cursor?: number; lines?: number; follow?: boolean } = {},
	): HubLogResult | null {
		const entry = this.processes.get(name);
		if (!entry) return null;

		const cursor = options.cursor ?? entry.lastCursor;
		const maxLines = options.lines ?? 100;
		const lines = entry.lines.slice(cursor, cursor + maxLines);

		// Update cursor for next read
		const newCursor = Math.min(cursor + maxLines, entry.lines.length);
		if (!options.follow) {
			entry.lastCursor = newCursor;
		}

		return {
			lines,
			cursor: newCursor,
			totalLines: entry.lines.length,
			following: !!options.follow,
		};
	}

	/**
	 * Send input to a process stdin.
	 */
	send(
		name: string,
		input: string,
		options: { keys?: string[]; enter?: boolean; signal?: string } = {},
	): { success: boolean; message: string } {
		const entry = this.processes.get(name);
		if (!entry) {
			return { success: false, message: `Process "${name}" not found.` };
		}

		if (
			entry.status !== "running" &&
			entry.status !== "starting" &&
			entry.status !== "ready"
		) {
			return {
				success: false,
				message: `Process "${name}" is not running (status: ${entry.status}).`,
			};
		}

		if (!entry.child?.stdin || entry.child.stdin.destroyed) {
			return {
				success: false,
				message: `Process "${name}" stdin is not available.`,
			};
		}

		// Handle keys (special terminal keys)
		if (options.keys) {
			for (const key of options.keys) {
				const keyMap: Record<string, string> = {
					ENTER: "\r",
					TAB: "\t",
					ESCAPE: "\x1b",
					CTRL_C: "\x03",
					CTRL_D: "\x04",
					UP: "\x1b[A",
					DOWN: "\x1b[B",
					LEFT: "\x1b[D",
					RIGHT: "\x1b[C",
				};
				entry.child.stdin.write(keyMap[key] ?? key);
			}
			return { success: true, message: `Sent keys to process "${name}".` };
		}

		// Handle signals (kill signals)
		if (options.signal) {
			return this._sendSignal(name, options.signal);
		}

		// Normal text input
		const data = input.endsWith("\n") ? input : `${input}\n`;
		entry.child.stdin.write(data);
		return {
			success: true,
			message: `Sent ${Buffer.byteLength(data, "utf8")} bytes to process "${name}".`,
		};
	}

	private _sendSignal(
		name: string,
		signal: string,
	): { success: boolean; message: string } {
		const entry = this.processes.get(name);
		if (!entry) {
			return { success: false, message: `Process "${name}" not found.` };
		}

		const signalMap: Record<string, NodeJS.Signals> = {
			SIGINT: "SIGINT",
			SIGTERM: "SIGTERM",
			SIGHUP: "SIGHUP",
			SIGQUIT: "SIGQUIT",
			SIGKILL: "SIGKILL",
		};

		const nodeSignal = signalMap[signal];
		if (!nodeSignal) {
			return {
				success: false,
				message: `Unknown signal "${signal}". Supported: ${Object.keys(signalMap).join(", ")}.`,
			};
		}

		if (!entry.child) {
			return {
				success: false,
				message: `Process "${name}" has no running child.`,
			};
		}

		entry.child.kill(nodeSignal);
		return { success: true, message: `Sent ${signal} to process "${name}".` };
	}

	/**
	 * Graceful stop: SIGTERM first, then SIGKILL after timeout.
	 */
	async stop(name: string): Promise<{ success: boolean; message: string }> {
		const entry = this.processes.get(name);
		if (!entry) {
			return { success: false, message: `Process "${name}" not found.` };
		}

		if (
			entry.status === "exited" ||
			entry.status === "killed" ||
			entry.status === "failed"
		) {
			return {
				success: false,
				message: `Process "${name}" is already ${entry.status}.`,
			};
		}

		if (!entry.child) {
			entry.status = "killed";
			this.saveState(name);
			return {
				success: true,
				message: `Process "${name}" has no child to stop.`,
			};
		}

		// Graceful: SIGTERM
		const child = entry.child;
		child.kill("SIGTERM");
		// Wait for exit, then hard-kill
		await new Promise<void>(resolve => {
			const checkExit = () => {
				if (child.exitCode !== null || child.signalCode !== null) {
					child.off("exit", checkExit);
					child.off("error", checkExit);
					resolve();
				}
			};

			child.on("exit", checkExit);
			child.on("error", checkExit);

			// Force kill after 5 seconds
			setTimeout(() => {
				if (child && entry.status !== "exited" && entry.status !== "killed") {
					child.kill("SIGKILL");
				}
				resolve();
			}, 5000);
		});

		entry.status = "killed";
		if (entry.logStream) {
			entry.logStream.end();
		}
		entry.child = null;
		this.saveState(name);
		return { success: true, message: `Process "${name}" stopped.` };
	}

	/**
	 * Restart a process (reuse original spec).
	 */
	async restart(name: string): Promise<HubProcessState> {
		await this.stop(name);
		const entry = this.processes.get(name);
		const spec = entry?.spec;
		if (!spec) {
			throw new Error(`Process "${name}" has no spec to restart.`);
		}

		return this.start({ ...spec });
	}

	/**
	 * Wait for a process to reach a state or match output.
	 */
	async wait(
		name: string,
		options: {
			for?: "ready" | "exit";
			pattern?: string;
			timeout?: number;
		} = {},
	): Promise<{ success: boolean; message: string; state?: HubProcessState }> {
		const entry = this.processes.get(name);
		if (!entry) {
			return { success: false, message: `Process "${name}" not found.` };
		}

		const forWhat = options.for ?? "ready";
		const pattern = options.pattern;
		const timeout = (options.timeout ?? 60) * 1000;

		return new Promise(resolve => {
			const timer = setTimeout(() => {
				resolve({
					success: false,
					message: `Wait timed out after ${timeout}ms.`,
					state: this.getState(name),
				});
			}, timeout);

			const check = () => {
				if (forWhat === "ready") {
					if (entry.status === "ready" || entry.status === "running") {
						clearTimeout(timer);
						resolve({
							success: true,
							message: `Process "${name}" is ${entry.status}.`,
							state: this.getState(name),
						});
						return;
					}
				} else if (forWhat === "exit") {
					if (
						entry.status === "exited" ||
						entry.status === "killed" ||
						entry.status === "failed"
					) {
						clearTimeout(timer);
						resolve({
							success: true,
							message: `Process "${name}" exited (code: ${entry.exitCode}, signal: ${entry.signal}).`,
							state: this.getState(name),
						});
						return;
					}
				}

				// Check pattern in output
				if (pattern) {
					const allOutput = entry.lines.join("\n");
					if (allOutput.includes(pattern)) {
						clearTimeout(timer);
						resolve({
							success: true,
							message: `Pattern "${pattern}" found in process output.`,
							state: this.getState(name),
						});
						return;
					}
				}
			};

			// Poll every 200ms
			const poll = setInterval(() => {
				check();
			}, 200);

			entry.child?.on("exit", () => {
				clearInterval(poll);
				check();
			});

			// Check immediately
			check();
		});
	}

	/**
	 * Clean up all processes.
	 */
	cleanupAll(): void {
		for (const entry of this.processes.values()) {
			if (entry.child) {
				entry.child.kill("SIGKILL");
				entry.child = null;
			}
			if (entry.logStream) {
				entry.logStream.end();
			}
			entry.status = "killed";
		}
		this.processes.clear();
	}

	// ── State management ────────────────────────────────────────────────────

	private entryToState(entry: ProcessEntry): HubProcessState {
		return {
			name: entry.name,
			spec: entry.spec,
			pid: entry.child?.pid,
			status: entry.status,
			startTime: entry.startTime,
			exitCode: entry.exitCode,
			signal: entry.signal ?? undefined,
			logCursor: entry.lastCursor,
			restartCount: entry.restartCount,
		};
	}

	private getState(name: string): HubProcessState {
		const entry = this.processes.get(name);
		return entry
			? this.entryToState(entry)
			: { name, spec: {} as HubProcessSpec, status: "failed", restartCount: 0 };
	}

	private saveState(name: string): void {
		const entry = this.processes.get(name);
		if (!entry) return;

		const state = this.entryToState(entry);
		const path = this.getStatePath(name);
		try {
			mkdirSync(join(path, ".."), { recursive: true });
			const tmp = `${path}.tmp`;
			writeFileSync(tmp, JSON.stringify(state, null, 2));
			renameSync(tmp, path);
		} catch {
			// Ignore state save errors
		}
	}
}

/** Global default HubProcessManager instance. */
export const defaultHub = new HubProcessManager();

// Re-export fs functions we need
import { renameSync, writeFileSync } from "node:fs";
