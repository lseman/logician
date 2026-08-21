// ── PersistentTerminalManager ───────────────────────────────────────────────
// Maintains long-lived interactive shell sessions so that environment variables,
// working directory changes, and shell functions persist across tool calls.

import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
	getShellConfig,
	getShellEnv,
	killProcessTree,
	trackDetachedChildPid,
	untrackDetachedChildPid,
} from "./shell.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	OutputAccumulator,
	type OutputSnapshot,
	type TruncationResult,
} from "./truncate.ts";

export interface PersistentTerminalSession {
	id: string;
	cwd: string;
	child: ChildProcess;
	createdAt: number;
	busy: boolean;
	buffer: string;
	errBuffer: string;
}

export interface PersistentExecutionResult {
	content: string;
	exitCode: number | null;
	signal: string | null;
	status: "completed" | "failed" | "timed_out" | "aborted";
	terminalId: string;
	details?: Record<string, unknown>;
}

export class PersistentTerminalManager {
	private terminals = new Map<string, PersistentTerminalSession>();

	getTerminal(id = "default"): PersistentTerminalSession | undefined {
		return this.terminals.get(id);
	}

	getOrCreate(
		id = "default",
		cwd?: string,
	): PersistentTerminalSession {
		const existing = this.terminals.get(id);
		if (existing && !existing.child.killed && existing.child.exitCode === null) {
			return existing;
		}

		const resolvedCwd = cwd || process.cwd();
		const { shell } = getShellConfig();
		const shellEnv = getShellEnv(resolvedCwd);

		// Launch shell with stdin/stdout/stderr pipes
		const child = spawn(shell, ["--noprofile", "--norc", "-s"], {
			cwd: resolvedCwd,
			stdio: ["pipe", "pipe", "pipe"],
			env: shellEnv,
			detached: process.platform !== "win32",
		});

		if (child.pid && process.platform !== "win32") {
			trackDetachedChildPid(child.pid);
		}

		const session: PersistentTerminalSession = {
			id,
			cwd: resolvedCwd,
			child,
			createdAt: Date.now(),
			busy: false,
			buffer: "",
			errBuffer: "",
		};

		child.on("close", () => {
			if (child.pid) untrackDetachedChildPid(child.pid);
			this.terminals.delete(id);
		});

		child.on("error", () => {
			if (child.pid) untrackDetachedChildPid(child.pid);
			this.terminals.delete(id);
		});

		this.terminals.set(id, session);
		return session;
	}

	async execute(
		id = "default",
		command: string,
		options: {
			cwd?: string;
			timeout?: number;
			signal?: AbortSignal;
			onUpdate?: (delta: string) => void;
		} = {},
	): Promise<PersistentExecutionResult> {
		const session = this.getOrCreate(id, options.cwd);

		if (session.busy) {
			return {
				content: `Error: Persistent terminal "${id}" is currently busy executing another command.`,
				exitCode: null,
				signal: null,
				status: "failed",
				terminalId: id,
			};
		}

		session.busy = true;
		const execId = randomBytes(6).toString("hex");
		const delimiterPrefix = `__LOGICIAN_DELIM_${execId}_`;
		const outDelimiterRegex = new RegExp(
			`__LOGICIAN_DELIM_${execId}_(-?\\d+)__\\r?\\n?`,
		);

		const accumulator = new OutputAccumulator({
			tempFilePrefix: `terminal-${id}`,
		});

		return new Promise<PersistentExecutionResult>(resolve => {
			let settled = false;
			let timeoutHandle: NodeJS.Timeout | undefined;
			let timedOut = false;

			const cleanup = () => {
				session.busy = false;
				if (timeoutHandle) clearTimeout(timeoutHandle);
				session.child.stdout?.removeListener("data", onStdout);
				session.child.stderr?.removeListener("data", onStderr);
				session.child.removeListener("close", onClose);
				options.signal?.removeEventListener("abort", onAbort);
			};

			const finish = (result: PersistentExecutionResult) => {
				if (!settled) {
					settled = true;
					cleanup();
					accumulator.finish();
					resolve(result);
				}
			};

			const onClose = (code: number | null, signal: string | null) => {
				const snapshot = accumulator.snapshot();
				finish({
					content:
						snapshot.content ||
						(code !== 0 && code !== null
							? `Process exited with code ${code}`
							: "(no output)"),
					exitCode: code,
					signal,
					status: code === 0 ? "completed" : "failed",
					terminalId: id,
				});
			};

			session.child.once("close", onClose);

			const onAbort = () => {
				// Send SIGINT (Ctrl+C) to persistent shell stdin
				try {
					session.child.stdin?.write("\x03\n");
				} catch {
					// Ignore stdin write error
				}
				finish({
					content: accumulator.snapshot().content || "Command aborted",
					exitCode: null,
					signal: "SIGINT",
					status: "aborted",
					terminalId: id,
				});
			};

			if (options.signal) {
				if (options.signal.aborted) {
					onAbort();
					return;
				}
				options.signal.addEventListener("abort", onAbort, { once: true });
			}

			if (options.timeout && options.timeout > 0) {
				timeoutHandle = setTimeout(() => {
					timedOut = true;
					try {
						session.child.stdin?.write("\x03\n");
					} catch {
						// Ignore
					}
					finish({
						content:
							accumulator.snapshot().content ||
							`Command timed out after ${options.timeout}s`,
						exitCode: null,
						signal: null,
						status: "timed_out",
						terminalId: id,
					});
				}, options.timeout * 1000);
			}

			let capturedOutput = "";

			const onStdout = (chunk: Buffer) => {
				const text = chunk.toString("utf8");
				capturedOutput += text;
				accumulator.append(chunk);

				if (options.onUpdate) {
					options.onUpdate(accumulator.snapshot().content);
				}

				const match = outDelimiterRegex.exec(capturedOutput);
				if (match) {
					const exitCode = parseInt(match[1], 10);
					const rawContent = capturedOutput.replace(outDelimiterRegex, "");
					const trimmedContent = rawContent.replace(/\r?\n$/, "");

					finish({
						content: trimmedContent || "(no output)",
						exitCode: Number.isNaN(exitCode) ? 0 : exitCode,
						signal: null,
						status: exitCode === 0 ? "completed" : "failed",
						terminalId: id,
						details: {
							terminalId: id,
							durationMs: Date.now() - session.createdAt,
						},
					});
				}
			};

			const onStderr = (chunk: Buffer) => {
				accumulator.append(chunk);
				if (options.onUpdate) {
					options.onUpdate(accumulator.snapshot().content);
				}
			};

			session.child.stdout?.on("data", onStdout);
			session.child.stderr?.on("data", onStderr);

			// Write wrapped command to persistent shell stdin
			const script = [
				`{`,
				command,
				`}`,
				`__LOGICIAN_EXIT=$?`,
				`echo "${delimiterPrefix}\${__LOGICIAN_EXIT}__"`,
				"",
			].join("\n");

			try {
				session.child.stdin?.write(script);
			} catch (err) {
				finish({
					content: `Error writing to persistent terminal: ${(err as Error).message}`,
					exitCode: null,
					signal: null,
					status: "failed",
					terminalId: id,
				});
			}
		});
	}

	close(id: string): void {
		const session = this.terminals.get(id);
		if (session) {
			if (session.child.pid) {
				killProcessTree(session.child.pid);
				untrackDetachedChildPid(session.child.pid);
			} else {
				session.child.kill("SIGKILL");
			}
			this.terminals.delete(id);
		}
	}

	closeAll(): void {
		for (const id of Array.from(this.terminals.keys())) {
			this.close(id);
		}
	}
}

/** Global default PersistentTerminalManager instance. */
export const defaultPersistentTerminalManager =
	new PersistentTerminalManager();
