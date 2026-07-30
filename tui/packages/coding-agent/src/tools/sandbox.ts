// ── sandbox tool ────────────────────────────────────────────────────────────────
// Execute commands inside a Bubblewrap-isolated sandbox (Linux only).
// Falls back to regular bash execution on non-Linux or when bwrap is unavailable.

import { spawn, spawnSync } from "node:child_process";
import { constants, access as fsAccess, rm as fsRm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import type { Tool, ToolResult } from "@logician/agent-core/agent/types.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	OutputAccumulator,
	type TruncationResult,
} from "./truncate.ts";
import {
	getShellConfig,
	getShellEnv,
	killProcessTree,
} from "./shell.ts";

// ── Types ──────────────────────────────────────────────────────────────────────

export interface SandboxDetails {
	bwrapAvailable: boolean;
	bwrapPath: string | null;
	profile: SandboxProfile;
	truncation?: TruncationResult;
	fullOutputPath?: string;
}

export type SandboxProfile = "none" | "code" | "full";

// ── Session default profile ─────────────────────────────────────────────────
// Applied when a tool call omits `profile`. Cycled by the UI (Ctrl+K); a
// model-supplied `profile` argument always overrides this.
let _defaultProfile: SandboxProfile = "code";

export function getDefaultSandboxProfile(): SandboxProfile {
	return _defaultProfile;
}

export function setDefaultSandboxProfile(profile: SandboxProfile): void {
	_defaultProfile = profile;
}

export interface SandboxRunResult {
	content: string;
	exitCode: number | null;
	signal: string | null;
	status: "completed" | "failed" | "timed_out" | "aborted";
	details: SandboxDetails;
}

// ── Bubblewrap detection ───────────────────────────────────────────────────────

let _bwrapCache: { available: boolean; path: string | null } | null = null;

function detectBwrap(): { available: boolean; path: string | null } {
	if (_bwrapCache) return _bwrapCache;

	const bwrapPath = findBwrapOnPath();
	let available = false;

	if (bwrapPath && process.platform === "linux") {
		try {
			const result = spawnSync(bwrapPath, ["--version"], {
				timeout: 5000,
				stdio: ["ignore", "pipe", "pipe"],
			} as Parameters<typeof spawnSync>[2]);
			if (result.status === 0 && result.stdout) {
				const versionStr = result.stdout.toString().trim();
				const match = versionStr.match(/bubblewrap\s+([\d.]+)/);
				if (match) {
					const version = match[1].split(".").map(Number);
					// Minimum v0.4.1 for --unshare-all
					available =
						version.length >= 3 &&
						(version[0] > 0 ||
							(version[0] === 0 && version[1] > 4) ||
							(version[0] === 0 &&
								version[1] === 4 &&
								version[2] >= 1));
				}
			}
		} catch {
			// bwrap binary exists but version check failed
			available = false;
		}
	}

	_bwrapCache = { available, path: bwrapPath };
	return _bwrapCache;
}

function findBwrapOnPath(): string | null {
	const pathEnv = process.env.PATH ?? "";
	const entries = pathEnv.split(path.delimiter);
	for (const dir of entries) {
		const fullPath = path.join(dir, "bwrap");
		if (existsSync(fullPath)) {
			return fullPath;
		}
	}
	return null;
}

// ── Bwrap command builder ──────────────────────────────────────────────────────

function buildBwrapCommand(
	command: string,
	profile: SandboxProfile,
	cwd: string,
	_sandboxTmpdir: string,
): string[] {
	const cmd: string[] = [];

	// Namespace isolation (FULL profile)
	if (profile === "full") {
		cmd.push("--unshare-all");
		cmd.push("--die-with-parent");
	}

	// Session + no new privileges
	cmd.push("--new-session");

	// Root filesystem — read-only bind of host
	cmd.push("--ro-bind", "/", "/");

	// Writable tmpfs
	cmd.push("--tmpfs", "/tmp");
	cmd.push("--tmpfs", "/var/tmp");

	// Minimal /dev (tmpfs-based, no host devices)
	cmd.push("--tmpfs", "/dev");
	cmd.push("--dir", "/dev");
	cmd.push("--symlink", "proc/self/fd", "/dev/fd");
	cmd.push("--symlink", "proc/self/fd/0", "/dev/stdin");
	cmd.push("--symlink", "proc/self/fd/1", "/dev/stdout");
	cmd.push("--symlink", "proc/self/fd/2", "/dev/stderr");

	// Proc filesystem
	cmd.push("--proc", "/proc");

	// Home directory
	cmd.push("--bind", path.join(_sandboxTmpdir, "home"), "/home");

	// Working directory
	cmd.push("--chdir", cwd);

	// Unmount IPC
	cmd.push("--unshare-ipc");

	// Execute
	cmd.push("--");
	cmd.push("bash", "-c", command);

	return cmd;
}

// ── Execution ──────────────────────────────────────────────────────────────────

async function executeSandboxed(
	command: string,
	profile: SandboxProfile,
	cwd: string,
	timeout: number | undefined,
	ctx: Parameters<Tool["execute"]>[1],
): Promise<SandboxRunResult> {
	const bwrap = detectBwrap();

	if (!bwrap.available) {
		// Fallback: execute normally without sandbox
		return {
			...((await executeFallback(command, cwd, timeout, ctx)) as unknown as Omit<SandboxRunResult, "details">),
			details: {
				bwrapAvailable: false,
				bwrapPath: bwrap.path,
				profile,
			},
		};
	}

	const sandboxTmpdir = path.join(tmpdir(), `logician_sandbox_${Date.now()}`);
	try {
		// Create sandbox directory structure
		const { mkdir } = await import("node:fs/promises");
		await mkdir(path.join(sandboxTmpdir, "home"), { recursive: true });

		const bwrapArgs = buildBwrapCommand(command, profile, cwd, sandboxTmpdir);

		const output = new OutputAccumulator({
			tempFilePrefix: "logician-sandbox",
		});
		const timeoutSeconds = timeout;
		let settled = false;
		let timedOut = false;
		let hasError = false;

		const settle = (fn: () => void) => {
			if (!settled) {
				settled = true;
				fn();
			}
		};

		const result = await new Promise<SandboxRunResult>((resolve) => {
			const child = spawn(bwrap.path!, bwrapArgs, {
				cwd,
				stdio: ["ignore", "pipe", "pipe"],
				env: getShellEnv(),
				timeout: timeoutSeconds ? timeoutSeconds * 1000 : undefined,
			});

			let timeoutHandle: NodeJS.Timeout | undefined;

			const onAbort = () => {
				if (child.pid) killProcessTree(child.pid);
			};

			// Abort signal
			if (ctx.signal) {
				if (ctx.signal.aborted) {
					onAbort();
				} else {
					ctx.signal.addEventListener("abort", onAbort, { once: true });
				}
			}

			// Timeout
			if (timeoutSeconds && timeoutSeconds > 0) {
				timeoutHandle = setTimeout(() => {
					timedOut = true;
					if (child.pid) killProcessTree(child.pid);
				}, timeoutSeconds * 1000);
			}

			// Stream output
			const handleData = (data: Buffer) => {
				output.append(data);
				const snapshot = output.snapshot();
				if (snapshot.content && ctx.onUpdate) {
					ctx.onUpdate(snapshot.content);
				}
			};

			child.stdout?.on("data", handleData);
			child.stderr?.on("data", handleData);

			child.on("error", (err: Error) => {
				hasError = true;
				output.finish();
				settle(() => {
					if (timeoutHandle) clearTimeout(timeoutHandle);
					ctx.signal?.removeEventListener("abort", onAbort);
					resolve({
						content: `Sandbox error: ${err.message || "Command failed"}`,
						exitCode: null,
						signal: null,
						status: "failed",
						details: {
							bwrapAvailable: true,
							bwrapPath: bwrap.path,
							profile,
						},
					});
				});
			});

			child.on("close", (code, signal) => {
				output.finish();
				if (timeoutHandle) clearTimeout(timeoutHandle);
				ctx.signal?.removeEventListener("abort", onAbort);

				settle(() => {
					const snapshot = output.snapshot({ persistIfTruncated: true });
					output.closeTempFile().catch(() => {});

					if (hasError) return;

					const truncated = snapshot.content.length >= DEFAULT_MAX_BYTES;
					let content = snapshot.content || "(no output)";

					if (truncated && snapshot.truncation) {
						const trunc = snapshot.truncation;
						const notices: string[] = [];
						if (trunc.truncatedBy === "lines") {
							notices.push(
								`Showing lines ${trunc.totalLines - trunc.outputLines + 1}-${trunc.totalLines} of ${trunc.totalLines}. Full output: ${snapshot.fullOutputPath}`,
							);
						} else {
							notices.push(
								`Showing ${DEFAULT_MAX_BYTES / 1024}KB limit. Full output: ${snapshot.fullOutputPath}`,
							);
						}
						content += `\n\n[${notices.join(". ")}]`;
					}

					let status: SandboxRunResult["status"] = code === 0 ? "completed" : "failed";
					if (timedOut) status = "timed_out";
					else if (ctx.signal?.aborted) status = "aborted";

					resolve({
						content,
						exitCode: code,
						signal,
						status,
						details: {
							bwrapAvailable: true,
							bwrapPath: bwrap.path,
							profile,
							truncation: truncated ? snapshot.truncation : undefined,
							fullOutputPath: snapshot.fullOutputPath,
						},
					});
				});
			});
		});
		return result;
	} finally {
		// Cleanup sandbox directory
		try {
			await fsRm(sandboxTmpdir, { recursive: true, force: true });
		} catch {
			// Cleanup best-effort
		}
	}
}

async function executeFallback(
	command: string,
	cwd: string,
	timeout: number | undefined,
	ctx: Parameters<Tool["execute"]>[1],
): Promise<{
	content: string;
	exitCode: number | null;
	signal: string | null;
	status: SandboxRunResult["status"];
}> {
	const { shell, args: shellArgs } = getShellConfig();
	const shellEnv = getShellEnv();
	const output = new OutputAccumulator({
		tempFilePrefix: "logician-sandbox-fallback",
	});
	let settled = false;
	let timedOut = false;
	let hasError = false;

	const settle = (fn: () => void) => {
		if (!settled) {
			settled = true;
			fn();
		}
	};

	return new Promise<{
		content: string;
		exitCode: number | null;
		signal: string | null;
		status: SandboxRunResult["status"];
	}>((resolve) => {
		const child = spawn(shell, [...shellArgs, command], {
			cwd,
			stdio: ["ignore", "pipe", "pipe"],
			env: shellEnv,
			detached: process.platform !== "win32",
		});

		let timeoutHandle: NodeJS.Timeout | undefined;

		const onAbort = () => {
			if (child.pid) killProcessTree(child.pid);
		};

		if (ctx.signal) {
			if (ctx.signal.aborted) {
				onAbort();
			} else {
				ctx.signal.addEventListener("abort", onAbort, { once: true });
			}
		}

		if (timeout && timeout > 0) {
			timeoutHandle = setTimeout(() => {
				timedOut = true;
				if (child.pid) killProcessTree(child.pid);
			}, timeout * 1000);
		}

		const handleData = (data: Buffer) => {
			output.append(data);
			const snapshot = output.snapshot();
			if (snapshot.content && ctx.onUpdate) {
				ctx.onUpdate(snapshot.content);
			}
		};

		child.stdout?.on("data", handleData);
		child.stderr?.on("data", handleData);

		child.on("error", (err: Error) => {
			hasError = true;
			output.finish();
			settle(() => {
				if (timeoutHandle) clearTimeout(timeoutHandle);
				ctx.signal?.removeEventListener("abort", onAbort);
				resolve({
					content: `Sandbox fallback error: ${err.message || "Command failed"}`,
					exitCode: null,
					signal: null,
					status: "failed",
				});
			});
		});

		child.on("close", (code, signal) => {
			output.finish();
			if (timeoutHandle) clearTimeout(timeoutHandle);
			ctx.signal?.removeEventListener("abort", onAbort);

			settle(() => {
				const snapshot = output.snapshot({ persistIfTruncated: true });
				output.closeTempFile().catch(() => {});

				if (hasError) return;

				const truncated = snapshot.content.length >= DEFAULT_MAX_BYTES;
				let content = snapshot.content || "(no output)";

				if (truncated && snapshot.truncation) {
					const trunc = snapshot.truncation;
					const notices: string[] = [];
					if (trunc.truncatedBy === "lines") {
						notices.push(
							`Showing lines ${trunc.totalLines - trunc.outputLines + 1}-${trunc.totalLines} of ${trunc.totalLines}. Full output: ${snapshot.fullOutputPath}`,
						);
					} else {
						notices.push(
							`Showing ${DEFAULT_MAX_BYTES / 1024}KB limit. Full output: ${snapshot.fullOutputPath}`,
						);
					}
					content += `\n\n[${notices.join(". ")}]`;
				}

				let status: SandboxRunResult["status"] = code === 0 ? "completed" : "failed";
				if (timedOut) status = "timed_out";
				else if (ctx.signal?.aborted) status = "aborted";

				resolve({
					content,
					exitCode: code,
					signal,
					status,
				});
			});
		});
	});
}

// ── Schema ─────────────────────────────────────────────────────────────────────

const sandboxSchema = {
	type: "object",
	properties: {
		command: {
			type: "string",
			description: "Shell command to execute inside the sandbox",
		},
		profile: {
			type: "string",
			enum: ["none", "code", "full"],
			description:
				"Sandbox isolation profile. 'code' = read-only host fs, writable /tmp, no network, no devices. 'full' = adds user namespace + mount namespace.",
			default: "code",
		},
		timeout: {
			type: "number",
			description: "Timeout in seconds (default: 60)",
		},
	},
	required: ["command"],
} as const;

type SandboxArgs = {
	command: string;
	profile?: SandboxProfile;
	timeout?: number;
};

// ── Tool ───────────────────────────────────────────────────────────────────────

export const sandbox: Tool = {
	name: "sandbox",
	executionMode: "sequential",
	label: "Sandbox",
	description:
		"Execute a command inside a Bubblewrap-isolated sandbox. Linux-only; falls back to regular bash when bwrap is unavailable. " +
		"Profiles: none (no isolation), code (read-only host fs, writable /tmp, no network), full (code + namespaces). " +
		`Output truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB.`,
	promptSnippet:
		"Execute shell commands in a Bubblewrap-isolated sandbox with configurable isolation profiles",
	parameters: sandboxSchema,
	prepareArguments: (raw: unknown): Record<string, unknown> => {
		if (typeof raw === "string") return { command: raw };
		if (!raw || typeof raw !== "object") return {};
		const args = raw as Record<string, unknown>;
		const command = args.command ?? args.cmd ?? args.script ?? args.input;
		return {
			...args,
			...(command === undefined
				? {}
				: { command: String(command) }),
		};
	},
	resolveTimeoutMs: (args: Record<string, unknown>) => {
		const timeout = Number((args as SandboxArgs).timeout) || 0;
		return timeout > 0 ? timeout * 1000 + 30_000 : undefined;
	},
	execute: async (
		args: Record<string, unknown>,
		ctx: Parameters<Tool["execute"]>[1],
	): Promise<string | ToolResult> => {
		const cmd = args.command;
		if (!cmd || typeof cmd !== "string") {
			return "Error: command is required.";
		}

		const profile = ((args.profile ?? getDefaultSandboxProfile()) as string) as SandboxProfile;
		const cwd = ctx.cwd || process.cwd();

		// Validate cwd exists
		try {
			await fsAccess(cwd, constants.F_OK);
		} catch {
			return {
				content: `Error: Working directory does not exist: ${cwd}`,
				isError: true,
			};
		}

		const result = await executeSandboxed(
			cmd,
			profile,
			cwd,
			(args.timeout as number) ?? undefined,
			ctx,
		);

		// Append status info
		let content = result.content;
		if (result.status === "timed_out") {
			content += `\n\n[Command timed out after ${(args.timeout as number) ?? 60} seconds]`;
		} else if (result.status === "aborted") {
			content += "\n\n[Command aborted]";
		} else if (
			result.exitCode !== null &&
			result.exitCode !== 0 &&
			result.status !== "failed"
		) {
			content += `\n\n[Command exited with code ${result.exitCode}]`;
		}

		return {
			content,
			details: result.details as unknown as Record<string, unknown>,
		};
	},
};
