// ── bash tool ──────────────────────────────────────────────────────────────────────
// Execute shell commands with timeout, process tree tracking, and OutputAccumulator.
// Ported from Pi with logician integration.

import { spawn } from "node:child_process";
import { constants, access as fsAccess } from "node:fs/promises";

import type { Tool, ToolResult } from "@logician/agent-core/core/types.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	OutputAccumulator,
	type TruncationResult,
} from "./truncate.ts";
import {
	getShellConfig,
	getShellEnv,
	killProcessTree,
	trackDetachedChildPid,
	untrackDetachedChildPid,
} from "./shell.ts";

const bashSchema = {
	type: "object",
	properties: {
		command: { type: "string", description: "Bash command to execute" },
		timeout: {
			type: "number",
			description: "Timeout in seconds (optional, no default timeout)",
		},
	},
	required: ["command"],
} as const;

type BashArgs = {
	command: string;
	timeout?: number;
};

export interface BashDetails {
	truncation?: {
		truncated: boolean;
		totalLines: number;
		totalBytes: number;
		outputLines: number;
		outputBytes: number;
		lastLinePartial?: boolean;
		truncatedBy?: "lines" | "bytes";
		maxLines: number;
		maxBytes: number;
	};
	fullOutputPath?: string;
	[key: string]: unknown;
}

function prepareArguments(raw: unknown): Record<string, unknown> {
	if (typeof raw === "string") return { command: raw };
	if (!raw || typeof raw !== "object") return {};
	const args = raw as Record<string, unknown>;
	const command = args.command ?? args.cmd ?? args.script ?? args.input ?? "";
	return {
		...args,
		command: String(command),
	};
}

// ── Update throttling ──────────────────────────────────────────────────────────
// Throttle TUI updates to prevent flooding the interface.

function makeUpdateThrottler(): (callback: () => void) => void {
	let lastUpdate = 0;
	const THROTTLE_MS = 100;

	return function throttledUpdate(callback: () => void) {
		const now = Date.now();
		if (now - lastUpdate < THROTTLE_MS) {
			// Debounce: skip this update
			return;
		}
		lastUpdate = now;
		callback();
	};
}

export const bash: Tool = {
	name: "bash",
	executionMode: "sequential",
	label: "Bash",
	hookAliases: ["Bash"],
	description: `Execute bash commands with timeout. Output is streamed and truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB. Uses process tree tracking for proper cleanup.`,
	promptSnippet:
		"Execute shell commands in a managed subprocess with timeout and approval policy",
	promptGuidelines: [
		"Use bash for file operations like ls, grep, find; use read for file content instead of cat",
	],
	parameters: bashSchema,
	prepareArguments,
	// The registry enforces a default execution timeout; when the model passes
	// an explicit timeout, allow it plus grace for the tool's own kill+cleanup.
	resolveTimeoutMs: (args) => {
		const timeout = Number(args.timeout);
		return timeout > 0 ? timeout * 1000 + 30_000 : undefined;
	},
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const { command, timeout } = args as BashArgs;

		if (!command) return "Error: command is required.";

		// Resolve shell
		const { shell, args: shellArgs } = getShellConfig();

		// Resolve cwd
		const cwd = ctx.cwd || process.cwd();
		try {
			await fsAccess(cwd, constants.F_OK);
		} catch (e: unknown) {
			return `Error: Working directory does not exist: ${cwd}`;
		}

		const shellEnv = getShellEnv();
		const output = new OutputAccumulator({ tempFilePrefix: "logician-bash" });
		const timeoutSeconds = timeout;
		const throttledUpdate = makeUpdateThrottler();

		return new Promise<string | ToolResult>((resolve) => {
			let settled = false;
			const settle = (fn: () => void) => {
				if (!settled) {
					settled = true;
					fn();
				}
			};

			const child = spawn(shell, [...shellArgs, command], {
				cwd,
				stdio: ["ignore", "pipe", "pipe"],
				env: shellEnv,
				detached: process.platform !== "win32",
			});

			// Track detached child for cleanup
			if (child.pid) {
				if (process.platform !== "win32") {
					trackDetachedChildPid(child.pid);
				}
			}

			let timedOut = false;
			let timeoutHandle: NodeJS.Timeout | undefined;
			let hasError = false;

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
				// Stream updates to TUI (throttled)
				const snapshot = output.snapshot();
				if (snapshot.content && ctx.onUpdate) {
					throttledUpdate(() => {
						ctx.onUpdate?.(snapshot.content);
					});
				}
			};

			child.stdout?.on("data", handleData);
			child.stderr?.on("data", handleData);

			// Handle errors
			child.on("error", (err) => {
				hasError = true;
				output.finish();
				if (child.pid) untrackDetachedChildPid(child.pid);
				settle(() => {
					if (timeoutHandle) clearTimeout(timeoutHandle);
					ctx.signal?.removeEventListener("abort", onAbort);
					resolve(`Error: ${err.message || "Command failed"}`);
				});
			});

			// Handle exit
			child.on("close", (code, signal) => {
				output.finish();
				if (child.pid) untrackDetachedChildPid(child.pid);
				if (timeoutHandle) clearTimeout(timeoutHandle);
				ctx.signal?.removeEventListener("abort", onAbort);

				settle(() => {
					const snapshot = output.snapshot({ persistIfTruncated: true });
					output.closeTempFile().catch(() => {});

					// If we already had an error (e.g. abort during close), skip
					if (hasError) return;

					// Check for abort
					if (ctx.signal?.aborted) {
						const { text, details } = formatOutput(
							snapshot,
							code,
							signal,
							output.getLastLineBytes(),
						);
						resolve({
							content: appendStatus(
								text === "(no output)" ? "" : text,
								"Command aborted",
							),
							details,
						});
						return;
					}

					// Check for timeout
					if (timedOut) {
						const { text, details } = formatOutput(
							snapshot,
							code,
							signal,
							output.getLastLineBytes(),
						);
						resolve({
							content: appendStatus(
								text === "(no output)" ? "" : text,
								`Command timed out after ${timeoutSeconds} seconds`,
							),
							details,
						});
						return;
					}

					// Build result
					const { text, details } = formatOutput(
						snapshot,
						code,
						signal,
						output.getLastLineBytes(),
					);
					const result = { content: text, details };
					resolve(result);
				});
			});
		});
	},
};

function formatOutput(
	snapshot: {
		content: string;
		truncation: TruncationResult;
		fullOutputPath?: string;
	},
	code: number | null,
	signal: string | null,
	lastLineBytes = 0,
): { text: string; details: BashDetails | undefined } {
	const truncation = snapshot.truncation;
	const emptyText = "(no output)";
	let text = snapshot.content || emptyText;
	let details: BashDetails | undefined;

	if (truncation.truncated) {
		const startLine = truncation.totalLines - truncation.outputLines + 1;
		const endLine = truncation.totalLines;
		const fullPath = snapshot.fullOutputPath;

		const notices: string[] = [];
		if (truncation.lastLinePartial) {
			notices.push(
				`Showing last ${formatSize(truncation.outputBytes)} of line ${endLine} (line is ${formatSize(lastLineBytes)}). Full output: ${fullPath}`,
			);
		} else if (truncation.truncatedBy === "lines") {
			notices.push(
				`Showing lines ${startLine}-${endLine} of ${truncation.totalLines}. Full output: ${fullPath}`,
			);
		} else {
			notices.push(
				`Showing lines ${startLine}-${endLine} of ${truncation.totalLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Full output: ${fullPath}`,
			);
		}

		text += `\n\n[${notices.join(". ")}]`;
		details = {
			truncation: {
				truncated: true,
				totalLines: truncation.totalLines,
				totalBytes: truncation.totalBytes,
				outputLines: truncation.outputLines,
				outputBytes: truncation.outputBytes,
				lastLinePartial: truncation.lastLinePartial,
				truncatedBy: truncation.truncatedBy ?? undefined,
				maxLines: truncation.maxLines,
				maxBytes: truncation.maxBytes,
			},
			fullOutputPath: fullPath,
		};
	}

	if (code !== 0 && code !== null) {
		text = appendStatus(text, `Command exited with code ${code}`);
	} else if (code === null && signal) {
		text = appendStatus(text, `Command terminated by signal ${signal}`);
	}

	return { text, details };
}

function appendStatus(text: string, status: string): string {
	return `${text ? `${text}\n\n` : ""}${status}`;
}
