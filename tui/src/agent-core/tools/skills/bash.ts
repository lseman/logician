// ── bash tool ──────────────────────────────────────────────────────────────────────
// Execute shell commands with timeout, process tree tracking, and OutputAccumulator.
// Ported from Pi with logician integration.

import { spawn } from "node:child_process";
import { constants, access as fsAccess } from "node:fs/promises";

import type { Tool, ToolResult } from "../../core/types.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
	formatSize,
	type TruncationResult,
} from "./truncate.ts";
import { OutputAccumulator } from "./truncate.ts";
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
	label: "Bash",
	hookAliases: ["Bash"],
	description: `Execute bash commands with timeout. Output is streamed and truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB. Uses process tree tracking for proper cleanup.`,
	promptSnippet: "Execute shell commands in a sandboxed subprocess with timeout",
	promptGuidelines: ["Use bash for file operations like ls, grep, find; use read for file content instead of cat"],
	parameters: bashSchema,
	prepareArguments,
	execute: async (args, ctx): Promise<string | ToolResult> => {
		const { command, timeout } = args as BashArgs;

		if (!command) return "Error: command is required.";

		// Resolve shell
		const { shell, args: shellArgs } = getShellConfig();

		// Resolve cwd
		const cwd = ctx.cwd || process.cwd();
		try {
			await fsAccess(cwd, constants.F_OK);
		} catch {
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
						resolve("Error: Command aborted");
						return;
					}

					// Check for timeout
					if (timedOut) {
						resolve({
							content:
								snapshot.content ||
								`Error: Command timed out after ${timeoutSeconds}s`,
							details: buildBashDetails(snapshot),
						});
						return;
					}

					// Build result
					const { text, details } = formatOutput(snapshot, code, signal);
					if (code === 0) {
						resolve({ content: text, details });
					} else {
						resolve({ content: text, details });
					}
				});
			});
		});
	},
};

function buildBashDetails(snapshot: {
	content: string;
	truncation: TruncationResult;
	fullOutputPath?: string;
}): BashDetails | undefined {
	if (!snapshot.truncation.truncated) return undefined;
	return {
		truncation: {
			truncated: true,
			totalLines: snapshot.truncation.totalLines,
			totalBytes: snapshot.truncation.totalBytes,
			outputLines: snapshot.truncation.outputLines,
			outputBytes: snapshot.truncation.outputBytes,
			lastLinePartial: snapshot.truncation.lastLinePartial,
			truncatedBy: snapshot.truncation.truncatedBy ?? undefined,
			maxLines: snapshot.truncation.maxLines,
			maxBytes: snapshot.truncation.maxBytes,
		},
		fullOutputPath: snapshot.fullOutputPath,
	};
}

function formatOutput(
	snapshot: {
		content: string;
		truncation: TruncationResult;
		fullOutputPath?: string;
	},
	_code: number | null,
	_signal: string | null,
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
				`Showing last ${formatSize(truncation.outputBytes)} of line ${endLine} (line is ${formatSize(0)}). Full output: ${fullPath}`,
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

	return { text, details };
}
